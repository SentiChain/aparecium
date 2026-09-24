"""Small, resumable, reproducible sampling from Hugging Face datasets.

Instead of downloading whole datasets, read a seeded random selection of
Parquet row groups (1,000 rows each on the Hub's Parquet export), fetching only
the needed columns over HTTP range requests. Extracted text units are appended
to a per-source pool file after each row group, so an interrupted run resumes
where it stopped and larger stages reuse what smaller stages already fetched.

A pool is pinned to the Parquet-export commit it was first built from; later
runs read that same commit even if the Hub re-converts the dataset.
"""

from __future__ import annotations

import fcntl
import json
import os
import random
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem

from .common import DATA_DIR, SEED

POOL_DIR = DATA_DIR / "pool"
Group = Tuple[int, int]  # (file index, row-group index)


@lru_cache(maxsize=None)
def live_export_sha(repo: str) -> str:
    refs = HfApi().list_repo_refs(repo, repo_type="dataset")
    return next(r.target_commit for r in refs.converts if r.name == "parquet")


class GroupSampler:
    def __init__(self, repo: str, config: Optional[str], split: str, columns: List[str], sha: str):
        self.repo = repo
        self.config = config or "default"
        self.split = split
        self.sha = sha
        self.columns = columns
        self.fs = HfFileSystem()
        root = f"datasets/{repo}@{sha}/{self.config}"
        # Big datasets are exported only partially, into "partial-<split>".
        dirs = [d for d in self.fs.ls(root, detail=False)
                if d.rsplit("/", 1)[-1] in (split, f"partial-{split}")]
        self.files = sorted(f for d in dirs for f in self.fs.ls(d, detail=False) if f.endswith(".parquet"))
        if not self.files:
            raise ValueError(f"no parquet export for {repo}@{sha} {self.config}/{split}")
        self._pf: Dict[int, pq.ParquetFile] = {}

    def _file(self, i: int) -> pq.ParquetFile:
        if i not in self._pf:
            self._pf[i] = pq.ParquetFile(self.fs.open(self.files[i]))
        return self._pf[i]

    def names(self) -> List[str]:
        return [f.split(f"@{self.sha}/", 1)[1] for f in self.files]

    def n_groups(self) -> List[int]:
        return [self._file(i).metadata.num_row_groups for i in range(len(self.files))]

    def order(self, n_groups: List[int]) -> List[Group]:
        groups = [(i, g) for i, n in enumerate(n_groups) for g in range(n)]
        random.Random(f"{SEED}|{self.repo}|{self.config}|{self.split}").shuffle(groups)
        return groups

    def read(self, g: Group) -> List[Dict]:
        pf = self._file(g[0])
        cols = [c for c in self.columns if c in pf.schema_arrow.names]
        return pf.read_row_group(g[1], columns=cols).to_pylist()


@contextmanager
def _locked(key: str):
    POOL_DIR.mkdir(parents=True, exist_ok=True)
    with open(POOL_DIR / f"{key}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _read_pool(path: Path) -> List[Dict]:
    """Read pool lines, repairing a torn final line left by a hard kill."""
    if not path.exists():
        return []
    lines = [x for x in path.read_text(encoding="utf-8").split("\n") if x.strip()]
    rows = []
    for k, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if k != len(lines) - 1:  # only the final line may be torn
                raise
    if len(rows) != len(lines):
        path.write_text("".join(x + "\n" for x in lines[:-1]), encoding="utf-8")
    return rows


def pool(src, n_units: int, per_group: Optional[int], config: Optional[str] = None,
         key: Optional[str] = None) -> List[Dict]:
    """Return the first n_units extracted units of a source, fetching more row groups if needed.

    per_group caps how many units one row group contributes (None = all), which
    spreads a sample across many parts of the dataset.
    """
    key = key or src.key
    with _locked(key):
        return _pool(src, n_units, per_group, config or src.config, key)


def _pool(src, n_units, per_group, config, key):
    path, meta_path = POOL_DIR / f"{key}.jsonl", POOL_DIR / f"{key}.meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else None
    settings = {"repo": src.repo, "config": config or "default", "split": src.split,
                "columns": list(src.columns), "per_group": per_group}
    if meta and any(meta.get(k) != v for k, v in settings.items()):
        raise RuntimeError(f"pool {key} was built with different settings; delete {path} to rebuild")
    sampler = GroupSampler(src.repo, config, src.split, src.columns,
                           meta["sha"] if meta else live_export_sha(src.repo))
    if meta and meta["files"] != sampler.names():
        raise RuntimeError(f"pool {key}: file set changed under {meta['sha']}; delete {path} to rebuild")
    if not meta:
        meta = {**settings, "sha": sampler.sha, "files": sampler.names(),
                "n_groups": sampler.n_groups(), "groups_done": [], "exhausted": False}
    done = {tuple(g) for g in meta["groups_done"]}
    units, ids = [], set()
    for u in _read_pool(path):
        if tuple(u["group"]) in done and u["id"] not in ids:
            ids.add(u["id"])
            units.append(u)
    if len(units) >= n_units or meta["exhausted"]:
        return units[:n_units]

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for g in sampler.order(meta["n_groups"]):
            if g in done:
                continue
            rows = sampler.read(g)
            idx = list(range(len(rows)))
            random.Random(f"{SEED}|{key}|{g}").shuffle(idx)
            got = []
            for j in idx:
                u = src.extract(rows[j], f"{g[0]}:{g[1]}:{j}")
                if u and u["id"] not in ids:
                    u["group"] = list(g)
                    ids.add(u["id"])
                    got.append(u)
                if per_group and len(got) >= per_group:
                    break
            for u in got:
                f.write(json.dumps(u, ensure_ascii=False) + "\n")
            f.flush()
            units += got
            done.add(g)
            meta["groups_done"].append(list(g))
            _write_meta(meta_path, meta)
            print(f"    {key}: {len(units)}/{n_units} units after {len(done)} row groups", flush=True)
            if len(units) >= n_units:
                break
        else:
            meta["exhausted"] = True
            _write_meta(meta_path, meta)
    return units[:n_units]


def pool_shas() -> Dict[str, str]:
    """Export commit each existing pool is pinned to."""
    out = {}
    for p in sorted(POOL_DIR.glob("*.meta.json")):
        m = json.loads(p.read_text())
        out[p.name[: -len(".meta.json")]] = f"{m['repo']}@{m['sha']}"
    return out


def _write_meta(path: Path, meta: Dict) -> None:
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(meta, indent=1))
    tmp.replace(path)
