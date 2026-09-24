"""Build a training set of a given size from the open-licensed training sources.

Usage:
    python -m bench.build_trainset --size 10000 --name stage1

Items are drawn in order from the same per-source pools as test set A (see
fetch.py), so larger stages are supersets of smaller ones (up to the leakage
filter). A candidate is excluded if it matches any test item by normalized-text
hash, or if it is "same meaning" as any test item by the benchmark's own
calibrated bar on either encoder (MPNet or the independent encoder).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

from .build_testsets import A_POOL, TRAIN_PER_GROUP
from .common import DATA_DIR, INDEP_ID, MANIFEST_DIR, MPNET_ID, read_jsonl, text_hash, write_json, write_jsonl
from .embed import array_sha256, cached, embed
from .fetch import pool, pool_shas
from .sources import STAGE_QUOTA_WEIGHTS, TRAIN_SOURCES


def load_test_items():
    hashes, vecs = set(), {MPNET_ID: [], INDEP_ID: []}
    for path in sorted((DATA_DIR / "testsets").glob("*.jsonl")):
        rows = list(read_jsonl(path))
        hashes |= {r["hash"] for r in rows}
        for model in vecs:
            vecs[model].append(cached(path, [r["text"] for r in rows], model))
    return hashes, {m: np.concatenate(v, axis=0) for m, v in vecs.items()}


def max_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros(len(a), dtype=np.float32)
    for s in range(0, len(a), 4096):
        out[s:s + 4096] = (a[s:s + 4096] @ b.T).max(axis=1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, required=True)
    ap.add_argument("--name", type=str, required=True)
    ap.add_argument("--overdraw", type=float, default=1.25)
    args = ap.parse_args()

    calib = json.loads((MANIFEST_DIR / "calibration.json").read_text())["encoders"]
    thresholds = {MPNET_ID: calib["mpnet"]["same_meaning_threshold"]["threshold"],
                  INDEP_ID: calib["bge"]["same_meaning_threshold"]["threshold"]}
    test_hashes, test_vecs = load_test_items()
    seen = set(test_hashes)
    quota = {k: round(args.size * w) for k, w in STAGE_QUOTA_WEIGHTS.items()}
    rows = []
    for key, n in quota.items():
        src = TRAIN_SOURCES[key]
        want = int(n * args.overdraw) + 10
        got = 0
        # Over-fetch by the A slice so held-out items never eat into the quota.
        for u in pool(src, want + A_POOL, TRAIN_PER_GROUP):
            if got >= want:
                break
            h = text_hash(u["text"])
            if h in seen:
                continue
            seen.add(h)
            rows.append({"id": u["id"], "text": u["text"], "source": key,
                         "license": src.license, "hash": h})
            got += 1
        print(f"  {key}: drew {got} (quota {n})", flush=True)

    texts = [r["text"] for r in rows]
    vecs = embed(texts, MPNET_ID)
    leak = {m: max_sim(embed(texts, m) if m != MPNET_ID else vecs, test_vecs[m]) >= t
            for m, t in thresholds.items()}
    keep, per_source, dropped = [], defaultdict(int), defaultdict(lambda: defaultdict(int))
    for i, r in enumerate(rows):
        hits = [m for m in thresholds if leak[m][i]]
        if hits:
            for m in hits:
                dropped[r["source"]]["mpnet" if m == MPNET_ID else "bge"] += 1
            continue
        if per_source[r["source"]] >= quota[r["source"]]:
            continue
        per_source[r["source"]] += 1
        keep.append(i)

    out_path = DATA_DIR / "train" / f"{args.name}.jsonl"
    write_jsonl(out_path, [rows[i] for i in keep])
    kept_vecs = vecs[keep].astype(np.float32)
    np.save(out_path.with_suffix(".mpnet.npy"), kept_vecs)
    short = [k for k, v in quota.items() if per_source[k] < v]
    write_json(MANIFEST_DIR / f"train_{args.name}.json", {
        "name": args.name,
        "count": len(keep),
        "quota": quota,
        "per_source": dict(per_source),
        "under_quota": short,
        "leakage_filter": {"same_meaning_thresholds": {("mpnet" if m == MPNET_ID else "bge"): t
                                                       for m, t in thresholds.items()},
                           "dropped": {k: dict(v) for k, v in dropped.items()}},
        "licenses": {k: TRAIN_SOURCES[k].license for k in quota},
        "pools": {k: v for k, v in pool_shas().items() if k in quota},
        "vectors_sha256": array_sha256(kept_vecs),
        "items": [{"id": rows[i]["id"], "hash": rows[i]["hash"], "source": rows[i]["source"]} for i in keep],
    })
    print(f"saved {len(keep)} items -> {out_path} | per source {dict(per_source)} | "
          f"leakage-dropped {({k: dict(v) for k, v in dropped.items()})} | under quota {short}")


if __name__ == "__main__":
    main()
