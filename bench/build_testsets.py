"""Build the frozen Stage-0 test sets (A-D) and calibration pairs (E).

Usage:
    python -m bench.build_testsets

Text is sampled from seeded random Parquet row groups (see fetch.py); each
source pool is pinned to the export commit recorded in
bench/manifests/testsets.json, along with encoder versions and checksums of the
frozen vectors. Text goes to data/bench/testsets/ (gitignored).
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from typing import Dict, List, Set

from .common import DATA_DIR, INDEP_ID, MANIFEST_DIR, MPNET_ID, SEED, clean_text, stable_int, text_hash, write_json, write_jsonl
from .embed import array_sha256, cached, encoder_info
from .fetch import pool, pool_shas
from .sources import BBC_MONTHS, CALIBRATION_SOURCES, TEST_SOURCES, TRAIN_SOURCES

A_POOL = 500  # set A is sampled from the first A_POOL units of each training pool
TRAIN_PER_GROUP = 100  # training pools take at most this many units per row group


def pick(units: List[Dict], n: int, seen: Set[str], src, shuffle: bool = False) -> List[Dict]:
    order = list(units)
    if shuffle:
        random.Random(f"{SEED}|pick|{src.key}").shuffle(order)
    out = []
    for u in order:
        if len(out) >= n:
            break
        h = text_hash(u["text"])
        if h in seen:
            continue
        seen.add(h)
        out.append({"id": u["id"], "text": u["text"], "source": src.key, "license": src.license,
                    "hash": h, **({"day": u["day"]} if "day" in u else {})})
    print(f"  {src.key}: {len(out)}/{n}", flush=True)
    return out


def build_bbc_days(seen: Set[str], per_day_cap: int) -> List[Dict]:
    src = TEST_SOURCES["bbc_2025"]
    chosen = []
    for month in BBC_MONTHS:
        units = pool(src, 10**9, per_group=None, config=month, key=f"bbc_{month}")
        by_day = defaultdict(list)
        for u in units:
            if u.get("day", "").startswith(month):
                by_day[u["day"]].append(u)
        full = sorted(d for d, items in by_day.items() if len(items) >= 30)
        day = min(full, key=lambda d: stable_int(d, SEED))
        items = sorted(by_day[day], key=lambda u: stable_int(u["id"], SEED))
        chosen += pick(items, per_day_cap, seen, src)
        print(f"    bbc {month}: day {day} ({len(full)} full days available)", flush=True)
    return chosen


def build_calibration(max_paws: int) -> Dict[str, List[Dict]]:
    stsb = pool(CALIBRATION_SOURCES["stsb"], 10**9, per_group=None)
    paws = pool(CALIBRATION_SOURCES["paws"], 10**9, per_group=None)
    paws = sorted(paws, key=lambda r: stable_int(r["id"], SEED))[:max_paws]
    return {
        "stsb_test": [{"id": r["id"], "a": clean_text(r["sentence1"]), "b": clean_text(r["sentence2"]),
                       "score5": round(float(r["score"]) * 5, 3)} for r in stsb],
        "paws_test": [{"id": r["id"], "a": clean_text(r["sentence1"]), "b": clean_text(r["sentence2"]),
                       "label": int(r["label"])} for r in paws],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a_per_source", type=int, default=50)
    ap.add_argument("--b_per_source", type=int, default=150)
    ap.add_argument("--c_per_day", type=int, default=55)
    ap.add_argument("--d_per_source", type=int, default=100)
    ap.add_argument("--max_paws", type=int, default=2000)
    args = ap.parse_args()

    seen: Set[str] = set()
    sets: Dict[str, List[Dict]] = {}

    print("A: held-out slice of each training source", flush=True)
    sets["A_heldout"] = [
        u for src in TRAIN_SOURCES.values()
        for u in pick(pool(src, A_POOL, TRAIN_PER_GROUP), args.a_per_source, seen, src, shuffle=True)
    ]
    print("B: news from outlets not in training", flush=True)
    sets["B_unseen_outlets"] = [
        u for key in ("cnn_dm", "xsum")
        for u in pick(pool(TEST_SOURCES[key], args.b_per_source + 30, 40), args.b_per_source,
                      seen, TEST_SOURCES[key])
    ]
    print("C: BBC Feb-Jun 2025, one full day per month", flush=True)
    sets["C_bbc_2025_days"] = build_bbc_days(seen, args.c_per_day)
    print("D: other registers (tweets, Q&A titles, crypto headlines)", flush=True)
    sets["D_other_registers"] = [
        u for key, per_group in (("tweets", 40), ("stackexchange", 20), ("crypto", 20))
        for u in pick(pool(TEST_SOURCES[key], args.d_per_source + 30, per_group), args.d_per_source,
                      seen, TEST_SOURCES[key])
    ]

    manifest = {"seed": SEED, "pools": {}, "encoders": encoder_info(), "sets": {}}
    for name, rows in sets.items():
        for r in rows:
            r["set"] = name
        path = DATA_DIR / "testsets" / f"{name}.jsonl"
        write_jsonl(path, rows)
        texts = [r["text"] for r in rows]
        per_source = defaultdict(int)
        for r in rows:
            per_source[r["source"]] += 1
        manifest["sets"][name] = {
            "count": len(rows),
            "per_source": dict(per_source),
            "vectors_sha256": {tag: array_sha256(cached(path, texts, model))
                               for tag, model in (("mpnet", MPNET_ID), ("bge", INDEP_ID))},
            "items": [{"id": r["id"], "hash": r["hash"], "source": r["source"],
                       **({"day": r["day"]} if "day" in r else {})} for r in rows],
        }

    print("E: calibration pairs", flush=True)
    for name, rows in build_calibration(args.max_paws).items():
        write_jsonl(DATA_DIR / "calibration" / f"{name}.jsonl", rows)
        manifest["sets"][f"E_{name}"] = {"count": len(rows)}

    manifest["pools"] = pool_shas()
    write_json(MANIFEST_DIR / "testsets.json", manifest)
    print({k: v["count"] for k, v in manifest["sets"].items()})


if __name__ == "__main__":
    main()
