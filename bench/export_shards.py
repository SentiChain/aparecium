"""Export a benchmark training set to the JSONL shard format the v2 trainers read.

Usage:
    python -m bench.export_shards --name stage1

Writes data/bench/train/<name>_shards/<name>_NNN_emb.jsonl with
{"text", "embedding"} per line (1,000 lines per shard, since the v2 dataset
class rescans a shard file on every item).
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from .common import DATA_DIR, read_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--shard_size", type=int, default=1000)
    args = ap.parse_args()

    rows = list(read_jsonl(DATA_DIR / "train" / f"{args.name}.jsonl"))
    vecs = np.load(DATA_DIR / "train" / f"{args.name}.mpnet.npy")
    assert len(rows) == len(vecs)
    out_dir = DATA_DIR / "train" / f"{args.name}_shards"
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*_emb.jsonl"):
        old.unlink()
    for k, start in enumerate(range(0, len(rows), args.shard_size)):
        with open(out_dir / f"{args.name}_{k:03d}_emb.jsonl", "w", encoding="utf-8") as f:
            for r, v in zip(rows[start:start + args.shard_size], vecs[start:start + args.shard_size]):
                f.write(json.dumps({"text": r["text"], "embedding": v.tolist()}, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows to {out_dir}")


if __name__ == "__main__":
    main()
