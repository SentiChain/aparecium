"""Split a raw JSONL into train/val/test sets with ratios.

Usage:
  python -m aparecium.scripts.split_jsonl \
    --input raw_10k.jsonl \
    --out_dir data/shards \
    --train 0.9 --val 0.05 --test 0.05

Outputs:
  data/shards/train.jsonl, val.jsonl, test.jsonl
"""

import os, random, argparse


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--train", type=float, default=0.9)
    ap.add_argument("--val", type=float, default=0.05)
    ap.add_argument("--test", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1234)
    return ap.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.input, "r", encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]

    random.shuffle(lines)
    n = len(lines)
    n_train = int(n * args.train)
    n_val = int(n * args.val)
    n_test = n - n_train - n_val

    splits = {
        "train": lines[:n_train],
        "val": lines[n_train : n_train + n_val],
        "test": lines[n_train + n_val :],
    }

    for name, data in splits.items():
        path = os.path.join(args.out_dir, f"{name}.jsonl")
        with open(path, "w", encoding="utf-8") as fout:
            for line in data:
                fout.write(line)
        print(f"Wrote {name}: {len(data)} -> {path}")


if __name__ == "__main__":
    main()
