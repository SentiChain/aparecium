"""Split a large JSONL into multiple shard JSONLs of size N."""

import os, argparse


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--shard_size", type=int, default=50000)
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    i = 0
    shard = 0
    fout = None
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            if i % args.shard_size == 0:
                if fout:
                    fout.close()
                path = os.path.join(args.out_dir, f"shard_{shard:05d}.jsonl")
                fout = open(path, "w", encoding="utf-8")
                shard += 1
            fout.write(line)
            i += 1
    if fout:
        fout.close()
    print(f"Wrote {shard} shards to {args.out_dir}")


if __name__ == "__main__":
    main()
