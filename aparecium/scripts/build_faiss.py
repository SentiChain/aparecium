"""Build a FAISS index over embeddings stored in JSONL shards (for hard negative mining)."""

import os, json, glob, argparse, numpy as np


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards_dir", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    return ap.parse_args()


def main():
    args = parse_args()
    files = sorted(glob.glob(os.path.join(args.shards_dir, "*.jsonl")))
    vecs = []
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                vecs.append(rec["embedding"])
    vecs = np.array(vecs, dtype=np.float32)
    try:
        import faiss
    except Exception as e:
        print("faiss not installed; exiting")
        return
    index = faiss.IndexFlatIP(vecs.shape[1])
    faiss.normalize_L2(vecs)
    index.add(vecs)
    faiss.write_index(index, args.out)
    print("Saved index to", args.out)


if __name__ == "__main__":
    main()
