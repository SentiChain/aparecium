import argparse
import json
import math
from typing import List, Dict, Any

import numpy as np


def count_lines(path: str) -> int:
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for _ in f:
            n += 1
    return n


def read_first_jsonl(path: str, k: int) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                out.append(json.loads(s))
            except Exception:
                continue
            if len(out) >= k:
                break
    return out


def cosine_similarity_matrix(X: np.ndarray) -> np.ndarray:
    # X: (N, D)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return Xn @ Xn.T


def summarize(args):
    # Counts
    total_raw = count_lines(args.raw)
    total_emb = count_lines(args.emb)

    # Alignment check on first M raw/emb
    M = min(args.check_first, total_raw, total_emb)
    raw_head = read_first_jsonl(args.raw, M)
    emb_head = read_first_jsonl(args.emb, M)
    alignment_ok = all(
        (r.get("text", "").strip() == e.get("text", "").strip())
        for r, e in zip(raw_head, emb_head)
    )

    # Sample K for deeper checks
    K = min(args.sample, len(emb_head))
    emb_sample = emb_head[:K]
    texts = [e.get("text", "") for e in emb_sample]
    embs = []
    lengths_ok = True
    norms = []
    for e in emb_sample:
        arr = np.asarray(e.get("embedding", []), dtype=np.float32)
        embs.append(arr)
        lengths_ok = lengths_ok and (arr.shape == (768,))
        nrm = float(np.linalg.norm(arr))
        norms.append(nrm)

    embs_mat = np.stack(embs, axis=0) if embs else np.zeros((0, 768), dtype=np.float32)
    cos = (
        cosine_similarity_matrix(embs_mat)
        if embs_mat.shape[0] > 0
        else np.zeros((0, 0), dtype=np.float32)
    )
    # exclude diagonal for min/max stats
    cos_vals = (
        cos[np.triu_indices_from(cos, k=1)]
        if cos.size > 0
        else np.array([], dtype=np.float32)
    )
    cos_min = float(np.min(cos_vals)) if cos_vals.size else float("nan")
    cos_max = float(np.max(cos_vals)) if cos_vals.size else float("nan")
    cos_median = float(np.median(cos_vals)) if cos_vals.size else float("nan")

    # Check exact duplicates within sample
    duplicates = 0
    for i in range(len(embs)):
        for j in range(i + 1, len(embs)):
            if np.array_equal(embs[i], embs[j]):
                duplicates += 1

    # Sample preview
    preview = []
    for i in range(K):
        t = texts[i]
        snippet = (
            (t[:160] + ("…" if len(t) > 160 else "")) if isinstance(t, str) else ""
        )
        preview.append(
            {
                "idx": i,
                "text": snippet,
                "emb_len": int(embs[i].size),
                "l2_norm": round(norms[i], 6),
            }
        )

    report = {
        "raw_total": total_raw,
        "emb_total": total_emb,
        "alignment_ok_first": alignment_ok,
        "sample_size": K,
        "emb_dim_768_all": bool(lengths_ok),
        "l2_norm_stats": {
            "min": round(float(np.min(norms)), 6) if norms else None,
            "max": round(float(np.max(norms)), 6) if norms else None,
            "median": round(float(np.median(norms)), 6) if norms else None,
        },
        "pairwise_cos": {
            "min": round(cos_min, 6) if not math.isnan(cos_min) else None,
            "max": round(cos_max, 6) if not math.isnan(cos_max) else None,
            "median": round(cos_median, 6) if not math.isnan(cos_median) else None,
        },
        "exact_duplicate_pairs_in_sample": duplicates,
        "preview": preview,
    }
    # ensure_ascii=True avoids Windows console encoding issues
    print(json.dumps(report, ensure_ascii=True))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=str, required=True, help="posts_raw.jsonl path")
    ap.add_argument("--emb", type=str, required=True, help="posts_emb.jsonl path")
    ap.add_argument("--sample", type=int, default=10)
    ap.add_argument("--check_first", type=int, default=50)
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summarize(args)
