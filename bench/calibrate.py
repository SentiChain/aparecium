"""Calibrate what cosine similarity means "same meaning" for each encoder.

Uses STS-B test (human 0-5 similarity) and PAWS (word-scrambled pairs with
near-identical wording but often different meaning).

Usage:
    python -m bench.calibrate
Writes bench/manifests/calibration.json (numbers only).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata, spearmanr

from .common import DATA_DIR, INDEP_ID, MANIFEST_DIR, MPNET_ID, read_jsonl, write_json
from .embed import embed

SAME_MEANING_SCORE = 4.0  # STS-B: 4 = "mostly equivalent", 5 = "completely equivalent"


def pair_cos(rows, model_id):
    a = embed([r["a"] for r in rows], model_id)
    b = embed([r["b"] for r in rows], model_id)
    return (a * b).sum(axis=1)


def best_threshold(cos: np.ndarray, positive: np.ndarray) -> dict:
    """Threshold maximizing F1 for predicting human 'same meaning' from cosine."""
    best = {"f1": -1.0}
    for t in np.unique(np.round(cos, 3)):
        pred = cos >= t
        tp = float((pred & positive).sum())
        prec = tp / max(1.0, float(pred.sum()))
        rec = tp / max(1.0, float(positive.sum()))
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        if f1 > best["f1"]:
            best = {"threshold": float(t), "f1": f1, "precision": prec, "recall": rec}
    return best


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Probability a random positive pair scores above a random negative pair."""
    ranks = rankdata(np.concatenate([pos, neg]))  # average ranks for ties
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    stsb = list(read_jsonl(DATA_DIR / "calibration" / "stsb_test.jsonl"))
    paws = list(read_jsonl(DATA_DIR / "calibration" / "paws_test.jsonl"))
    score = np.array([r["score5"] for r in stsb])
    label = np.array([r["label"] for r in paws])
    out = {"same_meaning_score": SAME_MEANING_SCORE, "encoders": {}}
    for tag, model_id in (("mpnet", MPNET_ID), ("bge", INDEP_ID)):
        c = pair_cos(stsb, model_id)
        p = pair_cos(paws, model_id)
        bins = {}
        for lo, hi in ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5.01)):
            m = (score >= lo) & (score < hi)
            bins[f"{lo}-{min(hi, 5)}"] = {"n": int(m.sum()), "median_cos": float(np.median(c[m]))}
        spearman = float(spearmanr(c, score).statistic)
        out["encoders"][tag] = {
            "model": model_id,
            "stsb_spearman": spearman,
            "stsb_median_cos_by_human_score": bins,
            "same_meaning_threshold": best_threshold(c, score >= SAME_MEANING_SCORE),
            "paws": {
                "median_cos_paraphrase": float(np.median(p[label == 1])),
                "median_cos_not_paraphrase": float(np.median(p[label == 0])),
                "auc": auc(p[label == 1], p[label == 0]),
            },
        }
    write_json(MANIFEST_DIR / "calibration.json", out)
    for tag, e in out["encoders"].items():
        t = e["same_meaning_threshold"]
        print(f"{tag}: spearman {e['stsb_spearman']:.3f} | same-meaning cos >= {t['threshold']:.3f} "
              f"(F1 {t['f1']:.2f}) | PAWS median cos para {e['paws']['median_cos_paraphrase']:.3f} "
              f"vs non-para {e['paws']['median_cos_not_paraphrase']:.3f}, AUC {e['paws']['auc']:.3f}")


if __name__ == "__main__":
    main()
