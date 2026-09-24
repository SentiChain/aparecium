"""Score saved reconstructions on the frozen test sets.

Usage:
    python -m bench.score --systems s1 nn_stage1 --report stage0

Writes numbers only (no dataset text) to bench/results/<report>.json/.md and a
few qualitative examples to data/bench/reports/<report>_examples.md (gitignored).
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict

import numpy as np

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from .common import DATA_DIR, INDEP_ID, MANIFEST_DIR, MPNET_ID, REPO_ROOT, gpt2_len, in_subset, norm_key, read_jsonl, stable_int, write_json
from .embed import cached, embed

RESULTS_DIR = REPO_ROOT / "bench" / "results"
SET_LABELS = {
    "A_heldout": "A. Held-out (training sources)",
    "B_unseen_outlets": "B. Unseen outlets (CNN/DailyMail, XSum)",
    "C_bbc_2025_days": "C. BBC Feb-Jun 2025 (5 days)",
    "D_other_registers": "D. Tweets / Q&A titles / crypto",
}


def words(s):
    return re.findall(r"\w+", s.lower())


def token_f1(pred, ref):
    p, r = Counter(words(pred)), Counter(words(ref))
    overlap = sum((p & r).values())
    if not p or not r or not overlap:
        return 0.0
    prec, rec = overlap / sum(p.values()), overlap / sum(r.values())
    return 2 * prec * rec / (prec + rec)


def _tokens(text):
    return [t.strip(".,'\"()!?;:") for t in re.findall(r"[\w$#%][\w$#%.,'&-]*", text)]


def key_terms(text):
    """Names and numbers (heuristic): anything with a digit, $tickers/#tags, and
    capitalized non-stopword tokens. A sentence-initial capitalized word counts
    only if it looks like a name (internal capital/digit, or the next token is
    also capitalized). Returned lowercased; matching is case-insensitive."""
    terms = set()
    for sent in re.split(r"(?<=[.!?])\s+", text):
        toks = [t for t in _tokens(sent) if t]
        for k, t in enumerate(toks):
            low = t.lower()
            if len(t) < 2 or low in ENGLISH_STOP_WORDS:
                continue
            if re.search(r"\d", t) or t[0] in "$#":
                terms.add(low)
            elif t[0].isupper():
                nxt = toks[k + 1] if k + 1 < len(toks) else ""
                if k > 0 or re.search(r"[A-Z0-9]", t[1:]) or nxt[:1].isupper():
                    terms.add(low)
    return terms


def terms_kept(pred, ref):
    terms = key_terms(ref)
    if not terms:
        return None
    ptoks = {t.lower() for t in _tokens(pred)}
    return len(terms & ptoks) / len(terms)


def mean_or_none(values):
    vals = [v for v in values if v is not None]
    return float(np.mean(vals)) if vals else None


def chance_cos(pred_vecs, ref_vecs):
    """Mean cosine over mismatched (pred i, ref j != i) pairs: what an unrelated
    output scores. Anisotropic encoders have a high floor, so report lift over it."""
    sims = pred_vecs @ ref_vecs.T
    n = len(sims)
    return float((sims.sum() - np.trace(sims)) / max(1, n * (n - 1)))


def score_set(name, rows, preds, target_vecs, ref_b, calib):
    from rouge_score import rouge_scorer
    import sacrebleu

    by_id = {p["id"]: p for p in preds}
    preds = [by_id[r["id"]] for r in rows]
    hyp = [p["pred"] for p in preds]
    ref = [r["text"] for r in rows]
    pv = embed(hyp, MPNET_ID)
    cos_m = (pv * target_vecs).sum(axis=1)
    pb = embed(hyp, INDEP_ID)
    cos_b = (pb * ref_b).sum(axis=1)
    t_m = calib["encoders"]["mpnet"]["same_meaning_threshold"]["threshold"]
    t_b = calib["encoders"]["bge"]["same_meaning_threshold"]["threshold"]
    rs = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    kept = [k for k in (terms_kept(h, r) for h, r in zip(hyp, ref)) if k is not None]
    m = {
        "n": len(rows),
        "cos_mpnet": float(cos_m.mean()),
        "same_meaning_mpnet": float((cos_m >= t_m).mean()),
        "lift_mpnet": float(cos_m.mean()) - chance_cos(pv, target_vecs),
        "cos_indep": float(cos_b.mean()),
        "lift_indep": float(cos_b.mean()) - chance_cos(pb, ref_b),
        "same_meaning_indep": float((cos_b >= t_b).mean()),
        "bleu": float(sacrebleu.corpus_bleu(hyp, [ref]).score),
        "token_f1": float(np.mean([token_f1(h, r) for h, r in zip(hyp, ref)])),
        "rouge_l": float(np.mean([rs.score(r, h)["rougeL"].fmeasure for h, r in zip(hyp, ref)])),
        "exact": float(np.mean([norm_key(h) == norm_key(r) for h, r in zip(hyp, ref)])),
        "names_numbers_kept": float(np.mean(kept)) if kept else None,
        "avg_tokens": float(np.mean([gpt2_len(h) for h in hyp])),
        "ref_avg_tokens": float(np.mean([gpt2_len(r) for r in ref])),
        "stopped_rate": mean_or_none([p.get("stopped") for p in preds]),
        "distinct_candidates": mean_or_none([p.get("n_distinct") for p in preds]),
    }
    if name == "C_bbc_2025_days":
        days = defaultdict(list)
        for i, r in enumerate(rows):
            days[r["day"]].append(i)
        unit = lambda v: v / np.linalg.norm(v)
        ref_c = {d: unit(target_vecs[idx].mean(axis=0)) for d, idx in days.items()}
        pred_c = {d: unit(pv[idx].mean(axis=0)) for d, idx in days.items()}
        per_day, margins = {}, {}
        for d in sorted(days):
            own = float(pred_c[d] @ ref_c[d])
            other = float(np.mean([pred_c[d] @ ref_c[o] for o in days if o != d]))
            per_day[d], margins[d] = own, own - other
        m["day_gist_cos"] = float(np.mean(list(per_day.values())))
        # How much closer a day's reconstructions are to THAT day's news than to other days'.
        m["day_gist_margin"] = float(np.mean(list(margins.values())))
        m["day_gist_cos_per_day"] = per_day
        m["day_gist_margin_per_day"] = margins
    examples = sorted(range(len(rows)), key=lambda i: stable_int(rows[i]["id"], "ex"))[:4]
    ex = [(rows[i]["source"], ref[i], hyp[i], float(cos_m[i])) for i in examples]
    return m, ex


COLUMNS = [
    ("same_meaning_mpnet", "Same meaning (MPNet)", "{:.0%}"),
    ("same_meaning_indep", "Same meaning (indep.)", "{:.0%}"),
    ("cos_mpnet", "Cos MPNet", "{:.3f}"),
    ("lift_mpnet", "Lift MPNet", "{:+.3f}"),
    ("cos_indep", "Cos indep.", "{:.3f}"),
    ("lift_indep", "Lift indep.", "{:+.3f}"),
    ("token_f1", "Token F1", "{:.3f}"),
    ("bleu", "BLEU", "{:.1f}"),
    ("names_numbers_kept", "Names/numbers kept", "{:.0%}"),
    ("stopped_rate", "Stops by itself", "{:.0%}"),
    ("distinct_candidates", "Distinct cands", "{:.1f}"),
]


def fmt(v, f):
    return "—" if v is None else f.format(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--subset", choices=["full", "half"], default="full")
    args = ap.parse_args()

    calib = json.load(open(MANIFEST_DIR / "calibration.json"))
    results, examples = defaultdict(dict), []
    for path in sorted((DATA_DIR / "testsets").glob("*.jsonl")):
        name = path.stem
        rows = list(read_jsonl(path))
        texts = [r["text"] for r in rows]
        keep = [i for i, r in enumerate(rows) if in_subset(r["id"], args.subset)]
        target = cached(path, texts)[keep]
        target_b = cached(path, texts, INDEP_ID)[keep]
        rows = [rows[i] for i in keep]
        for system in args.systems:
            ppath = DATA_DIR / "preds" / system / f"{name}.jsonl"
            if not ppath.exists():
                continue
            m, ex = score_set(name, rows, list(read_jsonl(ppath)), target, target_b, calib)
            meta = json.load(open(ppath.with_suffix(".meta.json")))
            m["sec_per_item"] = meta.get("sec_per_item")
            results[system][name] = m
            examples += [(system, name, *e) for e in ex]

    write_json(RESULTS_DIR / f"{args.report}.json",
               {"subset": args.subset, "calibration": calib, "results": results})
    lines = [f"# Benchmark report: {args.report} ({args.subset} test sets)", ""]
    t_m = calib["encoders"]["mpnet"]["same_meaning_threshold"]["threshold"]
    t_b = calib["encoders"]["bge"]["same_meaning_threshold"]["threshold"]
    lines += [f"'Same meaning' = cosine at or above the level STS-B humans rate >= 4/5 "
              f"(MPNet {t_m:.3f}, independent {t_b:.3f}). 'Lift' = mean cosine to the right "
              f"original minus mean cosine to unrelated originals (item-specific signal).", "",
              "Read the independent-encoder columns first: v3 and nn pick their output by MPNet "
              "similarity to the input, so their MPNet columns are flattered by construction.", ""]
    for name, label in SET_LABELS.items():
        lines += [f"## {label}", "", "| System | " + " | ".join(c[1] for c in COLUMNS) + " |",
                  "|---|" + "---|" * len(COLUMNS)]
        for system in args.systems:
            m = results[system].get(name)
            if m:
                lines.append(f"| {system} | " + " | ".join(fmt(m.get(k), f) for k, _, f in COLUMNS) + " |")
        if name == "C_bbc_2025_days":
            lines.append("")
            lines.append("Day gist: cosine between a day's average reconstruction vector and that "
                         "day's average original vector; margin = how much closer it is to its own "
                         "day than to the other days. " + "; ".join(
                             f"{s} {results[s][name]['day_gist_cos']:.3f} "
                             f"(margin {results[s][name]['day_gist_margin']:+.3f})"
                             for s in args.systems if name in results[s]))
        lines.append("")
    (RESULTS_DIR / f"{args.report}.md").write_text("\n".join(lines) + "\n")
    ex_lines = [f"# Examples: {args.report}", ""]
    for system, name, source, ref, hyp, c in examples:
        ex_lines += [f"- **{system} / {name} / {source}** (cos {c:.3f})", f"  - ref:  {ref}", f"  - pred: {hyp}"]
    out = DATA_DIR / "reports" / f"{args.report}_examples.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(ex_lines) + "\n")
    print("\n".join(lines))
    print(f"examples -> {out}")


if __name__ == "__main__":
    main()
