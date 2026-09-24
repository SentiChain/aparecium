"""Run an inverter over the frozen test sets and save its reconstructions.

Systems:
  s1  - the released v2 S1 checkpoint from Hugging Face, decoded exactly as the
        public API does (deterministic beam search, beam=5, max_len=64,
        surrogate-r rerank every 8 steps, alpha=1.0), batched for speed.
  nn  - nearest-neighbour baseline: return the training text whose MPNet
        vector is closest to the input vector (--train NAME).

Usage:
    python -m bench.predict --system s1
    python -m bench.predict --system nn --train stage1
"""

from __future__ import annotations

import argparse
import os
import time

# The surrogate-r encoder uses an op missing on Apple-Silicon MPS; let PyTorch
# run just that op on CPU instead of failing.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np

from .common import DATA_DIR, device_name, in_subset, read_jsonl, write_json, write_jsonl
from .embed import cached


def test_sets(subset: str = "full"):
    for path in sorted((DATA_DIR / "testsets").glob("*.jsonl")):
        rows = list(read_jsonl(path))
        vecs = cached(path, [r["text"] for r in rows])
        keep = [i for i, r in enumerate(rows) if in_subset(r["id"], subset)]
        yield path.stem, [rows[i] for i in keep], vecs[keep]


def run_s1(args):
    import torch

    from aparecium import Aparecium
    from aparecium.infer.decode import deterministic_beam_search

    device = args.device or device_name()
    model = Aparecium(device=device)
    eos = model.tokenizer.eos_token_id
    for name, rows, vecs in test_sets(args.subset):
        out, t0 = [], time.time()
        for start in range(0, len(rows), args.batch):
            e = torch.tensor(vecs[start:start + args.batch], device=model.device)
            with torch.no_grad():
                H = model.adapter(e)
                res = deterministic_beam_search(
                    model.decoder, model.tokenizer, H, beam=args.beam, max_len=args.max_len,
                    rnet=model.rnet, e=e, rerank_every=8, constraints=None, alpha=1.0,
                )
            ids = res["ids"].cpu()
            for j, cands in enumerate(res["texts"]):
                r = rows[start + j]
                out.append({
                    "id": r["id"], "pred": cands[0].strip(), "candidates": [c.strip() for c in cands],
                    "n_distinct": len({c.strip() for c in cands}),
                    "stopped": bool((ids[j, 0, 1:] == eos).any()),
                })
            print(f"  {name}: {len(out)}/{len(rows)}", flush=True)
        secs = time.time() - t0
        save(args, name, out, secs, {"device": device, "beam": args.beam, "max_len": args.max_len,
                                     "batch": args.batch, "rerank_every": 8, "alpha": 1.0,
                                     "subset": args.subset})


def run_nn(args):
    train_rows = list(read_jsonl(DATA_DIR / "train" / f"{args.train}.jsonl"))
    train_vecs = np.load(DATA_DIR / "train" / f"{args.train}.mpnet.npy")
    for name, rows, vecs in test_sets(args.subset):
        t0 = time.time()
        sims = vecs @ train_vecs.T
        best = sims.argmax(axis=1)
        # Decoder-only health fields are undefined for retrieval.
        out = [{"id": r["id"], "pred": train_rows[b]["text"], "candidates": [train_rows[b]["text"]],
                "n_distinct": None, "stopped": None}
               for r, b in zip(rows, best)]
        save(args, name, out, time.time() - t0,
             {"train": args.train, "pool": len(train_rows), "subset": args.subset})


def save(args, name, out, secs, info):
    system = args.system if args.system != "nn" else f"nn_{args.train}"
    write_jsonl(DATA_DIR / "preds" / system / f"{name}.jsonl", out)
    write_json(DATA_DIR / "preds" / system / f"{name}.meta.json",
               {**info, "n": len(out), "seconds": secs, "sec_per_item": secs / max(1, len(out))})
    print(f"{system} {name}: {len(out)} items in {secs:.1f}s ({secs / max(1, len(out)):.2f}s/item)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=["s1", "nn"], required=True)
    ap.add_argument("--train", type=str, default="stage1")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--subset", choices=["full", "half"], default="full")
    args = ap.parse_args()
    run_s1(args) if args.system == "s1" else run_nn(args)


if __name__ == "__main__":
    main()
