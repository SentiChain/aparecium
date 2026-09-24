"""Run an inverter over the frozen test sets and save its reconstructions.

Systems:
  s1  - the released v2 S1 checkpoint from Hugging Face, decoded exactly as the
        public API does (deterministic beam search, beam=5, max_len=64,
        surrogate-r rerank every 8 steps, alpha=1.0), batched for speed.
  nn  - nearest-neighbour baseline: return the training text whose MPNet
        vector is closest to the input vector (--train NAME).
  v3  - a trained Aparecium v3 model directory (--ckpt DIR --name NAME): one
        greedy + (n_cand - 1) sampled candidates, reranked by stock MPNet.
  ckpt - a locally trained v2 checkpoint (--ckpt PATH --name NAME), decoded
        with the same beam search. No surrogate r is loaded: with the current
        beam search all beams are identical, so r reranking cannot change the
        output, and skipping it saves time.

Usage:
    python -m bench.predict --system s1
    python -m bench.predict --system nn --train stage1
    python -m bench.predict --system ckpt --ckpt data/bench/ckpt/stage1/aparecium_v2_s1.pt --name v2_stage1
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
    from aparecium import Aparecium

    device = args.device or device_name()
    m = Aparecium(device=device)
    decode_all(args, device, m.tokenizer, m.adapter, m.decoder, m.rnet)


def run_ckpt(args):
    import torch

    from aparecium.infer.service import load_models

    device = torch.device(args.device or device_name())
    # load_models also tries an r checkpoint from APARECIUM_R_CKPT / a relative path;
    # point it at a missing file so no surrogate is picked up by accident.
    os.environ["APARECIUM_R_CKPT"] = "/nonexistent/r_best.pt"
    tokenizer, adapter, _sketcher, decoder, _rnet = load_models(args.ckpt, device)
    decode_all(args, str(device), tokenizer, adapter, decoder, None)


def run_v3(args):
    from aparecium.v3.infer import V3Inverter

    device = args.device or device_name()
    inv = V3Inverter(args.ckpt, device=device)
    for name, rows, vecs in test_sets(args.subset):
        out, t0 = [], time.time()
        for start in range(0, len(rows), args.batch):
            for r, res in zip(rows[start:start + args.batch],
                              inv.invert(vecs[start:start + args.batch], n=args.n_cand)):
                out.append({"id": r["id"], "pred": res["text"], "candidates": res["candidates"],
                            "n_distinct": len(set(res["candidates"])), "stopped": res["stopped"]})
            print(f"  {name}: {len(out)}/{len(rows)}", flush=True)
        save(args, name, out, time.time() - t0, {"device": device, "n_cand": args.n_cand,
                                                   "batch": args.batch, "subset": args.subset,
                                                   "ckpt": args.ckpt})


def decode_all(args, device, tokenizer, adapter, decoder, rnet):
    import torch

    from aparecium.infer.decode import deterministic_beam_search

    eos = tokenizer.eos_token_id
    for name, rows, vecs in test_sets(args.subset):
        out, t0 = [], time.time()
        for start in range(0, len(rows), args.batch):
            e = torch.tensor(vecs[start:start + args.batch], device=torch.device(device))
            with torch.no_grad():
                H = adapter(e)
                res = deterministic_beam_search(
                    decoder, tokenizer, H, beam=args.beam, max_len=args.max_len,
                    rnet=rnet, e=e, rerank_every=8, constraints=None, alpha=1.0,
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
                                     "surrogate_r": rnet is not None, "subset": args.subset,
                                     **({"ckpt": args.ckpt} if args.ckpt else {})})


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
    system = {"nn": f"nn_{args.train}", "ckpt": args.name, "v3": args.name}.get(args.system, args.system)
    write_jsonl(DATA_DIR / "preds" / system / f"{name}.jsonl", out)
    write_json(DATA_DIR / "preds" / system / f"{name}.meta.json",
               {**info, "n": len(out), "seconds": secs, "sec_per_item": secs / max(1, len(out))})
    print(f"{system} {name}: {len(out)} items in {secs:.1f}s ({secs / max(1, len(out)):.2f}s/item)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=["s1", "nn", "ckpt", "v3"], required=True)
    ap.add_argument("--n_cand", type=int, default=5, help="v3: candidates per input")
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--name", type=str, default=None, help="system name for --system ckpt")
    ap.add_argument("--train", type=str, default="stage1")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--subset", choices=["full", "half"], default="full")
    args = ap.parse_args()
    if args.system in ("ckpt", "v3") and not (args.ckpt and args.name):
        ap.error(f"--system {args.system} needs --ckpt and --name")
    {"s1": run_s1, "nn": run_nn, "ckpt": run_ckpt, "v3": run_v3}[args.system](args)


if __name__ == "__main__":
    main()
