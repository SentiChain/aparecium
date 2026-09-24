"""Train an Aparecium v3 model on (text, pooled MPNet vector) pairs.

Usage (trial run):
    python -m aparecium.v3.train --base openai-community/gpt2 --mode finetune \\
        --train_jsonl data/bench/train/stage1.jsonl --train_vecs data/bench/train/stage1.mpnet.npy \\
        --save_dir data/bench/ckpt/v3_trial_gpt2 --max_steps 150

A seeded slice of the training set (--dev_size) is held out for the dev loss;
the best checkpoint by dev loss is saved. Logs throughput and peak memory.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import resource
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch

from .model import V3Config, V3Model


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="openai-community/gpt2")
    ap.add_argument("--mode", choices=["finetune", "lora", "frozen"], default="finetune")
    ap.add_argument("--prefix_len", type=int, default=10)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--train_vecs", required=True)
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--dev_size", type=int, default=300)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--max_steps", type=int, default=0, help="stop after this many steps (0 = all epochs)")
    ap.add_argument("--lr_mapper", type=float, default=1e-3)
    ap.add_argument("--lr_lm", type=float, default=None, help="default: 5e-5 finetune, 5e-4 lora")
    ap.add_argument("--warmup_frac", type=float, default=0.05)
    ap.add_argument("--eval_every", type=int, default=0, help="steps between dev evals (0 = once per epoch)")
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--device", default=None)
    ap.add_argument("--grad_ckpt", action="store_true",
                    help="recompute layer activations in the backward pass (less memory, ~30%% slower)")
    ap.add_argument("--mps_mem_cap_gb", type=float, default=0,
                    help="hard cap on Apple-GPU memory; training fails instead of exceeding it (0 = no cap)")
    ap.add_argument("--seed", type=int, default=1234)
    return ap.parse_args()


_MPS_PEAK = [0]


def peak_memory_gb(device: torch.device) -> dict:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_gb = rss / 1e9 if sys.platform == "darwin" else rss / 1e6  # bytes on macOS, KB on Linux
    out = {"process_peak_rss_gb": round(rss_gb, 2)}
    if device.type == "mps":
        # Live tensors (what the model needs) vs. the driver's total, which includes cached blocks.
        _MPS_PEAK[0] = max(_MPS_PEAK[0], torch.mps.current_allocated_memory())
        out["mps_tensors_peak_gb"] = round(_MPS_PEAK[0] / 1e9, 2)
        out["mps_driver_gb"] = round(torch.mps.driver_allocated_memory() / 1e9, 2)
    elif device.type == "cuda":
        out["cuda_peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    return out


def length_batches(lengths: torch.Tensor, batch_size: int, gen: torch.Generator):
    """Shuffle, then sort by length inside windows of 50 batches, so each batch
    holds texts of similar length (little padding); batch order is shuffled."""
    perm = torch.randperm(len(lengths), generator=gen)
    window = batch_size * 50
    batches = []
    for w in range(0, len(perm), window):
        chunk = perm[w:w + window]
        chunk = chunk[torch.argsort(lengths[chunk], stable=True)]
        batches += [chunk[b:b + batch_size] for b in range(0, len(chunk), batch_size)]
    order = torch.randperm(len(batches), generator=gen)
    return [batches[i] for i in order]


def take(data, idx, device):
    """Select rows and trim padding to the longest text in this batch."""
    mask = data["attention_mask"][idx]
    width = int(mask.sum(dim=1).max())
    return (data["e"][idx].to(device), data["input_ids"][idx][:, :width].to(device), mask[:, :width].to(device))


@torch.no_grad()
def dev_loss(model, dev, device, batch_size):
    model.eval()
    total, count = 0.0, 0
    for s in range(0, len(dev["e"]), batch_size):
        e, ids, mask = take(dev, torch.arange(s, min(s + batch_size, len(dev["e"]))), device)
        n_tok = int(mask.sum())
        total += float(model(e, ids, mask).loss) * n_tok
        count += n_tok
    model.train()
    return total / max(1, count)


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available()
                              else "mps" if torch.backends.mps.is_available() else "cpu")

    if device.type == "mps" and args.mps_mem_cap_gb:
        torch.mps.set_per_process_memory_fraction(args.mps_mem_cap_gb * 1e9 / torch.mps.recommended_max_memory())

    rows = [json.loads(line) for line in open(args.train_jsonl, encoding="utf-8") if line.strip()]
    vecs = np.load(args.train_vecs).astype(np.float32)
    assert len(rows) == len(vecs), "texts and vectors are misaligned"
    order = list(range(len(rows)))
    random.Random(args.seed).shuffle(order)
    dev_idx, train_idx = order[: args.dev_size], order[args.dev_size:]

    cfg = V3Config(base=args.base, mode=args.mode, prefix_len=args.prefix_len, lora_r=args.lora_r)
    model = V3Model(cfg).to(device)
    if args.grad_ckpt:
        model.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    lr_lm = args.lr_lm or (5e-5 if args.mode == "finetune" else 5e-4)

    def tensors(idx):
        enc = model.encode_texts([rows[i]["text"] for i in idx])
        return {"e": torch.from_numpy(vecs[idx]), **enc}

    train, dev = tensors(train_idx), tensors(dev_idx)
    n_train = len(train_idx)
    steps_per_epoch = math.ceil(n_train / args.batch_size)
    total_steps = args.max_steps or steps_per_epoch * args.epochs
    warmup = max(1, int(total_steps * args.warmup_frac))

    lm_params = [p for p in model.lm.parameters() if p.requires_grad]
    groups = [{"params": list(model.mapper.parameters()), "lr": args.lr_mapper}]
    if lm_params:
        groups.append({"params": lm_params, "lr": lr_lm})
    opt = torch.optim.AdamW(groups, weight_decay=0.01)

    def lr_scale(s):  # linear warm-up, then linear decay to zero
        if s < warmup:
            return (s + 1) / warmup
        return max(0.0, (total_steps - s) / max(1, total_steps - warmup))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_scale)
    trainable = sum(p.numel() for g in groups for p in g["params"])
    print(f"[v3] base={args.base} mode={args.mode} device={device} train={n_train} dev={len(dev_idx)} "
          f"trainable={trainable / 1e6:.1f}M steps={total_steps}", flush=True)

    log = {"args": vars(args), "trainable_params": trainable, "evals": [], "steps": []}
    best, step, t_start = float("inf"), 0, time.time()
    eval_every = args.eval_every or steps_per_epoch
    lengths = train["attention_mask"].sum(dim=1)
    gen = torch.Generator().manual_seed(args.seed)
    model.train()
    while step < total_steps:
        for idx in length_batches(lengths, args.batch_size, gen):
            if step >= total_steps:
                break
            t0 = time.time()
            loss = model(*take(train, idx, device)).loss
            if not torch.isfinite(loss):
                sys.exit(f"[v3] training diverged: non-finite loss at step {step + 1}; no checkpoint written")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
            opt.step()
            sched.step()
            if device.type == "mps":
                torch.mps.synchronize()
                peak_memory_gb(device)  # sample live-tensor memory every step
                if args.mps_mem_cap_gb:
                    torch.mps.empty_cache()  # return cached blocks so the cap measures real need
            step += 1
            dt = time.time() - t0
            if step % args.log_every == 0 or step == 1:
                rec = {"step": step, "loss": round(float(loss), 4), "samples_per_s": round(len(idx) / dt, 1),
                       **peak_memory_gb(device)}
                log["steps"].append(rec)
                print(f"[v3] {rec}", flush=True)
            if step % eval_every == 0 or step == total_steps:
                dl = dev_loss(model, dev, device, args.batch_size)
                if not math.isfinite(dl):
                    sys.exit(f"[v3] training diverged: non-finite dev loss at step {step}")
                log["evals"].append({"step": step, "dev_loss": round(dl, 4)})
                print(f"[v3] step {step} dev_loss {dl:.4f}", flush=True)
                if dl < best:
                    best = dl
                    model.save(args.save_dir)

    log.update(best_dev_loss=best, minutes=round((time.time() - t_start) / 60, 2), **peak_memory_gb(device))
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.save_dir) / "train_log.json").write_text(json.dumps(log, indent=1))
    print(f"[v3] done: best dev loss {best:.4f} in {log['minutes']} min | {peak_memory_gb(device)}", flush=True)


if __name__ == "__main__":
    main()
