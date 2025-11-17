import os, argparse
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from ..utils.common import set_determinism, AverageMeter, to_device
from ..utils.tokens import build_tokenizer, encode_batch
from ..data.datasets import TweetsShardDataset, collate
from ..models.emb_adapter import EmbAdapter
from ..models.sketcher import Sketcher
from ..models.decoder import RealizerDecoder


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--shards",
        type=str,
        required=True,
        help="Directory with JSONL shards (train split)",
    )
    ap.add_argument(
        "--val_shards",
        type=str,
        required=False,
        help="Directory with JSONL shards (val split)",
    )
    ap.add_argument("--save_dir", type=str, required=True)
    ap.add_argument(
        "--pattern",
        type=str,
        default="*_emb.jsonl",
        help="Glob pattern for shards (use *_emb.jsonl)",
    )
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup_steps", type=int, default=10000)
    ap.add_argument("--steps", type=int, default=300000)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--log_every", type=int, default=20)
    return ap.parse_args()


def ce_loss(logits, target_ids, pad_id):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = target_ids[:, 1:].contiguous()
    loss = nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=pad_id,
    )
    return loss


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    set_determinism(args.seed)
    tokenizer = build_tokenizer("gpt2", max_len=args.max_len)

    train_ds = TweetsShardDataset(
        args.shards, pattern=args.pattern, max_len=args.max_len
    )
    print(
        f"[S1] Found {len(train_ds)} samples across {len(train_ds.files)} files (pattern={args.pattern})",
        flush=True,
    )
    workers = 0 if os.name == "nt" else 2
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate(tokenizer, b, args.max_len),
        num_workers=workers,
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    vocab = len(tokenizer)

    adapter = EmbAdapter(d_in=768, D=384, S_list=(8, 16), dropout=0.1)
    sketcher = Sketcher(d_in=768, hidden=768, dropout=0.1)
    decoder = RealizerDecoder(
        vocab_size=vocab, d_model=768, n_head=8, n_layer=12, d_ff=3072, dropout=0.1
    )

    adapter.to(device)
    sketcher.to(device)
    decoder.to(device)
    opt = torch.optim.AdamW(
        list(adapter.parameters())
        + list(sketcher.parameters())
        + list(decoder.parameters()),
        lr=args.lr,
        weight_decay=0.01,
    )
    total_steps = min(args.steps, args.epochs * len(train_dl))
    sched = get_cosine_schedule_with_warmup(
        opt, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps
    )

    pad_id = tokenizer.pad_token_id
    loss_meter = AverageMeter()

    import time

    step = 0
    decoder.train()
    adapter.train()
    sketcher.train()
    for epoch in range(args.epochs):
        for batch in train_dl:
            t0 = time.time()
            batch = to_device(batch, device)
            H = adapter(batch["e"])  # (B,S,D)
            # Optional: use sketcher outputs later; here not used in forward
            logits = decoder(batch["input_ids"], memory=H)
            loss = ce_loss(logits, batch["input_ids"], pad_id)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(adapter.parameters())
                + list(sketcher.parameters())
                + list(decoder.parameters()),
                1.0,
            )
            opt.step()
            sched.step()

            loss_meter.update(loss.item(), n=batch["input_ids"].size(0))
            step += 1
            dt = time.time() - t0
            if step % max(1, args.log_every) == 0 or step == 1:
                bsz = batch["input_ids"].size(0)
                toks = int(batch["input_ids"].numel())
                speed = bsz / max(dt, 1e-6)
                tps = toks / max(dt, 1e-6)
                print(
                    f"[S1] step {step}/{total_steps} loss {loss_meter.avg:.4f} | {speed:.1f} samples/s, {tps:.0f} tok/s",
                    flush=True,
                )
                loss_meter.reset()
            if step >= total_steps:
                break
        if step >= total_steps:
            break

    torch.save(
        {
            "adapter": adapter.state_dict(),
            "sketcher": sketcher.state_dict(),
            "decoder": decoder.state_dict(),
            "tokenizer": tokenizer.name_or_path,
            "config": {"max_len": args.max_len},
        },
        os.path.join(args.save_dir, "aparecium_v2_s1.pt"),
    )
    print("Saved", os.path.join(args.save_dir, "aparecium_v2_s1.pt"))


if __name__ == "__main__":
    main()
