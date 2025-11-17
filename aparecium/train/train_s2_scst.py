import os, argparse, time
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from ..utils.common import set_determinism, AverageMeter, to_device
from ..utils.tokens import build_tokenizer, encode_batch, decode_tokens
from ..data.datasets import TweetsShardDataset, collate
from ..models.emb_adapter import EmbAdapter
from ..models.sketcher import Sketcher
from ..models.decoder import RealizerDecoder
from ..models.surrogate_r import SurrogateR
from ..models.constraints import MustIncludeStrings, apply_constraints_penalty
from ..infer.decode import deterministic_beam_search


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=str, required=True)
    ap.add_argument("--ckpt_s1", type=str, required=True)
    ap.add_argument("--save_dir", type=str, required=True)
    ap.add_argument("--pattern", type=str, default="*_emb.jsonl")
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--gamma", type=float, default=0.2)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--log_every", type=int, default=20)
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    set_determinism(args.seed)
    tokenizer = build_tokenizer("gpt2", max_len=args.max_len)
    vocab = len(tokenizer)

    # Load S1
    ckpt = torch.load(args.ckpt_s1, map_location="cpu")
    adapter = EmbAdapter()
    sketcher = Sketcher()
    decoder = RealizerDecoder(vocab_size=vocab)
    adapter.load_state_dict(ckpt["adapter"])
    sketcher.load_state_dict(ckpt["sketcher"])
    decoder.load_state_dict(ckpt["decoder"])

    # Surrogate r
    rnet = SurrogateR(vocab_size=vocab)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    adapter.to(device)
    sketcher.to(device)
    decoder.to(device)
    rnet.to(device)

    train_ds = TweetsShardDataset(
        args.shards, pattern=args.pattern, max_len=args.max_len
    )
    print(
        f"[S2] Train samples: {len(train_ds)} across {len(getattr(train_ds, 'files', []))} files (pattern={args.pattern})",
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

    # Optimizers: fine‑tune decoder + r; optionally adapter/sketcher small LR
    opt = torch.optim.AdamW(
        list(decoder.parameters()) + list(rnet.parameters()), lr=args.lr
    )

    step = 0
    loss_meter = AverageMeter()

    while step < args.steps:
        for batch in train_dl:
            t0 = time.time()
            batch = to_device(batch, device)
            e = batch["e"]
            # Build constraints from plans (tickers, hashtags, addresses, amounts strings)
            req_strings = []
            for pl in batch["plans"]:
                s = []
                for t in pl.get("tickers", []):
                    s.append(f"${t}")
                s += pl.get("hashtags", [])
                s += pl.get("addresses", [])
                s += [a for a in pl.get("amounts", []) if isinstance(a, str)]
                req_strings.append(s)

            with torch.no_grad():
                H = adapter(e)
            # Decode beams (deterministic) and also get logprobs for RL
            outs = deterministic_beam_search(
                decoder, tokenizer, H, beam=args.beam, max_len=args.max_len
            )

            texts = outs["texts"]  # list of list: B x beam
            ids_list = outs["ids"]  # B x beam x T
            logp_list = outs["logp"]  # B x beam

            # Surrogate r scores per candidate
            B = len(texts)
            r_scores = torch.zeros(B, args.beam, device=device)
            for b in range(B):
                toks = tokenizer(
                    texts[b],
                    padding=True,
                    truncation=True,
                    max_length=args.max_len,
                    return_tensors="pt",
                ).to(device)
                ee = e[b].unsqueeze(0).repeat(args.beam, 1)
                r_scores[b] = rnet(
                    toks["input_ids"], toks.get("attention_mask", None), ee
                )

            # Domain penalties for missing required strings
            penalties = []
            for b in range(B):
                penalties.append(
                    apply_constraints_penalty(
                        texts[b], [MustIncludeStrings(req_strings[b])]
                    ).to(device)
                )
            penalties = torch.stack(penalties, dim=0)  # (B,beam)

            # LM prior: average token logprob per sequence (already in logp_list)
            lm_prior = logp_list / (outs["lengths"] + 1e-6)

            # Reward
            reward = (
                args.alpha * r_scores + args.beta * lm_prior + args.gamma * penalties
            )

            # Self‑critical: baseline = beam[0] (greedy path in our deterministic decode)
            baseline = reward[:, 0].unsqueeze(1).detach()
            advantage = reward - baseline

            # Policy gradient loss: negative advantage * logp
            loss = -(advantage * logp_list).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(decoder.parameters()) + list(rnet.parameters()), 1.0
            )
            opt.step()

            loss_meter.update(loss.item(), n=B)
            step += 1
            dt = time.time() - t0
            if step % max(1, args.log_every) == 0 or step == 1:
                bsz = B
                print(
                    f"[S2] step {step}/{args.steps} rl_loss {loss_meter.avg:.4f} | {bsz/max(dt,1e-6):.1f} samples/s, beam {args.beam}, max_len {args.max_len}",
                    flush=True,
                )
                loss_meter.reset()
            if step >= args.steps:
                break

    torch.save(
        {
            "adapter": adapter.state_dict(),
            "sketcher": sketcher.state_dict(),
            "decoder": decoder.state_dict(),
            "surrogate_r": rnet.state_dict(),
            "tokenizer": tokenizer.name_or_path,
            "config": {"max_len": args.max_len},
        },
        os.path.join(args.save_dir, "aparecium_v2_s2.pt"),
    )
    print("Saved", os.path.join(args.save_dir, "aparecium_v2_s2.pt"))


if __name__ == "__main__":
    main()
