import os
import time
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from ..data.datasets import TweetsShardDataset
from ..models.surrogate_r import SurrogateR


def collate_ids(batch, tokenizer, max_len: int):
    texts = [b["text"] for b in batch]
    # Efficiently stack numpy arrays then convert once
    e_np = np.stack([b["e"] for b in batch], axis=0).astype(np.float32, copy=False)
    e = torch.from_numpy(e_np)
    toks = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len
    )
    return {"ids": toks["input_ids"], "attn": toks.get("attention_mask", None), "e": e}


def main():
    max_len = int(os.environ.get("MAX_LEN", "64"))
    tokenizer_name = os.environ.get("TOKENIZER", "gpt2")
    train_dir = os.environ.get("TRAIN_JSONL_DIR", "./data/train")
    val_dir = os.environ.get("VAL_JSONL_DIR", "./data/val")
    log_every = int(os.environ.get("LOG_EVERY", "20"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = TweetsShardDataset(train_dir, pattern="*_emb.jsonl", max_len=max_len)
    val_ds = TweetsShardDataset(val_dir, pattern="*_emb.jsonl", max_len=max_len)
    print(
        f"[R] Train samples: {len(train_ds)} across {len(getattr(train_ds, 'files', []))} files; Val: {len(val_ds)} across {len(getattr(val_ds, 'files', []))}",
        flush=True,
    )
    workers = 0 if os.name == "nt" else 2
    train_dl = DataLoader(
        train_ds,
        batch_size=256,
        shuffle=True,
        collate_fn=lambda b: collate_ids(b, tokenizer, max_len),
        num_workers=workers,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=256,
        shuffle=False,
        collate_fn=lambda b: collate_ids(b, tokenizer, max_len),
        num_workers=workers,
    )

    scorer = SurrogateR(vocab_size=tokenizer.vocab_size).to(device)
    opt = torch.optim.AdamW(scorer.parameters(), lr=3e-4, weight_decay=0.01)
    loss_mse = nn.MSELoss()

    best = 1e9
    step = 0
    for epoch in range(1):
        scorer.train()
        for batch in train_dl:
            t0 = time.time()
            # Inputs
            ids = batch["ids"].to(device)
            attn = batch["attn"].to(device) if batch["attn"] is not None else None
            e = batch["e"].to(device)  # (B,768), already L2-normalized by prep script

            # Encode text and project embeddings to common space
            t_vec = scorer.text(ids, attn)  # (B, d)
            v_vec = scorer.emb(e)  # (B, d)
            t_vec = scorer.norm(t_vec)
            v_vec = scorer.norm(v_vec)

            # Predicted pairwise cosine matrix P (B,B)
            t_norm = F.normalize(t_vec, p=2, dim=-1)
            v_norm = F.normalize(v_vec, p=2, dim=-1)
            P = t_norm @ v_norm.T

            # Target pairwise cosine matrix T computed from in-batch e (B,B)
            e_norm = F.normalize(e, p=2, dim=-1)
            T = e_norm @ e_norm.T

            # Loss = MSE(P,T) + InfoNCE on rows (diagonal is positive)
            mse = loss_mse(P, T)
            tau = float(scorer.temp.item()) if hasattr(scorer, "temp") else 0.07
            logits = P / max(tau, 1e-6)
            targets = torch.arange(P.size(0), device=P.device)
            info_nce = F.cross_entropy(logits, targets)
            loss = mse + info_nce

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(scorer.parameters(), 1.0)
            opt.step()

            step += 1
            dt = time.time() - t0
            if step % max(1, log_every) == 0 or step == 1:
                bsz = ids.size(0)
                toks = int(ids.numel())
                print(
                    f"[R] step {step} loss {loss.item():.4f} (mse {mse.item():.4f}, nce {info_nce.item():.4f}) | {bsz/max(dt,1e-6):.1f} samples/s, {toks/max(dt,1e-6):.0f} tok/s",
                    flush=True,
                )
            if step % 1000 == 0:
                # Validation
                scorer.eval()
                with torch.no_grad():
                    vloss = 0.0
                    vcount = 0
                    for vb in val_dl:
                        ids_v = vb["ids"].to(device)
                        attn_v = (
                            vb["attn"].to(device) if vb["attn"] is not None else None
                        )
                        e_v = vb["e"].to(device)
                        t_v = scorer.text(ids_v, attn_v)
                        v_v = scorer.emb(e_v)
                        t_v = scorer.norm(t_v)
                        v_v = scorer.norm(v_v)
                        P_v = (
                            F.normalize(t_v, p=2, dim=-1)
                            @ F.normalize(v_v, p=2, dim=-1).T
                        )
                        T_v = (
                            F.normalize(e_v, p=2, dim=-1)
                            @ F.normalize(e_v, p=2, dim=-1).T
                        )
                        mse_v = loss_mse(P_v, T_v)
                        logits_v = P_v / max(tau, 1e-6)
                        targets_v = torch.arange(P_v.size(0), device=P_v.device)
                        nce_v = F.cross_entropy(logits_v, targets_v)
                        vloss += (mse_v + nce_v).item()
                        vcount += 1
                    vloss /= max(1, vcount)
                    print(f"val loss {vloss:.4f}")
                    if vloss < best:
                        best = vloss
                        os.makedirs("checkpoints", exist_ok=True)
                        torch.save(
                            {
                                "surrogate_r": scorer.state_dict(),
                                "tokenizer": tokenizer_name,
                                "config": {"max_len": max_len},
                            },
                            "checkpoints/r_best.pt",
                        )
                        print("Saved checkpoints/r_best.pt")
                scorer.train()

    # Final validation to ensure checkpoint is saved when dataset is small
    scorer.eval()
    with torch.no_grad():
        vloss = 0.0
        vcount = 0
        mse_tot = 0.0
        nce_tot = 0.0
        for vb in val_dl:
            ids_v = vb["ids"].to(device)
            attn_v = vb["attn"].to(device) if vb["attn"] is not None else None
            e_v = vb["e"].to(device)
            t_v = scorer.text(ids_v, attn_v)
            v_v = scorer.emb(e_v)
            t_v = scorer.norm(t_v)
            v_v = scorer.norm(v_v)
            P_v = F.normalize(t_v, p=2, dim=-1) @ F.normalize(v_v, p=2, dim=-1).T
            T_v = F.normalize(e_v, p=2, dim=-1) @ F.normalize(e_v, p=2, dim=-1).T
            mse_v = loss_mse(P_v, T_v)
            logits_v = P_v / max(tau, 1e-6)
            targets_v = torch.arange(P_v.size(0), device=P_v.device)
            nce_v = F.cross_entropy(logits_v, targets_v)
            vloss += (mse_v + nce_v).item()
            vcount += 1
            mse_tot += mse_v.item()
            nce_tot += nce_v.item()
        if vcount > 0:
            vloss /= vcount
            mse_avg = mse_tot / vcount
            nce_avg = nce_tot / vcount
            print(
                f"[R] val loss {vloss:.4f} (mse {mse_avg:.4f}, nce {nce_avg:.4f})",
                flush=True,
            )
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(
                {
                    "surrogate_r": scorer.state_dict(),
                    "tokenizer": tokenizer_name,
                    "config": {"max_len": max_len},
                },
                "checkpoints/r_best.pt",
            )
            print("Saved checkpoints/r_best.pt", flush=True)

    print("Done.")


if __name__ == "__main__":
    main()
