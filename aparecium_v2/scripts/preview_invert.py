"""Preview inversion quality on a small JSONL of embeddings.

Usage:
  python -m aparecium_v2.scripts.preview_invert \
    --ckpt checkpoints/aparecium_v2_s2.pt \
    --emb_jsonl aparecium_v2/data/preview_10_emb.jsonl \
    --beam 5 --max_len 64 --limit 10
"""

import os, json, argparse
import torch
from ..utils.tokens import build_tokenizer
from ..models.emb_adapter import EmbAdapter
from ..models.decoder import RealizerDecoder
from ..models.surrogate_r import SurrogateR
from ..infer.decode import deterministic_beam_search
from ..data.plans import extract_plan
from ..models.constraints import MustIncludeStrings


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--emb_jsonl", type=str, required=True)
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--device", type=str, default="cuda")
    return ap.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location="cpu")
    tokenizer = build_tokenizer(ckpt.get("tokenizer", "gpt2"), max_len=args.max_len)
    vocab = len(tokenizer)

    adapter = EmbAdapter().to(device).eval()
    decoder = RealizerDecoder(vocab_size=vocab).to(device).eval()
    adapter.load_state_dict(ckpt["adapter"])  # type: ignore
    decoder.load_state_dict(ckpt["decoder"])  # type: ignore

    # Optional surrogate reranker
    rnet = None
    r_path = os.environ.get("APARECIUM_R_CKPT", "checkpoints/r_best.pt")
    try:
        rck = torch.load(r_path, map_location="cpu")
        rnet = SurrogateR(vocab_size=vocab).to(device).eval()
        key = (
            "surrogate_r"
            if "surrogate_r" in rck
            else ("scorer" if "scorer" in rck else None)
        )
        if key is not None:
            rnet.load_state_dict(rck[key])
    except Exception:
        rnet = None

    shown = 0
    with open(args.emb_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if shown >= args.limit:
                break
            rec = json.loads(line)
            text = rec.get("text", "")
            emb = rec["embedding"]
            e = torch.tensor(emb, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                H = adapter(e)
            # Build light constraints from GT plan to bias entities/numbers
            plan = extract_plan(text)
            req = []
            for t in plan.get("tickers", []):
                req.append(f"${t}")
            req += plan.get("hashtags", [])
            req += plan.get("addresses", [])
            for a in plan.get("amounts", []):
                if isinstance(a, str):
                    req.append(a)
            constraints = [MustIncludeStrings(req)] if req else None
            outs = deterministic_beam_search(
                decoder,
                tokenizer,
                H,
                beam=args.beam,
                max_len=args.max_len,
                rnet=rnet,
                e=e,
                rerank_every=8,
                constraints=constraints,
            )
            pred = outs["texts"][0][0]
            print(f"{shown+1}. GT:   {text}")
            print(f"   -> Pred: {pred}")
            shown += 1


if __name__ == "__main__":
    main()
