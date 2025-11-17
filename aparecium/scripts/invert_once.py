"""Invert a single pooled embedding from JSON and print reconstructed text.
Input JSON should be: { "e": [float]*768 }
Usage:
    echo '{"e":[... 768 floats ...]}' | python -m aparecium_v2.scripts.invert_once --ckpt checkpoints/aparecium_v2_s1.pt
"""

import sys, json, argparse, torch
from transformers import AutoTokenizer
from ..models.emb_adapter import EmbAdapter
from ..models.decoder import RealizerDecoder
from ..models.surrogate_r import SurrogateR
from ..models.sketcher import Sketcher
from ..infer.decode import deterministic_beam_search


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to s1_best.pt or s2 checkpoint")
    ap.add_argument("--tokenizer", default="gpt2")
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--max_len", type=int, default=64)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = json.loads(sys.stdin.read())
    e = torch.tensor(data["e"], dtype=torch.float32, device=device)

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    ckpt = torch.load(args.ckpt, map_location=device)
    adapter = EmbAdapter().to(device).eval()
    decoder = RealizerDecoder(vocab_size=tok.vocab_size).to(device).eval()
    sketcher = Sketcher().to(device).eval()
    scorer = None

    adapter.load_state_dict(ckpt["adapter"])
    decoder.load_state_dict(ckpt["decoder"])
    if "sketcher" in ckpt:
        sketcher.load_state_dict(ckpt["sketcher"])
    # Optional: load trained scorer if available next to checkpoint
    r_path = "checkpoints/r_best.pt"
    try:
        rck = torch.load(r_path, map_location=device)
        key = "surrogate_r" if "surrogate_r" in rck else "scorer"
        sr = SurrogateR(vocab_size=tok.vocab_size).to(device).eval()
        sr.load_state_dict(rck[key])
        scorer = sr
    except Exception:
        scorer = None

    with torch.no_grad():
        H = adapter(e.unsqueeze(0))
    outs = deterministic_beam_search(
        decoder,
        tok,
        H,
        beam=args.beam,
        max_len=args.max_len,
        rnet=scorer,
        e=e.unsqueeze(0),
        rerank_every=8,
    )
    texts = outs["texts"][0]
    print(json.dumps({"text": texts[0], "candidates": texts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
