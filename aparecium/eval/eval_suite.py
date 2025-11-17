import json, os, argparse
from tqdm import tqdm
from ..utils.tokens import build_tokenizer
from ..models.emb_adapter import EmbAdapter
from ..models.decoder import RealizerDecoder
from ..models.surrogate_r import SurrogateR
from ..infer.decode import deterministic_beam_search
import torch
from .metrics import exact_match, number_accuracy


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--test_jsonl", type=str, required=True)
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--beam", type=int, default=5)
    return ap.parse_args()


def main():
    args = parse_args()
    ckpt = torch.load(args.ckpt, map_location="cpu")
    tokenizer = build_tokenizer(ckpt.get("tokenizer", "gpt2"), max_len=args.max_len)
    vocab = len(tokenizer)
    adapter = EmbAdapter()
    decoder = RealizerDecoder(vocab_size=vocab)
    adapter.load_state_dict(ckpt["adapter"])
    decoder.load_state_dict(ckpt["decoder"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adapter.to(device).eval()
    decoder.to(device).eval()

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

    em = 0
    num_acc = 0
    n = 0
    with open(args.test_jsonl, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            rec = json.loads(line)
            e = torch.tensor(
                rec["embedding"], dtype=torch.float32, device=device
            ).unsqueeze(0)
            H = adapter(e)
            outs = deterministic_beam_search(
                decoder,
                tokenizer,
                H,
                beam=args.beam,
                max_len=args.max_len,
                rnet=rnet,
                e=e,
                rerank_every=8,
            )
            pred = outs["texts"][0][0]
            gold = rec["text"]
            em += exact_match(pred, gold)
            num_acc += number_accuracy(pred, gold)
            n += 1
    print({"exact_match": em / max(1, n), "number_acc": num_acc / max(1, n), "n": n})


if __name__ == "__main__":
    main()
