import os, json, argparse
import torch
from fastapi import FastAPI
import uvicorn
from ..utils.tokens import build_tokenizer
from ..data.plans import extract_plan
from ..models.emb_adapter import EmbAdapter
from ..models.sketcher import Sketcher
from ..models.decoder import RealizerDecoder
from ..models.surrogate_r import SurrogateR
from ..decoding.embedding_scorer import MPNetEmbeddingScorer as _MPNetScorer
from .decode import deterministic_beam_search


def load_models(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    tokenizer = build_tokenizer(
        ckpt.get("tokenizer", "gpt2"), ckpt["config"].get("max_len", 64)
    )
    vocab = len(tokenizer)
    adapter = EmbAdapter().to(device).eval()
    sketcher = Sketcher().to(device).eval()
    decoder = RealizerDecoder(vocab_size=vocab).to(device).eval()
    adapter.load_state_dict(ckpt["adapter"])
    sketcher.load_state_dict(ckpt["sketcher"])
    decoder.load_state_dict(ckpt["decoder"])
    rnet = None
    # Prefer separate r_best checkpoint if available
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
        # fallback: load from combined ckpt if present
        if "surrogate_r" in ckpt:
            rnet = SurrogateR(vocab_size=vocab).to(device).eval()
            rnet.load_state_dict(ckpt["surrogate_r"])
    return tokenizer, adapter, sketcher, decoder, rnet


app = FastAPI()
MODELS = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MPNET_FINAL = None  # lazy-loaded for optional final rerank


@app.on_event("startup")
def _load():
    global MODELS
    ckpt = os.environ.get("APARECIUM_CKPT", "aparecium_v2_s2.pt")
    MODELS = load_models(ckpt, DEVICE)


@app.post("/invert")
def invert(payload: dict):
    tokenizer, adapter, sketcher, decoder, rnet = MODELS
    deterministic = bool(payload.get("deterministic", True))
    beam = int(payload.get("beam", 5))
    max_len = int(payload.get("max_len", 64))
    use_constraints = bool(payload.get("constraints", True))
    final_mpnet = bool(payload.get("final_mpnet", False))

    e = torch.tensor(
        payload["embedding"], dtype=torch.float32, device=DEVICE
    ).unsqueeze(0)
    with torch.no_grad():
        H = adapter(e)
    # Build simple constraint from sketcher prediction (placeholder): require tickers/hashtags if any
    constraints = None
    plan = None
    try:
        # If caller provided source text, extract plan for echo/reference; else run minimal sketch
        src_text = payload.get("source_text")
        if src_text:
            plan = extract_plan(src_text)
        else:
            # minimal plan from sketcher boolean head
            out = sketcher(e)
            plan = {
                "has_url": bool(
                    (out["has_url"] > 0.5).item()
                    if out["has_url"].ndim == 2
                    else out["has_url"] > 0.5
                )
            }
        if use_constraints and plan:
            from ..models.constraints import MustIncludeStrings

            req = []
            for k in ("tickers", "hashtags", "addresses", "amounts"):
                for v in plan.get(k, []) if isinstance(plan.get(k, []), list) else []:
                    if k == "tickers":
                        req.append(f"${v}")
                    else:
                        req.append(str(v))
            constraints = [MustIncludeStrings(req)]
    except Exception:
        pass

    outs = deterministic_beam_search(
        decoder,
        tokenizer,
        H,
        beam=beam,
        max_len=max_len,
        rnet=rnet,
        e=e,
        rerank_every=payload.get("rerank_every", 8),
        constraints=constraints,
        alpha=payload.get("alpha", 1.0),
    )
    texts = outs["texts"][0]
    scores = {"lm_logp": outs["logp"][0].tolist()}

    # Optional final MPNet rerank among top-K
    cos_mpnet = None
    if final_mpnet:
        global MPNET_FINAL
        if MPNET_FINAL is None:
            MPNET_FINAL = _MPNetScorer()
        if MPNET_FINAL is not None:
            with torch.no_grad():
                emb = MPNET_FINAL.encode_and_pool(texts)  # (K,768)
                e_norm = torch.nn.functional.normalize(e, p=2, dim=-1).to(emb.device)
                cos = torch.nn.functional.cosine_similarity(
                    torch.nn.functional.normalize(emb, p=2, dim=-1),
                    e_norm.expand_as(emb),
                    dim=-1,
                )
                # Rerank texts by cos desc
                order = torch.argsort(cos, descending=True)
                texts = [texts[i] for i in order.tolist()]
                cos_mpnet = float(cos[order[0]].item())
                scores["cos_mpnet"] = [float(x) for x in cos.tolist()]
    response = {
        "text": texts[0],
        "candidates": texts,
        "scores": scores,
        "plan": plan,
        "version": "2.0.0",
    }
    return response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    os.environ["APARECIUM_CKPT"] = args.ckpt
    uvicorn.run(
        "aparecium_v2.infer.service:app",
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
