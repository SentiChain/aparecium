import argparse
import json
import os
import re
from typing import List

# Ensure local repo is imported over any site-packages
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aparecium.vectorizer import Vectorizer  # type: ignore
from aparecium.reverser import Seq2SeqReverser  # type: ignore


DEFAULT_PROMPTS = [
    "BTC is hovering near resistance; a breakout could trigger momentum.",
    "ETH gas fees dropped after the latest upgrade; users are cheering.",
    "Solana NFTs are back in vogue; volumes spiked overnight.",
    "Macro FOMC minutes today; traders expect increased volatility.",
    "Whale moved 5,000 BTC to an exchange; fear of sell-off.",
    "Layer2 adoption continues; more dApps migrating for lower fees.",
    "Regulatory clarity in the EU boosts institutional interest.",
    "Altcoin season rumors again; be careful with leverage.",
    "On-chain metrics show healthy accumulation across addresses.",
    "Funding rates turned positive; sentiment leans bullish.",
]


def sanitize_text(text: str) -> str:
    text = re.sub(r"\[unused\d+\]\s*", "", text)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    text = re.sub(r"([#@$])\s+([A-Za-z0-9_]+)", r"\1\2", text)
    text = re.sub(r"(?<=\w)\s*-\s*(?=\w)", "-", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate model on custom prompts by re-embedding then decoding"
    )
    parser.add_argument("--config", type=str, default="pipeline/config.json")
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-source-length", type=int, default=None)
    parser.add_argument("--num-beams", type=int, default=5)
    parser.add_argument("--lambda-sim", type=float, default=0.3)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    model_dir = args.model_dir or cfg.get("train", {}).get(
        "model_dir", "models/pipeline_reverser"
    )
    model_name = args.model_name or cfg.get(
        "model_name", "sentence-transformers/all-mpnet-base-v2"
    )
    device = args.device or cfg.get("device", "auto")
    max_src_len = args.max_source_length or int(cfg.get("max_source_length", 384))

    reverser = Seq2SeqReverser()
    reverser.load_model(model_dir)

    vectorizer = Vectorizer(
        model_name=model_name, device=None if device == "auto" else device
    )

    for i, prompt in enumerate(DEFAULT_PROMPTS, start=1):
        mat = vectorizer.encode(prompt, max_length=max_src_len)
        text, info = reverser.generate_text(
            mat,
            max_length=128,
            num_beams=args.num_beams,
            deterministic=True,
            length_penalty_alpha=0.6,
            lambda_sim=args.lambda_sim,
            rescore_every_k=4,
            rescore_top_m=8,
            beta=10.0,
            enable_constraints=True,
            return_confidence=True,
        )
        text = sanitize_text(text)
        print("== PROMPT", i)
        print("INPUT:", prompt)
        print("RECONSTRUCTED:", text)
        print(
            "CONF:",
            {
                "cosine": round(float(info.get("cosine", 0.0)), 4),
                "score_norm": round(float(info.get("score_norm", 0.0)), 4),
                "fused_score": round(float(info.get("fused_score", 0.0)), 4),
            },
        )
        print()


if __name__ == "__main__":
    main()
