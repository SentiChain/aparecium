import argparse
import json
import os
import random
import re
from typing import List, Tuple

# Ensure local repo is imported over any site-packages
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aparecium.reverser import Seq2SeqReverser  # type: ignore
from aparecium.db_utils import ApareciumDB  # type: ignore
from aparecium.logger import logger  # type: ignore


def list_chunk_ids(db: ApareciumDB) -> List[int]:
    chunk_ids: List[int] = []
    probe = 0
    while True:
        try:
            exist = db.check_batch_exists(probe, probe)
        except Exception:
            break
        if not exist:
            break
        chunk_ids.append(probe)
        probe += 1
    return chunk_ids


def fetch_chunk(
    db: ApareciumDB, chunk_id: int
) -> Tuple[List[str], List[List[List[float]]]]:
    sentences, matrices = db.retrieve_batch(block_start=chunk_id, block_end=chunk_id)
    # Convert any numpy/torch matrices into nested lists of floats
    src_mats: List[List[List[float]]] = []
    for m in matrices:
        if m is None:
            continue
        try:
            # numpy
            src_mats.append(m.astype("float32").tolist())
        except Exception:
            try:
                # torch
                src_mats.append(m.detach().cpu().float().tolist())
            except Exception:
                # already a list
                src_mats.append(m)  # type: ignore
    return sentences, src_mats


def sanitize_text(text: str) -> str:
    # Drop tokens like [unused96]
    text = re.sub(r"\[unused\d+\]\s*", "", text)
    # Remove spaces before punctuation, keep punctuation
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    # Join hashtags/cashtags/mentions
    text = re.sub(r"([#@$])\s+([A-Za-z0-9_]+)", r"\1\2", text)
    # Join hyphenated words (e.g., layer - 2 -> layer-2)
    text = re.sub(r"(?<=\w)\s*-\s*(?=\w)", "-", text)
    # Collapse extra spaces
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate trained model on random DB samples"
    )
    parser.add_argument("--config", type=str, default="pipeline/config.json")
    parser.add_argument("--db", type=str, default=None)
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--num-examples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default=None, help="cpu|cuda|auto")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    db_path = args.db or cfg.get("db")
    model_dir = args.model_dir or cfg.get("train", {}).get(
        "model_dir", "models/pipeline_reverser"
    )

    if not db_path or not os.path.exists(db_path):
        raise SystemExit(f"DB not found: {db_path}")
    if not os.path.exists(os.path.join(model_dir, "reverser_seq2seq_state.pt")):
        raise SystemExit(f"Model checkpoint not found in {model_dir}")

    random.seed(args.seed)

    db = ApareciumDB(db_path)

    reverser = Seq2SeqReverser()
    reverser.load_model(model_dir)

    # Decoding defaults per README
    decode_kwargs = dict(
        max_length=128,
        num_beams=5,
        deterministic=True,
        length_penalty_alpha=0.6,
        lambda_sim=0.3,
        rescore_every_k=4,
        rescore_top_m=8,
        beta=10.0,
        enable_constraints=True,
        return_confidence=True,
    )

    chunks = list_chunk_ids(db)
    if not chunks:
        raise SystemExit("No chunks found in DB.")

    # Sample up to num-examples entries across chunks
    results: List[Tuple[str, str, dict]] = []
    while len(results) < args.num_examples and chunks:
        cid = random.choice(chunks)
        sentences, src_mats = fetch_chunk(db, cid)
        if not sentences:
            chunks.remove(cid)
            continue
        # Pair sentences with matrices
        pairs = list(zip(sentences, src_mats))
        random.shuffle(pairs)
        for tgt, src in pairs:
            try:
                text, info = reverser.generate_text(src, **decode_kwargs)
                text = sanitize_text(text)
            except Exception as e:
                logger.warning(f"Decoding error on chunk {cid}: {e}")
                continue
            results.append((tgt, text, info))
            if len(results) >= args.num_examples:
                break

    # Print
    for i, (tgt, gen, info) in enumerate(results, start=1):
        print("== EXAMPLE", i)
        print("TARGET:", tgt)
        print("GENERATED:", gen)
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
