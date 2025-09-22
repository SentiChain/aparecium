import argparse
import json
import os
import random
from pathlib import Path
from typing import Iterable, List

import numpy as np  # type: ignore

# Ensure local repo is imported over any site-packages
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aparecium.vectorizer import Vectorizer  # type: ignore
from aparecium.db_utils import ApareciumDB  # type: ignore
from aparecium.logger import logger  # type: ignore


def read_lines(path: str) -> Iterable[str]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line


def choose_samples(
    lines: Iterable[str], k: int, random_sample: bool, seed: int
) -> List[str]:
    if not random_sample:
        out: List[str] = []
        for i, line in enumerate(lines):
            if i >= k:
                break
            out.append(line)
        return out
    # Reservoir sample deterministically
    random.seed(seed)
    reservoir: List[str] = []
    n = 0
    for line in lines:
        n += 1
        if len(reservoir) < k:
            reservoir.append(line)
        else:
            j = random.randint(1, n)
            if j <= k:
                reservoir[j - 1] = line
    return reservoir


def ensure_parent(path: str) -> None:
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare SQLite DB with cached post embeddings"
    )
    parser.add_argument("--config", type=str, default="pipeline/config.json")
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--db", type=str, default=None)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--random-sample", action="store_true", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--device", type=str, default=None, help="cpu|cuda|auto")
    parser.add_argument("--max-source-length", type=int, default=None)
    args = parser.parse_args()

    # Load config defaults
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    input_path = args.input or cfg.get("input")
    db_path = args.db or cfg.get("db")
    samples = args.samples or int(cfg.get("samples", 10000))
    random_sample = (
        cfg.get("random_sample", True)
        if args.random_sample is None
        else bool(args.random_sample)
    )
    seed = args.seed or int(cfg.get("seed", 12345))
    chunk_size = args.chunk_size or int(cfg.get("chunk_size", 1000))
    model_name = args.model_name or cfg.get(
        "model_name", "sentence-transformers/all-mpnet-base-v2"
    )
    device_cfg = args.device or cfg.get("device", "auto")
    max_src_len = args.max_source_length or int(cfg.get("max_source_length", 384))

    if device_cfg == "auto":
        device = None
    else:
        device = device_cfg

    if not input_path:
        raise SystemExit("--input not provided and missing in config")
    if not os.path.exists(input_path):
        raise SystemExit(f"Input file not found: {input_path}")

    ensure_parent(db_path)

    logger.info(
        f"Preparing DB: input={input_path}, db={db_path}, samples={samples}, chunk_size={chunk_size}, model={model_name}"
    )

    # Read and sample posts
    all_lines_iter = read_lines(input_path)
    selected = choose_samples(
        all_lines_iter, samples, random_sample=random_sample, seed=seed
    )
    logger.info(f"Selected {len(selected)} lines for embedding")

    # Initialize DB and vectorizer
    db = ApareciumDB(db_path)
    vectorizer = Vectorizer(model_name=model_name, device=device)

    # Process in chunks and store
    total = len(selected)
    num_chunks = (total + chunk_size - 1) // chunk_size
    for chunk_id in range(num_chunks):
        start = chunk_id * chunk_size
        end = min(start + chunk_size, total)
        batch_lines = selected[start:end]
        logger.info(
            f"Processing chunk {chunk_id+1}/{num_chunks} with {len(batch_lines)} posts"
        )

        # Embed each line individually to keep sequence matrices separate
        matrices: List[np.ndarray] = []
        for text in batch_lines:
            mat = vectorizer.encode(text, max_length=max_src_len)  # List[List[float]]
            matrices.append(np.array(mat, dtype=np.float32))

        # Store into DB with block metadata (chunk_id)
        db.store_batch(
            block_start=chunk_id,
            block_end=chunk_id,
            sentences=batch_lines,
            matrices=matrices,
        )

    logger.info("Embedding cache complete.")


if __name__ == "__main__":
    main()
