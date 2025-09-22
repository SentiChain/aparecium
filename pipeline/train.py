import argparse
import json
import math
import os
from typing import List, Tuple

import torch  # type: ignore

# Ensure local repo is imported over any site-packages
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aparecium.reverser import Seq2SeqReverser  # type: ignore
from aparecium.db_utils import ApareciumDB  # type: ignore
from aparecium.logger import logger  # type: ignore


def fetch_chunk(db: ApareciumDB, chunk_id: int) -> Tuple[List[str], List[torch.Tensor]]:
    sentences, matrices = db.retrieve_batch(block_start=chunk_id, block_end=chunk_id)
    tensors: List[torch.Tensor] = []
    for m in matrices:
        if isinstance(m, torch.Tensor):
            t = m
        else:
            # numpy array -> torch
            t = torch.tensor(m, dtype=torch.float32)
        tensors.append(t.tolist())  # Seq2SeqReverser expects List[List[float]]
    return sentences, tensors


def iter_minibatches(
    src_batch: List[List[List[float]]],
    tgt_batch: List[str],
    batch_size: int,
):
    total = len(tgt_batch)
    for i in range(0, total, batch_size):
        yield src_batch[i : i + batch_size], tgt_batch[i : i + batch_size]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train or resume Seq2SeqReverser from cached DB"
    )
    parser.add_argument("--config", type=str, default="pipeline/config.json")
    parser.add_argument("--db", type=str, default=None)
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default=None, help="cpu|cuda|auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-source-length", type=int, default=None)
    parser.add_argument("--max-target-length", type=int, default=None)
    parser.add_argument(
        "--save-every", type=int, default=None, help="Save every N steps; 0 to disable"
    )
    args = parser.parse_args()

    # Load config defaults
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    db_path = args.db or cfg.get("db")
    model_dir = args.model_dir or cfg.get("train", {}).get(
        "model_dir", "models/pipeline_reverser"
    )
    epochs = args.epochs or int(cfg.get("train", {}).get("epochs", 3))
    batch_size = args.batch_size or int(cfg.get("train", {}).get("batch_size", 16))
    lr = args.lr or float(cfg.get("train", {}).get("lr", 1e-4))
    device_cfg = args.device or cfg.get("device", "auto")
    save_every = args.save_every
    if save_every is None:
        save_every = int(cfg.get("train", {}).get("save_every", 0))
    max_src_len = args.max_source_length or int(cfg.get("max_source_length", 384))
    max_tgt_len = args.max_target_length or int(cfg.get("max_target_length", 128))

    if device_cfg == "auto":
        device = None
    else:
        device = device_cfg

    if not db_path or not os.path.exists(os.path.dirname(db_path)):
        raise SystemExit(f"Database path invalid or directory missing: {db_path}")

    os.makedirs(model_dir, exist_ok=True)

    # Model
    reverser = Seq2SeqReverser(lr=lr, device=device)

    # Resume if requested
    if args.resume and os.path.exists(
        os.path.join(model_dir, "reverser_seq2seq_state.pt")
    ):
        logger.info(f"Resuming from checkpoint in {model_dir}")
        reverser.load_model(model_dir)
    else:
        logger.info("Starting fresh training run")

    # Training data iterator over chunks
    db = ApareciumDB(db_path)

    # Figure out how many chunks exist by probing consecutive ids
    # We will assume contiguous chunk ids starting at 0 until a miss
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

    if not chunk_ids:
        raise SystemExit("No chunks found in DB. Run pipeline/prepare_db.py first.")

    global_step = 0
    for epoch in range(1, epochs + 1):
        logger.info(f"Epoch {epoch}/{epochs} starting with {len(chunk_ids)} chunks")
        running_loss = 0.0
        num_steps = 0

        for cid in chunk_ids:
            sentences, src_mats = fetch_chunk(db, cid)
            # Use the original text as target for reconstruction
            targets = sentences

            # Train in mini-batches
            for src_mb, tgt_mb in iter_minibatches(src_mats, targets, batch_size):
                loss = reverser.train_step_batch(
                    source_rep_batch=src_mb,
                    target_text_batch=tgt_mb,
                    max_source_length=max_src_len,
                    max_target_length=max_tgt_len,
                )
                running_loss += float(loss)
                num_steps += 1
                global_step += 1

                if save_every and save_every > 0 and (global_step % save_every == 0):
                    logger.info(
                        f"Global step {global_step}: saving checkpoint to {model_dir}"
                    )
                    reverser.save_model(model_dir)

        epoch_loss = running_loss / max(1, num_steps)
        logger.info(f"Epoch {epoch} complete: avg loss={epoch_loss:.4f}")
        reverser.save_model(model_dir)

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
