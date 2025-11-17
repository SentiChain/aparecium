"""Compute pooled MPNet v2 embeddings for a JSONL of tweets, with resume and checkpointing.

Input JSONL lines: { "text": str }
Output JSONL lines: { "text": str, "embedding": [float;768], "plan": {...} }

Features:
- Resume from partially written outputs ("--resume").
- Flush to disk every N successful outputs ("--checkpoint-every").
- Periodic progress logs ("--log-every").
"""

import os, json, argparse
from sentence_transformers import SentenceTransformer
import torch
from ..data.plans import extract_plan


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True, help="tweets.jsonl")
    ap.add_argument("--output", type=str, required=True, help="shard.jsonl")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument(
        "--checkpoint-every",
        type=int,
        default=100,
        dest="checkpoint_every",
        help="Flush to disk every N outputs for crash safety",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output file by skipping processed inputs",
    )
    ap.add_argument(
        "--log-every",
        type=int,
        default=100,
        dest="log_every",
        help="Print progress every N outputs",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    model = SentenceTransformer(
        "sentence-transformers/all-mpnet-base-v2",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    model.eval()
    # Count total non-empty input lines for progress display
    total_inputs = 0
    with open(args.input, "r", encoding="utf-8") as fcount:
        for line in fcount:
            if line.strip():
                total_inputs += 1

    # Determine resume position by counting existing outputs
    processed_outputs = 0
    file_mode = "w"
    if args.resume and os.path.exists(args.output):
        try:
            with open(args.output, "r", encoding="utf-8") as fexist:
                for _ in fexist:
                    processed_outputs += 1
            file_mode = "a"
        except Exception:
            # Fallback: do not resume if counting failed
            processed_outputs = 0
            file_mode = "w"

    # Open files and skip already processed inputs
    buf_text = []
    written_since_flush = 0
    printed_since_log = 0
    with (
        open(args.input, "r", encoding="utf-8") as fin,
        open(args.output, file_mode, encoding="utf-8") as fout,
    ):
        # Skip processed inputs if resuming
        to_skip = processed_outputs if file_mode == "a" else 0
        skipped = 0
        while skipped < to_skip:
            line = fin.readline()
            if not line:
                break
            if line.strip():
                skipped += 1

        processed = processed_outputs

        def flush_now():
            nonlocal written_since_flush
            try:
                fout.flush()
                os.fsync(fout.fileno())
            except Exception:
                pass
            written_since_flush = 0

        for line in fin:
            s = line.strip()
            if not s:
                continue  # skip blank lines
            try:
                rec = json.loads(s)
                text = rec.get("text", "").strip()
                if not text:
                    continue
                buf_text.append(text)
            except Exception:
                continue

            if len(buf_text) >= args.batch:
                emb = model.encode(
                    buf_text,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                for text, e in zip(buf_text, emb):
                    out = {
                        "text": text,
                        "embedding": e.tolist(),
                        "plan": extract_plan(text),
                    }
                    fout.write(json.dumps(out) + "\n")
                    processed += 1
                    written_since_flush += 1
                    printed_since_log += 1
                    if written_since_flush >= args.checkpoint_every:
                        flush_now()
                    if printed_since_log >= args.log_every:
                        pct = (processed / max(1, total_inputs)) * 100.0
                        print(
                            f"[embed] {processed}/{total_inputs} ({pct:.1f}%)",
                            flush=True,
                        )
                        printed_since_log = 0
                buf_text = []

        # Tail
        if buf_text:
            emb = model.encode(
                buf_text,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            for text, e in zip(buf_text, emb):
                out = {
                    "text": text,
                    "embedding": e.tolist(),
                    "plan": extract_plan(text),
                }
                fout.write(json.dumps(out) + "\n")
                processed += 1
                written_since_flush += 1
                printed_since_log += 1
                if written_since_flush >= args.checkpoint_every:
                    flush_now()
                if printed_since_log >= args.log_every:
                    pct = (processed / max(1, total_inputs)) * 100.0
                    print(
                        f"[embed] {processed}/{total_inputs} ({pct:.1f}%)", flush=True
                    )
                    printed_since_log = 0

        # Final flush and summary
        if written_since_flush > 0:
            flush_now()
        print(f"[embed] done {processed}/{total_inputs} (100.0%)", flush=True)


if __name__ == "__main__":
    main()
