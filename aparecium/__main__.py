"""Command-line text reconstruction with Aparecium."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import sys
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Embed an English text with MPNet and reconstruct its meaning."
    )
    parser.add_argument("text", nargs="?", help="text to reconstruct (default: standard input)")
    parser.add_argument("--model-id", default="SentiChain/aparecium", help="Hugging Face model repository")
    parser.add_argument("--revision", help="model repository revision (default: main)")
    parser.add_argument("--model-path", help="local model directory")
    parser.add_argument("--device", help="cpu, cuda, cuda:<index>, or mps")
    parser.add_argument("--candidates", type=int, default=5, help="number of generated candidates, 1–32 (default: 5)")
    parser.add_argument("--max-length", type=int, help="maximum generated tokens (default: model configuration)")
    parser.add_argument("--seed", type=int, default=0, help="sampling seed (default: 0)")
    parser.add_argument("--local-files-only", action="store_true", help="use only locally cached models")
    parser.add_argument("--json", action="store_true", help="include all candidates and similarity scores as JSON")
    args = parser.parse_args(argv)
    text = args.text
    if text is None:
        if sys.stdin.isatty():
            parser.error("provide text as an argument or through standard input")
        text = sys.stdin.read().strip()
    if not text.strip():
        parser.error("text must not be empty")
    if not 1 <= args.candidates <= 32:
        parser.error("--candidates must be between 1 and 32")
    if args.max_length is not None and args.max_length < 1:
        parser.error("--max-length must be a positive integer")
    if not 0 <= args.seed < 2**63:
        parser.error("--seed must be between 0 and 2**63 - 1")
    try:
        from .api import Aparecium

        model = Aparecium(
            model_id=args.model_id,
            revision=args.revision,
            model_path=args.model_path,
            device=args.device,
            local_files_only=args.local_files_only,
        )
        result = model.invert_text(
            text, num_candidates=args.candidates, max_length=args.max_length, seed=args.seed
        )
        print(json.dumps(asdict(result), ensure_ascii=False) if args.json else result.text)
    except Exception as exc:
        print(f"aparecium: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
