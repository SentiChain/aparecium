"""
Simple CLI entrypoint for Aparecium v2.

Example:
    echo "Bitcoin ETF inflows hit a new weekly high as markets turn risk-on." ^| ^
      python -m aparecium "SentiChain/aparecium-v2-pooled-reverser"
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

from .api import Aparecium


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(
        description="Aparecium v2 (pooled‑only) — invert a pooled MPNet embedding or raw text."
    )
    ap.add_argument(
        "text",
        type=str,
        nargs="?",
        help="Raw text to embed with MPNet and invert (if omitted, read from stdin).",
    )
    ap.add_argument(
        "--repo-id",
        type=str,
        default="SentiChain/aparecium-v2-pooled-reverser",
        help="Hugging Face repo id to load the checkpoint from.",
    )
    ap.add_argument(
        "--ckpt",
        type=str,
        default="aparecium_v2_s1.pt",
        help="Checkpoint filename within the HF repo.",
    )
    ap.add_argument(
        "--device",
        type=str,
        default=None,
        help="torch device (default: cuda if available, else cpu).",
    )
    args = ap.parse_args(argv)

    # Read text from stdin if not provided as an argument.
    text = args.text
    if not text:
        text = sys.stdin.read().strip()
    if not text:
        print("No input text provided.", file=sys.stderr)
        raise SystemExit(1)

    model = Aparecium(
        repo_id=args.repo_id,
        ckpt_filename=args.ckpt,
        device=args.device,
    )
    out = model.invert_text(text)
    print(out.text)


if __name__ == "__main__":  # pragma: no cover - CLI only
    main()
