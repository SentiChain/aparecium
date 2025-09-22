"""
Tweet-aware decoding constraints for Aparecium.

Provides additive logit biases for certain token continuations such as
cashtags, hashtags, handles, and URL fragments. These biases are intentionally
small and optional to avoid destabilizing the decoder.
"""

import torch  # type: ignore


def compute_logit_biases(
    prev_token_id: int,
    vocab_size: int,
    tokenizer,
) -> torch.Tensor:
    """
    Compute additive logit biases conditioned on the previously emitted token.

    Args:
        prev_token_id: the last emitted token id
        vocab_size: size of vocabulary for output logits
        tokenizer: tokenizer used to map characters to token ids

    Returns:
        torch.Tensor of shape (vocab_size,) with small biases
    """
    # Create on CPU; caller moves to logits device if needed
    bias = torch.zeros(vocab_size, dtype=torch.float32)

    # Prefer piece-level inspection to handle SentencePiece tokens
    try:
        piece = tokenizer.convert_ids_to_tokens(prev_token_id)
    except Exception:
        piece = ""

    def add_bias_for_chars(chars: str, delta: float) -> None:
        for ch in chars:
            ids = tokenizer.encode(ch, add_special_tokens=False)
            for tid in ids:
                if 0 <= tid < vocab_size:
                    bias[tid] += delta

    if piece.endswith("$") or piece == "▁$" or piece == "$":
        # Favor uppercase letters and digits; penalize whitespace/punct
        add_bias_for_chars("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", 0.25)
        add_bias_for_chars(" \t\n.,;:!?'\"", -0.5)
    elif (
        piece.endswith("#")
        or piece == "▁#"
        or piece == "#"
        or piece.endswith("@")
        or piece == "▁@"
        or piece == "@"
    ):
        add_bias_for_chars(
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_", 0.2
        )
        add_bias_for_chars(" \t\n.,;:!?'\"", -0.3)
    elif any(s in piece for s in ["http", "https", "://", ".", "h"]):
        # Very light nudges for URLs (heuristic only)
        add_bias_for_chars("htp:/.", 0.05)

    return bias
