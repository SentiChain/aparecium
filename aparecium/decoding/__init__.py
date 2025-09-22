"""
Decoding helpers for Aparecium.

This subpackage contains utilities used during generation/decoding, including:
- MPNet-based embedding scorer for outer-loop rescoring
- Length penalty and pooling utilities
- Optional tweet-aware constraints (logit biases)
"""

from .utils import (
    build_memory_key_padding_mask,
    apply_length_penalty,
    mean_pool,
    cosine_sim,
    pool_source_memory,
)
from .embedding_scorer import MPNetEmbeddingScorer
from .constraints import compute_logit_biases

__all__ = [
    "build_memory_key_padding_mask",
    "apply_length_penalty",
    "mean_pool",
    "cosine_sim",
    "pool_source_memory",
    "MPNetEmbeddingScorer",
    "compute_logit_biases",
]
