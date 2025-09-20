"""
Decoding utility functions for Aparecium.

Includes helpers for building key padding masks, GNMT-style length penalty,
mask-aware mean pooling, and cosine similarity used during decoding.
"""

from typing import List, Optional
import torch  # type: ignore


def build_memory_key_padding_mask(lengths: List[int], max_src_len: int) -> torch.Tensor:
    """
    Build a BoolTensor mask of shape (B, max_src_len) with True indicating padding.

    Args:
            lengths: list of true sequence lengths before padding
            max_src_len: the maximum source length after padding

    Returns:
            BoolTensor of shape (B, max_src_len)
    """
    batch_size = len(lengths)
    mem_kpm = torch.ones(batch_size, max_src_len, dtype=torch.bool)
    for b, L in enumerate(lengths):
        L_safe = max(0, min(int(L), max_src_len))
        mem_kpm[b, :L_safe] = False
    return mem_kpm


def apply_length_penalty(logprob: float, length: int, alpha: float) -> float:
    """GNMT-style length penalty normalization.

    lp = logprob / (((5.0 + length) / 6.0) ** max(alpha, 0.0))
    """
    den = ((5.0 + max(length, 1)) / 6.0) ** max(alpha, 0.0)
    return float(logprob) / float(den)


def mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Mask-aware mean pool over sequence.

    Args:
            hidden: (B, T, D) or (T, B, D)
            mask: (B, T) with 1 for valid, 0 for pad (same as HF attention_mask)
    """
    if hidden.dim() == 3 and hidden.shape[0] != mask.shape[0]:
        # assume (T, B, D) -> (B, T, D)
        hidden = hidden.transpose(0, 1)
    mask = mask.to(dtype=hidden.dtype)
    sum_vec = (hidden * mask.unsqueeze(-1)).sum(dim=1)
    den = mask.sum(dim=1).clamp_min(1.0).unsqueeze(-1)
    return sum_vec / den


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Cosine similarity row-wise between a and b.

    Args:
            a: (N, D)
            b: (M, D) or (1, D)
    Returns:
            (N, M) cosine similarity
    """
    a_norm = torch.nn.functional.normalize(a, p=2, dim=-1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=-1)
    return a_norm @ b_norm.transpose(0, 1)


def pool_source_memory(
    memory: torch.Tensor, mem_kpm: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute a robust mean-pool of source memory for target embedding.

    Args:
            memory: (src_len, d) or (src_len, 1, d)
            mem_kpm: optional (B, src_len) or (src_len,) where True indicates padding

    Returns:
            (1, d) pooled vector
    """
    if memory.dim() == 3:
        # (T, B, D) → assume B=1 and squeeze
        memory_ = memory[:, 0, :]
    else:
        memory_ = memory

    if mem_kpm is None:
        # Try trimming trailing near-zero rows
        norms = memory_.norm(dim=1)
        valid_idx = (norms > 1e-8).nonzero(as_tuple=False).squeeze(-1)
        if valid_idx.numel() > 0:
            memory_ = memory_[: valid_idx[-1] + 1]
        pooled = memory_.mean(dim=0, keepdim=True)
        return pooled

    # Build keep mask
    if mem_kpm.dim() == 2:
        keep = ~mem_kpm[0]
    else:
        keep = ~mem_kpm
    if keep.any():
        pooled = memory_[keep].mean(dim=0, keepdim=True)
    else:
        pooled = memory_.mean(dim=0, keepdim=True)
    return pooled
