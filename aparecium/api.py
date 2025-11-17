"""
High–level Aparecium API for easy use from PyPI.

This module provides a small wrapper class that:
- Downloads the S1 checkpoint from Hugging Face by default.
- Instantiates the EmbAdapter, Sketcher, Decoder, and optional surrogate r.
- Exposes simple methods to invert a pooled MPNet embedding or a raw text
  (by first embedding it with sentence-transformers).

Intended usage:

    from aparecium import Aparecium

    model = Aparecium()  # loads S1 baseline from HF by default
    text = "Bitcoin ETF inflows hit a new weekly high as markets turn risk-on."
    out = model.invert_text(text)
    print(out)

The heavy-weight training scripts and FastAPI service remain in the
aparecium.aparecium.train / infer modules; this file just provides a
convenient entrypoint for downstream users.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer

from .infer.service import load_models
from .infer.decode import deterministic_beam_search


HF_DEFAULT_REPO_ID = "SentiChain/aparecium-v2-pooled-reverser"
HF_DEFAULT_CKPT = "aparecium_v2_s1.pt"
HF_DEFAULT_R_CKPT = "r_best.pt"
DEFAULT_MPNET_MODEL = "sentence-transformers/all-mpnet-base-v2"


def _get_device(explicit: Optional[str] = None) -> torch.device:
    if explicit is not None:
        return torch.device(explicit)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class InvertResult:
    """Structured result for a single inversion call."""

    text: str
    candidates: List[str]


class Aparecium:
    """
    High–level wrapper for the Aparecium v2 (pooled‑only) S1 baseline.

    By default this class downloads the S1 checkpoint from Hugging Face:
    - repo_id: SentiChain/aparecium-v2-pooled-reverser
    - filename: aparecium_v2_s1.pt

    and, if available, the surrogate r checkpoint:
    - filename: r_best.pt

    It exposes:
    - invert_embedding(e): pooled MPNet vector -> reconstructed text
    - invert_text(text): raw text -> pooled MPNet -> reconstructed text
    """

    def __init__(
        self,
        repo_id: str = HF_DEFAULT_REPO_ID,
        ckpt_filename: str = HF_DEFAULT_CKPT,
        r_ckpt_filename: str = HF_DEFAULT_R_CKPT,
        device: Optional[str] = None,
        mpnet_model: str = DEFAULT_MPNET_MODEL,
    ) -> None:
        self.device: torch.device = _get_device(device)
        self.repo_id = repo_id
        self.ckpt_filename = ckpt_filename
        self.r_ckpt_filename = r_ckpt_filename
        self.mpnet_model_name = mpnet_model

        # Download main checkpoint from HF.
        ckpt_path = hf_hub_download(repo_id, ckpt_filename)

        # Download optional surrogate r checkpoint (if present).
        r_ckpt_path: Optional[str]
        try:
            r_ckpt_path = hf_hub_download(repo_id, r_ckpt_filename)
        except Exception:
            r_ckpt_path = None

        # load_models will:
        # - build tokenizer
        # - create EmbAdapter, Sketcher, Decoder
        # - try to load surrogate_r from APARECIUM_R_CKPT or ckpt
        if r_ckpt_path is not None:
            # The service helper prefers APARECIUM_R_CKPT if set.
            import os

            os.environ["APARECIUM_R_CKPT"] = r_ckpt_path

        (
            self.tokenizer,
            self.adapter,
            self.sketcher,
            self.decoder,
            self.rnet,
        ) = load_models(ckpt_path, self.device)

        self._mpnet: Optional[SentenceTransformer] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_mpnet(self) -> SentenceTransformer:
        if self._mpnet is None:
            self._mpnet = SentenceTransformer(
                self.mpnet_model_name,
                device=str(self.device),
            )
        return self._mpnet

    def _invert_single_embedding(
        self,
        e: Sequence[float],
        beam: int = 5,
        max_len: int = 64,
        rerank_every: int = 8,
        alpha: float = 1.0,
    ) -> InvertResult:
        """Core inversion from a single pooled embedding."""
        # (1) move embedding to device
        e_tensor = torch.tensor(e, dtype=torch.float32, device=self.device).unsqueeze(0)

        # (2) adapter -> memory H
        with torch.no_grad():
            H = self.adapter(e_tensor)

        # (3) run deterministic beam search with optional surrogate r
        outs = deterministic_beam_search(
            self.decoder,
            self.tokenizer,
            H,
            beam=beam,
            max_len=max_len,
            rnet=self.rnet,
            e=e_tensor,
            rerank_every=rerank_every,
            constraints=None,  # simple API: no hard constraints by default
            alpha=alpha,
        )

        texts: List[str] = outs["texts"][0]
        return InvertResult(text=texts[0], candidates=texts)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def invert_embedding(
        self,
        embedding: Sequence[float],
        beam: int = 5,
        max_len: int = 64,
        rerank_every: int = 8,
        alpha: float = 1.0,
    ) -> InvertResult:
        """
        Invert a single pooled MPNet embedding into text.

        Args:
            embedding: iterable of 768 floats (L2-normalized is recommended).
            beam: beam size for deterministic beam search.
            max_len: maximum decoded length in tokens.
            rerank_every: how often to rerank beams with surrogate r.
            alpha: weight of surrogate r in the fused score.

        Returns:
            InvertResult with `.text` (top1) and `.candidates` (beam list).
        """
        return self._invert_single_embedding(
            embedding,
            beam=beam,
            max_len=max_len,
            rerank_every=rerank_every,
            alpha=alpha,
        )

    def invert_text(
        self,
        text: str,
        beam: int = 5,
        max_len: int = 64,
        rerank_every: int = 8,
        alpha: float = 1.0,
    ) -> InvertResult:
        """
        Convenience wrapper: embed a raw text with MPNet, then invert the pooled vector.

        This is mostly for quick experimentation. For production usage you
        usually embed many posts upfront and cache the 768-D pooled vectors.
        """
        mpnet = self._ensure_mpnet()
        e = mpnet.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        return self.invert_embedding(
            e,
            beam=beam,
            max_len=max_len,
            rerank_every=rerank_every,
            alpha=alpha,
        )
