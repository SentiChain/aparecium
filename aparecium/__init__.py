"""Aparecium‑V2 (pooled‑only) — embedding inversion from MPNet pooled vectors.

This package provides:

- Low‑level components (EmbAdapter, Sketcher, Decoder, surrogate r).
- Training scripts for S1 supervised and optional SCST (S2) fine‑tuning.
- A high‑level :class:`Aparecium` wrapper for easy use from PyPI, which can
  automatically download the S1 checkpoint from Hugging Face and expose simple
  `invert_embedding` / `invert_text` methods.

The class name :class:`Aparecium` is the main user‑facing API going forward.
"""

from .api import Aparecium  # primary user-facing name

__all__ = ["Aparecium"]
__version__ = "2.0.0"
