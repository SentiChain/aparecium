"""Aparecium‑V2 (pooled‑only) — embedding inversion from MPNet pooled vectors.
This package provides:
- EmbAdapter (multi‑channel) to convert a 768‑D pooled vector into a pseudo‑sequence.
- Sketcher (plan head) to predict crypto‑domain constraints.
- Decoder with constrained beam search and surrogate similarity reranking.
- Surrogate scorer r(x,e) approximating cos(MPNet(x), e).
- Supervised and SCST training loops.
"""

__version__ = "2.0.0"
