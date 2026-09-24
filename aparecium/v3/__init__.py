"""Aparecium v3 (in development): pooled MPNet vector -> soft prefix -> small pretrained LM.

See ``model.py`` for the architecture, ``train.py`` for training and
``infer.py`` for candidate generation and MPNet reranking.
"""

from .model import V3Config, V3Model

__all__ = ["V3Config", "V3Model"]
