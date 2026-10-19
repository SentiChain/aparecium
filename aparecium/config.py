"""Validated configuration for the embedding reconstruction model."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Union


@dataclass(frozen=True)
class ModelConfig:
    """Architecture and pinned pretrained model dependencies."""

    architecture: str = "prefix-decoder"
    base_model: str = "openai-community/gpt2"
    base_revision: str = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
    encoder_model: str = "sentence-transformers/all-mpnet-base-v2"
    encoder_revision: str = "e8c3b32edf5434bc2275fc9bab85f82640a19130"
    embedding_dim: int = 768
    prefix_length: int = 10
    mapper_hidden: int = 1024
    lora_rank: int = 16
    max_length: int = 64

    def __post_init__(self) -> None:
        if self.architecture != "prefix-decoder":
            raise ValueError("Unsupported model architecture")
        if self.base_model != "openai-community/gpt2":
            raise ValueError("base_model must be openai-community/gpt2")
        if self.encoder_model != "sentence-transformers/all-mpnet-base-v2":
            raise ValueError("encoder_model must be sentence-transformers/all-mpnet-base-v2")
        for name in ("base_revision", "encoder_revision"):
            value = getattr(self, name)
            if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{40}", value) is None:
                raise ValueError(f"{name} must be a full lowercase Git commit hash")
        for name in ("embedding_dim", "prefix_length", "mapper_hidden", "lora_rank", "max_length"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.embedding_dim != 768:
            raise ValueError("embedding_dim must be 768 for the supported encoder")
        if self.prefix_length + self.max_length > 1024:
            raise ValueError("prefix_length + max_length exceeds the language model context window")
        if self.lora_rank > 768:
            raise ValueError("lora_rank must not exceed the language model width")

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "ModelConfig":
        """Read a complete configuration without silently ignoring fields."""
        if not isinstance(values, Mapping):
            raise ValueError("Model configuration must be a JSON object")
        expected = {field.name for field in fields(cls)}
        missing = expected - set(values)
        unknown = set(values) - expected
        if missing or unknown:
            details = []
            if missing:
                details.append("missing fields: " + ", ".join(sorted(missing)))
            if unknown:
                details.append("unknown fields: " + ", ".join(sorted(map(str, unknown))))
            raise ValueError("Invalid model configuration (" + "; ".join(details) + ")")
        return cls(**dict(values))

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ModelConfig":
        """Load a configuration JSON file."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict:
        """Return the serializable configuration."""
        return asdict(self)
