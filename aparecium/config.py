"""The validated, portable token-state decoder configuration."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, fields
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


STATE_WIDTH = 768
VOCABULARY_SIZE = 30527
MAX_MATRIX_ROWS = 256
ENCODER_MODEL_ID = "sentence-transformers/all-mpnet-base-v2"
ENCODER_REVISION = "e8c3b32edf5434bc2275fc9bab85f82640a19130"
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt")
TENSOR_SHAPES = {
    "vocabulary": (VOCABULARY_SIZE, STATE_WIDTH),
    "dense.weight": (STATE_WIDTH, STATE_WIDTH),
    "dense.bias": (STATE_WIDTH,),
    "layer_norm.weight": (STATE_WIDTH,),
    "layer_norm.bias": (STATE_WIDTH,),
    "vocabulary_bias": (VOCABULARY_SIZE,),
}


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def _sha256(value: object, name: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class ModelConfig:
    """Exact architecture, encoder identity, and bundle integrity metadata."""

    weights_sha256: str
    source_checkpoint_sha256: str
    vocabulary_array_sha256: str
    tensor_sha256: Mapping[str, str]
    tokenizer_files_sha256: Mapping[str, str]
    schema_version: int = 1
    model_type: str = "aparecium_matrix"
    architecture: str = "dense_gelu_layernorm_frozen_vocabulary"
    state_width: int = STATE_WIDTH
    vocabulary_size: int = VOCABULARY_SIZE
    max_sequence_length: int = MAX_MATRIX_ROWS
    encoder_model_id: str = ENCODER_MODEL_ID
    encoder_revision: str = ENCODER_REVISION
    tokenizer_revision: str = ENCODER_REVISION
    pad_token_id: int = 1
    bos_token_id: int = 0
    eos_token_id: int = 2
    unk_token_id: int = 104
    layer_norm_eps: float = 1e-5
    weights_file: str = "model.safetensors"

    def __post_init__(self) -> None:
        expected = {
            "schema_version": 1,
            "model_type": "aparecium_matrix",
            "architecture": "dense_gelu_layernorm_frozen_vocabulary",
            "state_width": STATE_WIDTH,
            "vocabulary_size": VOCABULARY_SIZE,
            "max_sequence_length": MAX_MATRIX_ROWS,
            "encoder_model_id": ENCODER_MODEL_ID,
            "encoder_revision": ENCODER_REVISION,
            "tokenizer_revision": ENCODER_REVISION,
            "pad_token_id": 1,
            "bos_token_id": 0,
            "eos_token_id": 2,
            "unk_token_id": 104,
            "layer_norm_eps": 1e-5,
            "weights_file": "model.safetensors",
        }
        for name, wanted in expected.items():
            value = getattr(self, name)
            if type(value) is not type(wanted) or value != wanted:
                raise ValueError(f"{name} must be {wanted!r}")
        for name in ("weights_sha256", "source_checkpoint_sha256", "vocabulary_array_sha256"):
            _sha256(getattr(self, name), name)
        for name, keys in (("tensor_sha256", TENSOR_SHAPES), ("tokenizer_files_sha256", TOKENIZER_FILES)):
            value = getattr(self, name)
            if not isinstance(value, Mapping) or set(value) != set(keys):
                raise ValueError(f"{name} must contain exactly the required filenames or tensor names")
            for key, digest in value.items():
                _sha256(digest, f"{name}.{key}")
            object.__setattr__(self, name, MappingProxyType(dict(value)))
        if self.vocabulary_array_sha256 != self.tensor_sha256["vocabulary"]:
            raise ValueError("vocabulary_array_sha256 must match the vocabulary tensor digest")

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "ModelConfig":
        """Require every supported field, with no ignored or inferred settings."""
        if not isinstance(values, Mapping):
            raise ValueError("Model configuration must be a JSON object")
        expected = {field.name for field in fields(cls)}
        missing, unknown = expected - set(values), set(values) - expected
        if missing or unknown:
            details = []
            if missing:
                details.append("missing fields: " + ", ".join(sorted(missing)))
            if unknown:
                details.append("unknown fields: " + ", ".join(sorted(map(str, unknown))))
            raise ValueError("Invalid model configuration (" + "; ".join(details) + ")")
        return cls(**dict(values))

    @classmethod
    def load(cls, path: str | Path) -> "ModelConfig":
        """Read a bounded JSON configuration and reject duplicate fields."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Required model file is missing: {path.name}")
        if not 1 <= path.stat().st_size <= 65536:
            raise ValueError("Model configuration file size is invalid")
        try:
            value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_json_object)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("Model configuration must be valid UTF-8 JSON") from exc
        return cls.from_dict(value)

    def to_dict(self) -> dict:
        """Return independent JSON-compatible configuration values."""
        result = {field.name: getattr(self, field.name) for field in fields(self)}
        for name in ("tensor_sha256", "tokenizer_files_sha256"):
            result[name] = dict(result[name])
        return result
