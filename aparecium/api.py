"""Reconstruct text directly from complete MPNet token-state matrices."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from threading import RLock
from collections.abc import Sequence

import numpy as np
import torch
from huggingface_hub import snapshot_download

from .config import MAX_MATRIX_ROWS, STATE_WIDTH, TOKENIZER_FILES
from .model import MatrixDecoder, _get_device


DEFAULT_MODEL_ID = "SentiChain/aparecium-matrix"
DEFAULT_MODEL_REVISION = "188712f6bc4bda9ff257922703f29bc9b458bf70"
MAX_BATCH_SIZE = 32
MODEL_FILES = ("config.json", "model.safetensors", *TOKENIZER_FILES)


@dataclass
class InvertResult:
    """Decoded text, all interior token IDs, and visible framing-inclusive rows."""

    text: str
    token_ids: list[int]
    input_rows: int


def _bounded_integer(value: object, name: str, maximum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer between 1 and {maximum}")
    return int(value)


def _numeric_array(value: object, name: str, *, allow_bool: bool = False) -> np.ndarray:
    try:
        if isinstance(value, torch.Tensor):
            if value.layout != torch.strided or value.is_complex() or value.is_quantized or (
                value.dtype == torch.bool and not allow_bool
            ):
                raise ValueError(f"{name} must contain real numbers")
            array = value.detach().to(device="cpu", dtype=torch.float64).numpy()
        elif isinstance(value, np.ndarray) and value.dtype.kind in ("biuf" if allow_bool else "iuf"):
            with np.errstate(over="ignore", invalid="ignore"):
                array = value.astype(np.float64, copy=False)
        else:
            elements = np.asarray(value, dtype=object)
            for element in elements.flat:
                if isinstance(element, (bool, np.bool_)):
                    if allow_bool:
                        continue
                    raise ValueError(f"{name} must not contain booleans")
                if not isinstance(element, Real):
                    raise ValueError(f"{name} must contain real numbers, not strings or complex values")
            with np.errstate(over="ignore", invalid="ignore"):
                array = elements.astype(np.float64)
    except (TypeError, OverflowError, RuntimeError) as exc:
        raise ValueError(f"{name} must be a rectangular numerical array") from exc
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite numbers")
    return array


def _validate_matrix(matrix: object, attention_mask: object = None) -> tuple[torch.Tensor, int]:
    """Validate all supplied rows, then remove only explicit right padding."""
    values = _numeric_array(matrix, "matrix")
    if values.ndim != 2 or values.shape[1] != STATE_WIDTH or not 3 <= values.shape[0] <= MAX_MATRIX_ROWS:
        raise ValueError(f"matrix must have shape (3..{MAX_MATRIX_ROWS}, {STATE_WIDTH}); received {values.shape}")
    with np.errstate(over="ignore", invalid="ignore"):
        values = values.astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("matrix values must be representable as finite float32 states")
    visible = len(values)
    if attention_mask is not None:
        mask = _numeric_array(attention_mask, "attention_mask", allow_bool=True)
        if mask.shape != (len(values),) or not np.isin(mask, (0, 1)).all():
            raise ValueError("attention_mask must be a binary vector aligned with matrix rows")
        visible = int(mask.sum())
        if visible < 3 or not np.all(mask[:visible] == 1) or not np.all(mask[visible:] == 0):
            raise ValueError("attention_mask must be a right-padding prefix with at least 3 visible framing rows")
    return torch.from_numpy(values[:visible].copy()), visible


def _batch_items(value: object, name: str) -> list:
    if isinstance(value, (np.ndarray, torch.Tensor)):
        if (
            value.ndim != 3
            or value.shape[2] != STATE_WIDTH
            or not 3 <= value.shape[1] <= MAX_MATRIX_ROWS
        ):
            raise ValueError(f"{name} must be a batch of matrices")
        if len(value) > MAX_BATCH_SIZE:
            raise ValueError(f"{name} must contain at most {MAX_BATCH_SIZE} matrices")
        return list(value)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{name} must be a sequence of matrices")
    if len(value) > MAX_BATCH_SIZE:
        raise ValueError(f"{name} must contain at most {MAX_BATCH_SIZE} matrices")
    return list(value)


def _validate_matrices(matrices: object, attention_masks: object = None) -> list[tuple[torch.Tensor, int]]:
    """Validate an entire bounded batch before any decoder work."""
    items = _batch_items(matrices, "matrices")
    if attention_masks is None:
        masks = [None] * len(items)
    else:
        if isinstance(attention_masks, (np.ndarray, torch.Tensor)):
            if attention_masks.ndim != 2:
                raise ValueError("attention_masks must be a batch of binary vectors")
        elif not isinstance(attention_masks, Sequence) or isinstance(attention_masks, (str, bytes, bytearray)):
            raise ValueError("attention_masks must be a batch of binary vectors")
        if len(attention_masks) != len(items):
            raise ValueError("attention_masks must have one mask per matrix")
        masks = list(attention_masks)
    return [_validate_matrix(item, mask) for item, mask in zip(items, masks)]


class Aparecium:
    """Decode complete token-state matrices on CPU or explicit available MPS.

    Rows must be unpooled ``last_hidden_state`` output from the pinned encoder,
    including its first and last framing rows. Calls on an instance are serialized.
    Matrix decoding needs only the verified model bundle; text convenience methods
    also load the pinned encoder. Reconstruction can change factual details.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        revision: str | None = None,
        model_path: str | Path | None = None,
        device: str | None = None,
        local_files_only: bool = False,
        token_batch_size: int = 256,
    ) -> None:
        self.device = _get_device(device)
        self.token_batch_size = _bounded_integer(token_batch_size, "token_batch_size", 256)
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model_id must be a nonempty string")
        if revision is not None and (not isinstance(revision, str) or not revision.strip()):
            raise ValueError("revision must be a nonempty string or None")
        if type(local_files_only) is not bool:
            raise ValueError("local_files_only must be a boolean")
        self.model_id = model_id
        self.revision = DEFAULT_MODEL_REVISION if revision is None and model_id == DEFAULT_MODEL_ID else revision
        self.local_files_only = local_files_only
        self._lock = RLock()
        self._encoder = None
        self._encoder_tokenizer = None
        if model_path is None:
            model_path = snapshot_download(
                repo_id=model_id,
                revision=self.revision,
                allow_patterns=list(MODEL_FILES),
                local_files_only=local_files_only,
            )
        if not isinstance(model_path, (str, Path)):
            raise ValueError("model_path must be a model bundle directory")
        self.model = MatrixDecoder.load(model_path, device=self.device)

    def _ensure_encoder(self):
        if self._encoder is None:
            from transformers import AutoModel, AutoTokenizer

            config = self.model.config
            tokenizer = AutoTokenizer.from_pretrained(
                config.encoder_model_id,
                revision=config.tokenizer_revision,
                local_files_only=self.local_files_only,
                trust_remote_code=False,
                use_fast=True,
            )
            encoder = AutoModel.from_pretrained(
                config.encoder_model_id,
                revision=config.encoder_revision,
                local_files_only=self.local_files_only,
                trust_remote_code=False,
                use_safetensors=True,
                dtype=torch.float32,
            )
            encoder = encoder.to(device=self.device, dtype=torch.float32).eval().requires_grad_(False)
            self._encoder_tokenizer = tokenizer
            self._encoder = encoder
        return self._encoder_tokenizer, self._encoder

    def _invert_validated(self, matrices: list[tuple[torch.Tensor, int]]) -> list[InvertResult]:
        results = []
        for matrix, rows in matrices:
            token_ids = self.model.predict_token_ids(matrix[1:rows - 1], token_batch_size=self.token_batch_size)
            if len(token_ids) != rows - 2:
                raise RuntimeError("Decoder returned an unexpected number of token IDs")
            text = self.model.tokenizer.decode(
                token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            results.append(InvertResult(text=text, token_ids=token_ids, input_rows=rows))
        return results

    def invert_matrix(self, matrix: object, attention_mask: object = None) -> InvertResult:
        """Decode interior rows without source text, token truth, or an encoder."""
        validated = _validate_matrix(matrix, attention_mask)
        with self._lock, torch.inference_mode():
            return self._invert_validated([validated])[0]

    def invert_matrices(self, matrices: object, attention_masks: object = None) -> list[InvertResult]:
        """Decode up to 32 matrices, preserving batch and interior row order."""
        validated = _validate_matrices(matrices, attention_masks)
        with self._lock, torch.inference_mode():
            return self._invert_validated(validated)

    def invert_text(self, text: str) -> InvertResult:
        """Encode one nonempty text without truncation, then decode its token states."""
        return self.invert_texts([text])[0]

    def invert_texts(self, texts: Sequence[str], batch_size: int = 8) -> list[InvertResult]:
        """Encode and reconstruct up to 32 texts using native token-state matrices."""
        batch_size = _bounded_integer(batch_size, "batch_size", MAX_BATCH_SIZE)
        if not isinstance(texts, Sequence) or isinstance(texts, (str, bytes, bytearray)):
            raise ValueError("texts must be a sequence of nonempty strings")
        if len(texts) > MAX_BATCH_SIZE:
            raise ValueError(f"texts must contain at most {MAX_BATCH_SIZE} strings")
        items = list(texts)
        if any(not isinstance(text, str) or not text.strip() for text in items):
            raise ValueError("texts must contain only nonempty strings")
        if not items:
            return []
        with self._lock, torch.inference_mode():
            tokenizer, encoder = self._ensure_encoder()
            all_ids = tokenizer(items, add_special_tokens=True, padding=False, truncation=False)["input_ids"]
            if len(all_ids) != len(items):
                raise RuntimeError("Tokenizer returned an unexpected number of sequences")
            for ids in all_ids:
                if not 3 <= len(ids) <= MAX_MATRIX_ROWS:
                    raise ValueError(f"Each text must encode to 3..{MAX_MATRIX_ROWS} rows including framing")
                if ids[0] != self.model.config.bos_token_id or ids[-1] != self.model.config.eos_token_id:
                    raise ValueError("Encoder tokenizer did not provide the required framing tokens")
            results = []
            for start in range(0, len(items), batch_size):
                encoded = tokenizer.pad(
                    {"input_ids": all_ids[start:start + batch_size]},
                    padding=True,
                    return_attention_mask=True,
                    return_tensors="pt",
                )
                encoded = {name: value.to(self.device) for name, value in encoded.items()}
                with torch.autocast(device_type=self.device.type, enabled=False):
                    output = encoder(**encoded)
                states = output.last_hidden_state
                if not isinstance(states, torch.Tensor) or states.dtype != torch.float32:
                    raise RuntimeError("Encoder must return float32 last_hidden_state token matrices")
                if states.shape[:2] != encoded["attention_mask"].shape:
                    raise RuntimeError("Encoder token-state shape does not match its attention mask")
                validated = _validate_matrices(states, encoded["attention_mask"])
                results.extend(self._invert_validated(validated))
            return results
