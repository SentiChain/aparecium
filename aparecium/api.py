"""Reconstruct English text from stock MPNet sentence embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from threading import RLock
from typing import Sequence

import numpy as np
import torch
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer

from .model import PrefixDecoder, is_fluent


DEFAULT_MODEL_ID = "SentiChain/aparecium"
MODEL_FILES = (
    "config.json",
    "mapper.safetensors",
    "adapter/adapter_config.json",
    "adapter/adapter_model.safetensors",
)


@dataclass
class InvertResult:
    """A reconstruction and its candidates in generation order.

    ``scores`` contains MPNet cosine similarities, not confidence estimates.
    ``selected_index`` identifies the chosen candidate in both lists.
    ``stopped`` reports whether that candidate emitted its end-of-text token.
    """

    text: str
    candidates: list[str]
    scores: list[float]
    selected_index: int
    stopped: bool


def _get_device(explicit: str | None) -> torch.device:
    if explicit is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        device = torch.device(explicit)
    except (TypeError, RuntimeError) as exc:
        raise ValueError("device must be cpu, cuda, cuda:<index>, or mps") from exc
    if device.type not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be cpu, cuda, cuda:<index>, or mps")
    if device.type in {"cpu", "mps"} and device.index is not None:
        raise ValueError(f"{device.type} does not accept a device index")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available on this system")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device index {device.index} is not available")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is not available on this system")
    return device


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _generation_options(num_candidates: int, max_length: int | None, seed: int, maximum_length: int) -> None:
    _positive_integer(num_candidates, "num_candidates")
    if num_candidates > 32:
        raise ValueError("num_candidates must be between 1 and 32")
    if max_length is not None:
        _positive_integer(max_length, "max_length")
        if max_length > maximum_length:
            raise ValueError(f"max_length must be between 1 and {maximum_length}")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer between 0 and 2**63 - 1")


def _validate_vectors(value: object, dimension: int, *, single: bool = False) -> np.ndarray:
    """Validate shape and finite, nonzero coordinates without changing direction."""
    try:
        if isinstance(value, torch.Tensor):
            if value.dtype == torch.bool or value.is_complex():
                raise ValueError("embeddings must contain real numbers, not booleans")
            array = value.detach().to(device="cpu", dtype=torch.float64).numpy()
        elif isinstance(value, np.ndarray) and value.dtype.kind in "iuf":
            array = value.astype(np.float64, copy=False)
        else:
            elements = np.asarray(value, dtype=object)
            if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, Real) for x in elements.flat):
                raise ValueError("embeddings must contain real numbers, not booleans or strings")
            array = elements.astype(np.float64)
    except (TypeError, OverflowError, RuntimeError) as exc:
        raise ValueError("embeddings must be a rectangular array of real numbers") from exc

    if not single and array.shape == (0,):
        return np.empty((0, dimension), dtype=np.float32)
    expected_ndim = 1 if single else 2
    if array.ndim != expected_ndim or array.shape[-1] != dimension:
        shape = f"({dimension},)" if single else f"(batch, {dimension})"
        raise ValueError(f"embeddings must have shape {shape}; received {array.shape}")
    if single:
        array = array[np.newaxis, :]
    if not np.isfinite(array).all():
        raise ValueError("embeddings must contain only finite numbers")
    if np.any(np.max(np.abs(array), axis=1) == 0):
        raise ValueError("embeddings must not contain a zero vector")
    return array


def _normalised_tensor(vectors: np.ndarray, device: torch.device) -> torch.Tensor:
    """Normalize in model precision, rescaling only numerically extreme inputs."""
    with np.errstate(over="ignore", under="ignore"):
        tensor = torch.tensor(vectors.astype(np.float32), device=device)
    norms = tensor.norm(p=2, dim=-1)
    unsafe = ~torch.isfinite(norms) | (norms < 1e-12)
    if unsafe.any().item():
        rows = vectors[unsafe.cpu().numpy()]
        scaled = rows / np.max(np.abs(rows), axis=1, keepdims=True)
        tensor[unsafe] = torch.tensor(scaled.astype(np.float32), device=device)
    return torch.nn.functional.normalize(tensor, dim=-1)


class Aparecium:
    """Load a prefix decoder and reconstruct texts from 768-dimensional vectors.

    Inputs must come from the stock encoder recorded in the model configuration.
    The default device is CUDA when available, otherwise CPU. Set ``device="mps"``
    explicitly to use Apple Silicon acceleration. ``local_files_only=True``
    requires the decoder, base language model, and encoder to be available locally.

    Calls on the same instance are serialized. A seed makes the same call
    repeatable on the same device and software versions; changing batch size can
    change sampled candidates. Reconstructions may omit or invent factual details.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        revision: str | None = None,
        model_path: str | Path | None = None,
        device: str | None = None,
        local_files_only: bool = False,
    ) -> None:
        self.device = _get_device(device)
        self.model_id = model_id
        self.revision = revision
        self.local_files_only = local_files_only
        self._lock = RLock()
        self._encoder: SentenceTransformer | None = None
        path = model_path
        if path is None:
            path = snapshot_download(
                repo_id=model_id,
                revision=revision,
                allow_patterns=list(MODEL_FILES),
                local_files_only=local_files_only,
            )
        self.model = PrefixDecoder.load(
            Path(path), device=str(self.device), local_files_only=local_files_only
        )

    def _ensure_encoder(self) -> SentenceTransformer:
        if self._encoder is None:
            encoder = SentenceTransformer(
                self.model.config.encoder_model,
                revision=self.model.config.encoder_revision,
                device=str(self.device),
                local_files_only=self.local_files_only,
            )
            encoder.float().eval()
            self._encoder = encoder
        return self._encoder

    def _encode(self, texts: list[str], batch_size: int) -> np.ndarray:
        vectors = self._ensure_encoder().encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        validated = _validate_vectors(vectors, self.model.config.embedding_dim)
        with np.errstate(over="ignore", under="ignore"):
            encoded = validated.astype(np.float32)
        if not np.isfinite(encoded).all():
            raise RuntimeError("encoder embeddings must be representable as finite float32 values")
        return encoded

    def _invert(self, vectors: np.ndarray, num_candidates: int, max_length: int | None,
                seed: int, batch_size: int) -> list[InvertResult]:
        results: list[InvertResult] = []
        for offset in range(0, len(vectors), batch_size):
            inputs = _normalised_tensor(vectors[offset:offset + batch_size], self.device)
            drafts = self.model.generate(
                inputs,
                num_candidates=int(num_candidates),
                max_length=int(max_length) if max_length is not None else None,
                seed=int(seed),
            )
            targets = inputs.cpu().numpy()
            if len(drafts) != len(targets):
                raise RuntimeError("decoder returned an unexpected number of results")
            groups = [draft["candidates"] for draft in drafts]
            if any(len(group) != num_candidates for group in groups):
                raise RuntimeError("decoder returned an unexpected number of candidates")
            texts = [candidate["text"] for group in groups for candidate in group]
            candidate_vectors = self._encode(texts, 32)
            if len(candidate_vectors) != len(texts):
                raise RuntimeError("encoder returned an unexpected number of embeddings")
            for index, group in enumerate(groups):
                start = index * num_candidates
                scores = candidate_vectors[start:start + num_candidates] @ targets[index]
                eligible = [i for i, item in enumerate(group) if is_fluent(item["text"])]
                selected = max(eligible or range(num_candidates), key=lambda i: scores[i])
                results.append(InvertResult(
                    text=group[selected]["text"],
                    candidates=[item["text"] for item in group],
                    scores=[float(score) for score in scores],
                    selected_index=selected,
                    stopped=group[selected]["stopped"],
                ))
        return results

    def invert_embedding(
        self,
        embedding: Sequence[float] | np.ndarray | torch.Tensor,
        num_candidates: int = 5,
        max_length: int | None = None,
        seed: int = 0,
    ) -> InvertResult:
        """Reconstruct one vector; normalize a nonzero input automatically."""
        _generation_options(num_candidates, max_length, seed, self.model.config.max_length)
        vectors = _validate_vectors(embedding, self.model.config.embedding_dim, single=True)
        with self._lock, torch.inference_mode():
            return self._invert(vectors, num_candidates, max_length, seed, 8)[0]

    def invert_embeddings(
        self,
        embeddings: Sequence[Sequence[float]] | np.ndarray | torch.Tensor,
        num_candidates: int = 5,
        max_length: int | None = None,
        seed: int = 0,
        batch_size: int = 8,
    ) -> list[InvertResult]:
        """Reconstruct a matrix in bounded batches, preserving input order.

        An empty batch returns an empty list without generation or encoding.
        ``max_length=None`` uses the model's configured token limit.
        """
        _generation_options(num_candidates, max_length, seed, self.model.config.max_length)
        _positive_integer(batch_size, "batch_size")
        vectors = _validate_vectors(embeddings, self.model.config.embedding_dim)
        with self._lock, torch.inference_mode():
            return self._invert(vectors, num_candidates, max_length, seed, batch_size)

    def invert_text(
        self,
        text: str,
        num_candidates: int = 5,
        max_length: int | None = None,
        seed: int = 0,
    ) -> InvertResult:
        """Embed one nonempty text with stock MPNet, then reconstruct it."""
        return self.invert_texts([text], num_candidates, max_length, seed)[0]

    def invert_texts(
        self,
        texts: Sequence[str],
        num_candidates: int = 5,
        max_length: int | None = None,
        seed: int = 0,
        batch_size: int = 8,
    ) -> list[InvertResult]:
        """Embed and reconstruct texts in bounded batches, preserving order."""
        _generation_options(num_candidates, max_length, seed, self.model.config.max_length)
        _positive_integer(batch_size, "batch_size")
        if isinstance(texts, str):
            raise ValueError("texts must be a sequence of strings; use invert_text for one string")
        values = list(texts)
        if any(not isinstance(text, str) or not text.strip() for text in values):
            raise ValueError("texts must contain nonempty strings")
        results: list[InvertResult] = []
        with self._lock, torch.inference_mode():
            for offset in range(0, len(values), batch_size):
                vectors = self._encode(values[offset:offset + batch_size], batch_size)
                if len(vectors) != len(values[offset:offset + batch_size]):
                    raise RuntimeError("encoder returned an unexpected number of embeddings")
                results.extend(self._invert(vectors, num_candidates, max_length, seed, batch_size))
        return results
