"""Portable dense token-state decoding with verified safetensors weights."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig, TENSOR_SHAPES, TOKENIZER_FILES


_TOKENIZER_SIZE_LIMITS = {
    "tokenizer.json": 16 * 1024 * 1024,
    "tokenizer_config.json": 1024 * 1024,
    "special_tokens_map.json": 65536,
    "vocab.txt": 4 * 1024 * 1024,
}
_WEIGHT_BYTES = sum(math.prod(shape) * 4 for shape in TENSOR_SHAPES.values())


def _get_device(explicit: str | torch.device | None) -> torch.device:
    if explicit is None:
        return torch.device("cpu")
    try:
        device = torch.device(explicit)
    except (TypeError, RuntimeError) as exc:
        raise ValueError("device must be cpu or mps") from exc
    if device.type not in {"cpu", "mps"} or device.index is not None:
        raise ValueError("device must be cpu or mps without a device index")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is not available on this system")
    return device


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verified_file(root: Path, name: str, digest: str, maximum: int, minimum: int = 1) -> Path:
    path = root / name
    if not path.is_file():
        raise FileNotFoundError(f"Required model file is missing: {name}")
    if not minimum <= path.stat().st_size <= maximum:
        raise ValueError(f"Invalid model file size: {name}")
    if _file_sha256(path) != digest:
        raise ValueError(f"Model file checksum mismatch: {name}")
    return path


class MatrixDecoder(nn.Module):
    """Apply a fixed dense, GELU, LayerNorm, and vocabulary projection."""

    def __init__(self, config: ModelConfig, tensors: dict[str, torch.Tensor], tokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.register_buffer("vocabulary", tensors["vocabulary"], persistent=True)
        self.dense = nn.Linear(config.state_width, config.state_width, dtype=torch.float32, device="meta")
        self.layer_norm = nn.LayerNorm(config.state_width, eps=config.layer_norm_eps, dtype=torch.float32, device="meta")
        self.vocabulary_bias = nn.Parameter(torch.empty(config.vocabulary_size, dtype=torch.float32, device="meta"))
        self.load_state_dict(tensors, strict=True, assign=True)
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):
        if mode is not False:
            raise ValueError("The token-state decoder supports inference only")
        return super().train(False)

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu") -> "MatrixDecoder":
        """Validate the complete bundle before constructing its local tokenizer."""
        from safetensors import safe_open
        from transformers import MPNetTokenizer

        device = _get_device(device)
        root = Path(path)
        if not root.is_dir():
            raise ValueError("model_path must be a model bundle directory")
        config = ModelConfig.load(root / "config.json")
        weights = _verified_file(
            root, config.weights_file, config.weights_sha256, _WEIGHT_BYTES + 65536, _WEIGHT_BYTES + 8
        )
        for name in TOKENIZER_FILES:
            _verified_file(root, name, config.tokenizer_files_sha256[name], _TOKENIZER_SIZE_LIMITS[name])
        tensors = {}
        try:
            with safe_open(str(weights), framework="pt", device="cpu") as saved:
                if set(saved.keys()) != set(TENSOR_SHAPES):
                    raise ValueError("Model weights must contain exactly the required tensors")
                for name, shape in TENSOR_SHAPES.items():
                    view = saved.get_slice(name)
                    if tuple(view.get_shape()) != shape or view.get_dtype() != "F32":
                        raise ValueError(f"Invalid tensor shape or dtype: {name}")
                for name in TENSOR_SHAPES:
                    tensor = saved.get_tensor(name)
                    if not torch.isfinite(tensor).all().item():
                        raise ValueError(f"Model tensor contains non-finite values: {name}")
                    if hashlib.sha256(tensor.numpy().tobytes()).hexdigest() != config.tensor_sha256[name]:
                        raise ValueError(f"Model tensor checksum mismatch: {name}")
                    tensors[name] = tensor
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("Invalid safetensors model weights") from exc
        tokenizer = MPNetTokenizer.from_pretrained(
            str(root), use_fast=True, local_files_only=True, trust_remote_code=False
        )
        if not tokenizer.is_fast or len(tokenizer) != config.vocabulary_size:
            raise ValueError("Tokenizer vocabulary does not match the model configuration")
        for name in ("pad_token_id", "bos_token_id", "eos_token_id", "unk_token_id"):
            if type(getattr(tokenizer, name, None)) is not int or getattr(tokenizer, name) != getattr(config, name):
                raise ValueError(f"Tokenizer {name} does not match the model configuration")
        return cls(config, tensors, tokenizer).to(device=device, dtype=torch.float32).eval()

    @torch.inference_mode()
    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """Return native float32 vocabulary logits for one bounded row chunk."""
        if (
            not isinstance(states, torch.Tensor)
            or states.ndim != 2
            or states.shape[1] != self.config.state_width
            or not 1 <= states.shape[0] <= self.config.max_sequence_length
            or states.dtype != torch.float32
            or not torch.isfinite(states).all().item()
        ):
            raise ValueError("states must be finite float32 (1..256, 768) token rows")
        with torch.autocast(device_type=self.vocabulary.device.type, enabled=False):
            hidden = self.layer_norm(F.gelu(self.dense(states), approximate="none"))
            logits = F.linear(hidden, self.vocabulary, self.vocabulary_bias)
        if not torch.isfinite(logits).all().item():
            raise ValueError("Token-state values produced non-finite decoder logits")
        return logits

    @torch.inference_mode()
    def predict_token_ids(self, states: torch.Tensor, token_batch_size: int = 256) -> list[int]:
        """Select one token per input row without keeping a full sequence of logits."""
        if type(token_batch_size) is not int or not 1 <= token_batch_size <= 256:
            raise ValueError("token_batch_size must be an integer between 1 and 256")
        if (
            not isinstance(states, torch.Tensor)
            or states.ndim != 2
            or states.shape[1] != self.config.state_width
            or not 1 <= states.shape[0] <= self.config.max_sequence_length
            or states.dtype != torch.float32
            or not torch.isfinite(states).all().item()
        ):
            raise ValueError("states must be finite float32 (1..256, 768) token rows")
        result = []
        for start in range(0, len(states), token_batch_size):
            rows = states[start:start + token_batch_size].to(device=self.vocabulary.device, dtype=torch.float32)
            result.extend(self(rows).argmax(dim=-1).cpu().tolist())
        return result
