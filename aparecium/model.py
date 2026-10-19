"""Reconstruct text from sentence embeddings with a pretrained decoder."""

from __future__ import annotations

import json
import math
import re
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from torch import nn

from .config import ModelConfig


_RNG_LOCK = threading.RLock()


@contextmanager
def _seeded_sampling(seed: int, device: torch.device):
    """Seed sampling and restore the caller's random number generators."""
    with _RNG_LOCK:
        cpu_state = torch.random.get_rng_state()
        accelerator_state = None
        cuda_index = None
        if device.type == "cuda":
            cuda_index = device.index if device.index is not None else torch.cuda.current_device()
            accelerator_state = torch.cuda.get_rng_state(cuda_index)
        elif device.type == "mps":
            accelerator_state = torch.mps.get_rng_state()
        try:
            torch.random.default_generator.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.default_generators[cuda_index].manual_seed(seed)
            elif device.type == "mps":
                torch.mps.manual_seed(seed)
            yield
        finally:
            torch.random.set_rng_state(cpu_state)
            if device.type == "cuda":
                torch.cuda.set_rng_state(accelerator_state, cuda_index)
            elif device.type == "mps":
                torch.mps.set_rng_state(accelerator_state)


class PrefixMapper(nn.Module):
    """Map a sentence embedding to soft tokens in the decoder's input space."""

    def __init__(self, embedding_dim: int, hidden: int, prefix_length: int, lm_dim: int):
        super().__init__()
        self.prefix_length = prefix_length
        self.lm_dim = lm_dim
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, prefix_length * lm_dim),
        )

    def forward(self, vectors: torch.Tensor) -> torch.Tensor:
        return self.net(vectors).view(-1, self.prefix_length, self.lm_dim)


class PrefixDecoder(nn.Module):
    """GPT-2 with a sentence-embedding prefix mapper and LoRA adapter."""

    def __init__(self, config: ModelConfig, lm: nn.Module, tokenizer):
        super().__init__()
        self.config = config
        self.lm = lm
        self.tokenizer = tokenizer
        width = lm.get_input_embeddings().embedding_dim
        self.mapper = PrefixMapper(config.embedding_dim, config.mapper_hidden, config.prefix_length, width)
        self.eos_id = tokenizer.eos_token_id
        if type(self.eos_id) is not int or self.eos_id < 0:
            raise ValueError("The tokenizer must define an end-of-text token")

    @classmethod
    def load(
        cls, path: Union[str, Path], device: Union[str, torch.device] = "cpu", *, local_files_only: bool = False
    ) -> "PrefixDecoder":
        """Load safe tensor weights and the pinned pretrained dependencies."""
        from peft import PeftModel
        from safetensors.torch import load_file
        from transformers import AutoModelForCausalLM, AutoTokenizer

        path = Path(path)
        config = ModelConfig.load(path / "config.json")
        for relative in ("mapper.safetensors", "adapter/adapter_model.safetensors", "adapter/adapter_config.json"):
            if not (path / relative).is_file():
                raise FileNotFoundError(f"Required model file is missing: {relative}")
        adapter = json.loads((path / "adapter" / "adapter_config.json").read_text(encoding="utf-8"))
        if (
            not isinstance(adapter, dict)
            or adapter.get("peft_type") != "LORA"
            or adapter.get("r") != config.lora_rank
            or adapter.get("base_model_name_or_path") != config.base_model
            or adapter.get("revision") != config.base_revision
        ):
            raise ValueError("Adapter configuration does not match the model configuration")
        tokenizer = AutoTokenizer.from_pretrained(
            config.base_model, revision=config.base_revision, local_files_only=local_files_only
        )
        lm = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            revision=config.base_revision,
            dtype=torch.float32,
            use_safetensors=True,
            local_files_only=local_files_only,
        )
        lm = PeftModel.from_pretrained(lm, path / "adapter", is_trainable=False, local_files_only=True)
        model = cls(config, lm=lm, tokenizer=tokenizer)
        model.mapper.load_state_dict(load_file(str(path / "mapper.safetensors"), device="cpu"), strict=True)
        return model.to(device=device, dtype=torch.float32).eval()

    @torch.no_grad()
    def generate(
        self,
        vectors: torch.Tensor,
        num_candidates: int = 5,
        max_length: Optional[int] = None,
        seed: int = 0,
        temperature: float = 0.8,
        top_p: float = 0.9,
    ) -> List[Dict]:
        """Generate one greedy candidate and the remaining candidates by sampling."""
        if type(num_candidates) is not int or not 1 <= num_candidates <= 32:
            raise ValueError("num_candidates must be an integer between 1 and 32")
        if max_length is None:
            max_length = self.config.max_length
        if type(max_length) is not int or not 1 <= max_length <= self.config.max_length:
            raise ValueError(f"max_length must be an integer between 1 and {self.config.max_length}")
        if type(seed) is not int or not 0 <= seed <= 2**63 - 1:
            raise ValueError("seed must be an integer between 0 and 2**63 - 1")
        for name, value in (("temperature", temperature), ("top_p", top_p)):
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be greater than zero and at most one")
        if not isinstance(vectors, torch.Tensor) or vectors.ndim != 2:
            raise ValueError("vectors must be a two-dimensional torch.Tensor")
        if vectors.shape[1] != self.config.embedding_dim:
            raise ValueError(f"vectors must have {self.config.embedding_dim} columns")
        if not vectors.is_floating_point() or not torch.isfinite(vectors).all().item():
            raise ValueError("vectors must contain finite floating-point values")
        if vectors.shape[0] == 0:
            return []
        weight = self.mapper.net[0].weight
        vectors = vectors.to(device=weight.device, dtype=weight.dtype)
        if not torch.isfinite(vectors).all().item():
            raise ValueError("vectors must be representable as finite model inputs")
        prefix = self.mapper(vectors)
        attention_mask = torch.ones(prefix.shape[:2], dtype=torch.long, device=prefix.device)
        common = dict(
            inputs_embeds=prefix,
            attention_mask=attention_mask,
            max_new_tokens=max_length,
            eos_token_id=self.eos_id,
            pad_token_id=self.eos_id,
        )
        greedy = self.lm.generate(**common, do_sample=False)
        sampled = None
        if num_candidates > 1:
            with _seeded_sampling(seed, prefix.device):
                sampled = self.lm.generate(
                    **common,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    num_return_sequences=num_candidates - 1,
                )
        result = []
        for index in range(prefix.shape[0]):
            sequences = [greedy[index]]
            if sampled is not None:
                start = index * (num_candidates - 1)
                sequences.extend(sampled[start : start + num_candidates - 1])
            candidates = []
            for sequence in sequences:
                tokens = sequence.tolist()
                stopped = self.eos_id in tokens
                if stopped:
                    tokens = tokens[: tokens.index(self.eos_id)]
                candidates.append(
                    {"text": self.tokenizer.decode(tokens, skip_special_tokens=True).strip(), "stopped": stopped}
                )
            result.append({"candidates": candidates})
        return result


_REPEAT = re.compile(r"(\b\w+\b(?:\W+\b\w+\b){2})(?:\W+\1){2,}", re.I)


def is_fluent(text: str) -> bool:
    """Require at least three words and reject repeated three-word phrases."""
    return len(text.split()) >= 3 and not _REPEAT.search(text)
