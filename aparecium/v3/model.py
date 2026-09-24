"""Aparecium v3: a pooled MPNet vector steers a small pretrained language model.

A small prefix mapper turns the 768-D vector into ``prefix_len`` soft tokens in
the language model's input space; the model then writes the text and ends it
with its own end-of-text token. Only the mapper is always trained; the language
model is either fine-tuned ("finetune"), adapted with LoRA ("lora"), or kept
frozen ("frozen"). The embedding model stays the user's stock encoder.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn


@dataclass
class V3Config:
    base: str = "openai-community/gpt2"
    mode: str = "finetune"  # finetune | lora | frozen
    prefix_len: int = 10
    emb_dim: int = 768
    mapper_hidden: int = 1024
    lora_r: int = 16
    max_len: int = 64  # text tokens, including the end-of-text token


class PrefixMapper(nn.Module):
    """768-D sentence vector -> prefix_len soft tokens of the LM's width."""

    def __init__(self, emb_dim: int, hidden: int, prefix_len: int, lm_dim: int):
        super().__init__()
        self.prefix_len, self.lm_dim = prefix_len, lm_dim
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, prefix_len * lm_dim),
        )

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        return self.net(e).view(-1, self.prefix_len, self.lm_dim)


class V3Model(nn.Module):
    def __init__(self, cfg: V3Config, lm=None, tokenizer=None):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.cfg = cfg
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(cfg.base)
        lm = lm or AutoModelForCausalLM.from_pretrained(cfg.base, dtype=torch.float32)
        if cfg.mode == "lora" and not hasattr(lm, "peft_config"):
            from peft import LoraConfig, get_peft_model

            lm = get_peft_model(lm, LoraConfig(r=cfg.lora_r, lora_alpha=2 * cfg.lora_r,
                                               lora_dropout=0.05, target_modules="all-linear"))
        elif cfg.mode == "frozen":
            lm.requires_grad_(False)
        self.lm = lm
        dim = self.lm.get_input_embeddings().embedding_dim
        self.mapper = PrefixMapper(cfg.emb_dim, cfg.mapper_hidden, cfg.prefix_len, dim)
        self.eos_id = self.tokenizer.eos_token_id

    # ------------------------------------------------------------------ training

    def encode_texts(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Tokenize targets as text + end-of-text, right-padded; padding is masked
        by the attention mask (never by token id, since pad == eos for GPT-2)."""
        ids = [self.tokenizer(t, add_special_tokens=False)["input_ids"][: self.cfg.max_len - 1] + [self.eos_id]
               for t in texts]
        width = max(len(x) for x in ids)
        input_ids = torch.full((len(ids), width), self.eos_id, dtype=torch.long)
        mask = torch.zeros((len(ids), width), dtype=torch.long)
        for i, x in enumerate(ids):
            input_ids[i, : len(x)] = torch.tensor(x)
            mask[i, : len(x)] = 1
        return {"input_ids": input_ids, "attention_mask": mask}

    def forward(self, e, input_ids, attention_mask):
        prefix = self.mapper(e)
        tokens = self.lm.get_input_embeddings()(input_ids)
        b, p = prefix.shape[:2]
        inputs = torch.cat([prefix, tokens], dim=1)
        attn = torch.cat([torch.ones(b, p, dtype=attention_mask.dtype, device=e.device), attention_mask], dim=1)
        labels = input_ids.masked_fill(attention_mask == 0, -100)
        labels = torch.cat([torch.full((b, p), -100, dtype=labels.dtype, device=e.device), labels], dim=1)
        return self.lm(inputs_embeds=inputs, attention_mask=attn, labels=labels)

    # ----------------------------------------------------------------- generation

    @torch.no_grad()
    def generate(self, e: torch.Tensor, n: int = 5, max_new_tokens: Optional[int] = None,
                 temperature: float = 0.8, top_p: float = 0.9, seed: int = 0) -> List[Dict]:
        """Per input vector: one greedy candidate plus n-1 sampled ones."""
        max_new_tokens = max_new_tokens or self.cfg.max_len
        prefix = self.mapper(e)
        attn = torch.ones(prefix.shape[:2], dtype=torch.long, device=e.device)
        common = dict(inputs_embeds=prefix, attention_mask=attn, max_new_tokens=max_new_tokens,
                      eos_token_id=self.eos_id, pad_token_id=self.eos_id)
        runs = [self.lm.generate(**common, do_sample=False)]
        if n > 1:
            torch.manual_seed(seed)
            sampled = self.lm.generate(**common, do_sample=True, temperature=temperature, top_p=top_p,
                                       num_return_sequences=n - 1)
            runs.append(sampled)
        out = []
        for i in range(prefix.shape[0]):
            seqs = [runs[0][i]] + ([runs[1][i * (n - 1) + k] for k in range(n - 1)] if n > 1 else [])
            cands = []
            for s in seqs:
                s = s.tolist()
                stopped = self.eos_id in s
                s = s[: s.index(self.eos_id)] if stopped else s
                cands.append({"text": self.tokenizer.decode(s, skip_special_tokens=True).strip(),
                              "stopped": stopped})
            out.append({"candidates": cands})
        return out

    # ------------------------------------------------------------------- save/load

    def save(self, path: str) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "v3_config.json").write_text(json.dumps(asdict(self.cfg), indent=1))
        torch.save(self.mapper.state_dict(), path / "mapper.pt")
        if self.cfg.mode == "finetune":
            self.lm.save_pretrained(path / "lm")
            self.tokenizer.save_pretrained(path / "lm")
        elif self.cfg.mode == "lora":
            self.lm.save_pretrained(path / "lora")

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "V3Model":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        path = Path(path)
        cfg = V3Config(**json.loads((path / "v3_config.json").read_text()))
        if cfg.mode == "finetune":
            lm = AutoModelForCausalLM.from_pretrained(path / "lm", dtype=torch.float32)
            tok = AutoTokenizer.from_pretrained(path / "lm")
        else:
            lm = AutoModelForCausalLM.from_pretrained(cfg.base, dtype=torch.float32)
            tok = AutoTokenizer.from_pretrained(cfg.base)
            if cfg.mode == "lora":
                from peft import PeftModel

                lm = PeftModel.from_pretrained(lm, path / "lora")
        model = cls(cfg, lm=lm, tokenizer=tok)
        model.mapper.load_state_dict(torch.load(path / "mapper.pt", map_location="cpu"))
        return model.to(device).eval()


_REPEAT = re.compile(r"(\b\w+\b(?:\W+\b\w+\b){2})(?:\W+\1){2,}", re.I)


def is_fluent(text: str) -> bool:
    """Cheap fluency floor: non-trivial length and no phrase looping 3+ times."""
    return len(text.split()) >= 3 and not _REPEAT.search(text)
