"""Shared helpers for the Aparecium benchmark harness.

The harness is repository-only (not part of the PyPI package). Dataset text is
written under data/bench/ (gitignored) and never committed; only manifests with
ids, hashes, and counts live in bench/manifests/.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, Iterator, List

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "bench"
MANIFEST_DIR = REPO_ROOT / "bench" / "manifests"

MPNET_ID = "sentence-transformers/all-mpnet-base-v2"
# Independent check encoder: different family and training data from MPNet.
INDEP_ID = "BAAI/bge-small-en-v1.5"

# Text-unit bounds in GPT-2 tokens. The v2 decoder generates up to 64 tokens,
# so reference texts must fit inside that budget for a fair comparison.
MIN_TOKENS = 6
MAX_TOKENS = 60

SEED = 20260923


def clean_text(s: str) -> str:
    """Light, reversible-looking cleanup applied to every unit before embedding."""
    s = html.unescape(html.unescape(s or ""))
    s = s.replace("#39;", "'").replace("#36;", "$").replace("quot;", '"')
    s = s.replace("\\", " ")
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def norm_key(s: str) -> str:
    """Aggressive normalization used only for dedup/leakage checks."""
    s = unicodedata.normalize("NFKC", s).lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def text_hash(s: str) -> str:
    return hashlib.sha1(norm_key(s).encode("utf-8")).hexdigest()[:16]


def stable_int(*parts: object) -> int:
    h = hashlib.sha1("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(h[:12], 16)


def in_subset(item_id: str, subset: str) -> bool:
    """Fixed, seeded subsets of the frozen test sets ("full" or "half")."""
    return subset == "full" or stable_int(item_id, SEED, "half") % 2 == 0


_TOKENIZER = None


def gpt2_len(s: str) -> int:
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer

        _TOKENIZER = AutoTokenizer.from_pretrained("gpt2")
    return len(_TOKENIZER(s, add_special_tokens=False)["input_ids"])


def fits(s: str) -> bool:
    if not s or len(s.split()) < 4:
        return False
    n = gpt2_len(s)
    return MIN_TOKENS <= n <= MAX_TOKENS


_SEGMENTER = None


def split_sentences(s: str) -> List[str]:
    global _SEGMENTER
    if _SEGMENTER is None:
        import pysbd

        _SEGMENTER = pysbd.Segmenter(language="en", clean=False)
    return [x.strip() for x in _SEGMENTER.segment(s) if x.strip()]


def lede(paragraph: str, max_sents: int = 3) -> str:
    """Greedy lede: first 1..max_sents sentences that fit the token budget."""
    sents = split_sentences(paragraph)
    out = ""
    for sent in sents[:max_sents]:
        cand = (out + " " + sent).strip()
        if gpt2_len(cand) > MAX_TOKENS:
            break
        out = cand
    return out


def write_jsonl(path: Path, rows: Iterable[Dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> Iterator[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def device_name() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
