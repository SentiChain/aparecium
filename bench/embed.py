"""Embed text with the stock, frozen encoders exactly as a user would.

MPNet vectors of the reference texts are the INPUTS given to every inverter,
produced with plain SentenceTransformer(...).encode(..., normalize_embeddings=True)
(the model already ends in a Normalize layer, so this matches the default call).

Everything is embedded on CPU so frozen vectors are reproducible across
machines; caches are keyed by a digest of the exact texts, never by length.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import numpy as np

from .common import INDEP_ID, MPNET_ID

EMBED_DEVICE = "cpu"


@lru_cache(maxsize=4)
def encoder(model_id: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_id, device=EMBED_DEVICE)


def embed(texts: List[str], model_id: str = MPNET_ID, batch_size: int = 64) -> np.ndarray:
    vecs = encoder(model_id).encode(
        list(texts),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 2000,
    )
    return vecs.astype(np.float32)


def tag(model_id: str) -> str:
    return {MPNET_ID: "mpnet", INDEP_ID: "bge"}.get(model_id, model_id.replace("/", "_"))


def texts_digest(model_id: str, texts: List[str]) -> str:
    h = hashlib.sha256(model_id.encode("utf-8"))
    for t in texts:
        h.update(b"\x00" + t.encode("utf-8"))
    return h.hexdigest()


def cached(path: Path, texts: List[str], model_id: str = MPNET_ID) -> np.ndarray:
    """Embed once and cache next to the jsonl (foo.mpnet.npy + foo.mpnet.json digest)."""
    out = path.with_suffix(f".{tag(model_id)}.npy")
    side = path.with_suffix(f".{tag(model_id)}.json")
    digest = texts_digest(model_id, texts)
    if out.exists() and side.exists() and json.loads(side.read_text()).get("texts_sha256") == digest:
        return np.load(out)
    vecs = embed(texts, model_id)
    np.save(out, vecs)
    side.write_text(json.dumps({"model": model_id, "n": len(texts), "texts_sha256": digest,
                                "vectors_sha256": array_sha256(vecs)}, indent=1))
    return vecs


def array_sha256(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def encoder_info() -> Dict:
    import sentence_transformers
    import torch
    from huggingface_hub import HfApi

    return {
        "device": EMBED_DEVICE,
        "torch": torch.__version__,
        "sentence_transformers": sentence_transformers.__version__,
        "models": {m: HfApi().model_info(m).sha for m in (MPNET_ID, INDEP_ID)},
    }
