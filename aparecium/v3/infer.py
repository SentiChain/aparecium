"""Invert pooled MPNet vectors with a trained v3 model.

Candidates from the language model are re-embedded with the same stock MPNet
the user embedded with, and the fluent candidate closest to the input vector
wins. The encoder is only used to score; it is never trained.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from .model import V3Model, is_fluent

MPNET_ID = "sentence-transformers/all-mpnet-base-v2"


class V3Inverter:
    def __init__(self, path: str, device: str = "cpu", encoder_device: Optional[str] = None):
        from sentence_transformers import SentenceTransformer

        self.device = torch.device(device)
        self.model = V3Model.load(path, device=device)
        self.encoder = SentenceTransformer(MPNET_ID, device=encoder_device or device)

    def invert(self, vectors: Sequence[Sequence[float]], n: int = 5, seed: int = 0) -> List[Dict]:
        e = torch.tensor(np.asarray(vectors, dtype=np.float32), device=self.device)
        e = torch.nn.functional.normalize(e, dim=-1)
        drafts = self.model.generate(e, n=n, seed=seed)
        texts = [c["text"] for d in drafts for c in d["candidates"]]
        cand_vecs = self.encoder.encode(texts, convert_to_numpy=True, normalize_embeddings=True,
                                        show_progress_bar=False)
        target = e.cpu().numpy()
        out, k = [], 0
        for i, d in enumerate(drafts):
            cands = d["candidates"]
            sims = cand_vecs[k:k + len(cands)] @ target[i]
            k += len(cands)
            fluent = [j for j, c in enumerate(cands) if is_fluent(c["text"])]
            pool = fluent or list(range(len(cands)))
            best = max(pool, key=lambda j: sims[j])
            out.append({
                "text": cands[best]["text"],
                "stopped": cands[best]["stopped"],
                "candidates": [c["text"] for c in cands],
                "scores": [float(s) for s in sims],
            })
        return out
