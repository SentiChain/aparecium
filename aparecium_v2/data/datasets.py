import json, os, glob
from typing import List, Dict, Any
import numpy as np
import torch
from torch.utils.data import Dataset
from .plans import extract_plan


class TweetsShardDataset(Dataset):
    """Reads JSONL shards with fields:
    {
      "text": str,
      "embedding": [float; 768],  # pooled MPNet v2
      "plan": { ... }             # optional; will be auto if missing
    }
    """

    def __init__(self, shards_dir: str, pattern: str = "*.jsonl", max_len: int = 64):
        self.files = sorted(glob.glob(os.path.join(shards_dir, pattern)))
        assert self.files, f"No shard files found in {shards_dir}/{pattern}"
        self.index = []
        for fi, path in enumerate(self.files):
            with open(path, "r", encoding="utf-8") as f:
                for off, _ in enumerate(f):  # line index
                    self.index.append((fi, off))
        self.max_len = max_len

    def __len__(self):
        return len(self.index)

    def _read_line(self, file_path, lineno):
        with open(file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i == lineno:
                    return json.loads(line)
        raise IndexError("Line not found")  # shouldn't happen

    def __getitem__(self, i):
        fi, off = self.index[i]
        path = self.files[fi]
        rec = self._read_line(path, off)
        text = rec["text"]
        e = np.asarray(rec["embedding"], dtype=np.float32)  # (768,)
        plan = rec.get("plan") or extract_plan(text)
        return {"text": text, "e": e, "plan": plan}


def collate(tokenizer, batch, max_len: int = 64):
    texts = [b["text"] for b in batch]
    e = torch.tensor(np.stack([b["e"] for b in batch], axis=0))
    toks = tokenizer(
        texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt"
    )
    return {
        "e": e,
        "input_ids": toks["input_ids"],
        "attention_mask": toks["attention_mask"],
        "texts": texts,
        "plans": [b["plan"] for b in batch],
    }
