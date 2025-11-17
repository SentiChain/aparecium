import torch, torch.nn as nn
from typing import Dict, Any


class Sketcher(nn.Module):
    """Predicts plan attributes from pooled embedding e.
    This is a simple multi‑label MLP; you can extend with pointer heads.
    """

    def __init__(self, d_in=768, hidden=768, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # Multi‑label heads
        self.has_url = nn.Linear(hidden, 1)
        self.ticker_logits = nn.Linear(hidden, 1)  # proxy signal
        self.addr_logits = nn.Linear(hidden, 1)  # proxy signal
        self.num_logits = nn.Linear(hidden, 1)  # proxy signal

    def forward(self, e):
        h = self.net(e)
        return {
            "has_url": torch.sigmoid(self.has_url(h)),
            "ticker": torch.sigmoid(self.ticker_logits(h)),
            "address": torch.sigmoid(self.addr_logits(h)),
            "number": torch.sigmoid(self.num_logits(h)),
        }

    def decode_plan(self, e) -> Dict[str, Any]:
        out = self.forward(e)
        plan = {
            "need_url": bool(
                (out["has_url"] > 0.5).item()
                if out["has_url"].ndim == 2
                else out["has_url"] > 0.5
            ),
        }
        return plan
