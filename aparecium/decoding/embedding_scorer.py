"""
Embedding scorer utilities for decoding.

Provides `MPNetEmbeddingScorer` to re-embed candidate strings with the same
MPNet family model used to encode source memory, enabling outer-loop rescoring
by cosine similarity.
"""

from typing import List, Optional
import torch  # type: ignore
from transformers import AutoTokenizer, AutoModel  # type: ignore

from .utils import mean_pool


class MPNetEmbeddingScorer:
    """
    Encode texts with MPNet and return mask-aware pooled vectors on a device.

    Intended for use during outer-loop rescoring.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        device: Optional[torch.device] = None,
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode_and_pool(self, texts: List[str]) -> torch.Tensor:
        """
        Encode and pool texts to MPNet sentence vectors.

        Args:
                texts: list of strings to encode

        Returns:
                Tensor of shape (B, D), where D is the model hidden size.
        """
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=384,
        )
        input_ids = inputs["input_ids"].to(self.device)
        attn = inputs["attention_mask"].to(self.device)
        outputs = self.model(input_ids=input_ids, attention_mask=attn)
        last_hidden = outputs.last_hidden_state  # (B, T, D)
        pooled = mean_pool(last_hidden, attn)  # (B, D)
        return pooled
