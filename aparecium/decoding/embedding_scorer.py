import torch
from sentence_transformers import SentenceTransformer


class MPNetEmbeddingScorer:
    """Utility to encode texts with all-mpnet-base-v2 and return pooled 768-D embeddings.

    encode_and_pool(texts) -> torch.FloatTensor of shape (N, 768), L2-normalized.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        device: str | None = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)
        self.model.eval()

    @torch.no_grad()
    def encode_and_pool(self, texts: list[str]) -> torch.Tensor:
        emb = self.model.encode(
            texts,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        if not isinstance(emb, torch.Tensor):
            emb = torch.tensor(emb)
        return emb  # (N, 768)
