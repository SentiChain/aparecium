import torch, torch.nn as nn
from torch.nn import TransformerDecoder, TransformerDecoderLayer


class RealizerDecoder(nn.Module):
    """Transformer decoder that cross‑attends to pseudo‑sequence from EmbAdapter.
    For simplicity, we use PyTorch's TransformerDecoder with learned cross‑kv via a small projection.
    memory_dim is the channel dimension of EmbAdapter outputs (default 384).
    """

    def __init__(
        self,
        vocab_size,
        d_model=768,
        n_head=8,
        n_layer=12,
        d_ff=3072,
        dropout=0.1,
        memory_dim=384,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = SinusoidalPositionalEncoding(d_model)
        layer = TransformerDecoderLayer(
            d_model, n_head, d_ff, dropout, batch_first=True
        )
        self.decoder = TransformerDecoder(layer, n_layer)
        self.kv_proj = nn.Linear(
            memory_dim, d_model
        )  # project adapter features (B,S,memory_dim) -> (B,S,d_model)
        self.ln = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self,
        input_ids,
        memory,
        attn_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        x = self.embed(input_ids) + self.pos(
            input_ids.shape[1], device=input_ids.device
        )
        memory = self.kv_proj(memory)
        # Ensure causal masking for autoregressive decoding. If an explicit attn_mask was
        # provided by the caller, respect it; otherwise generate a standard square
        # subsequent mask of shape (T, T).
        if attn_mask is None:
            T = input_ids.shape[1]
            # Causal mask: 0 on/below diagonal, -inf above
            attn_mask = torch.triu(
                torch.full((T, T), float("-inf"), device=input_ids.device), diagonal=1
            )
        x = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=attn_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        x = self.ln(x)
        logits = self.lm_head(x)
        return logits


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, seq_len, device=None):
        pe = _build_sinusoidal_embeddings(seq_len, self.d_model, device=device)
        return pe


def _build_sinusoidal_embeddings(n_pos, dim, device=None):
    import math, torch

    pe = torch.zeros(1, n_pos, dim, device=device)
    position = torch.arange(0, n_pos, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device).float() * (-math.log(10000.0) / dim)
    )
    pe[..., 0::2] = torch.sin(position * div_term)
    pe[..., 1::2] = torch.cos(position * div_term)
    return pe
