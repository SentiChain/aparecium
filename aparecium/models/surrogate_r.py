import torch, torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TextEncoder(nn.Module):
    def __init__(
        self, vocab_size, d_model=512, n_head=8, n_layer=4, d_ff=2048, dropout=0.1
    ):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        layer = TransformerEncoderLayer(
            d_model, n_head, d_ff, dropout, batch_first=True
        )
        self.enc = TransformerEncoder(layer, n_layer)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, input_ids, attention_mask=None):
        x = self.emb(input_ids)
        x = self.enc(
            x,
            src_key_padding_mask=(
                (attention_mask == 0) if attention_mask is not None else None
            ),
        )
        x = self.ln(x)
        x = x.mean(dim=1)  # simple mean pool
        return x


class EmbProj(nn.Module):
    def __init__(self, dim_in=768, dim_out=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.GELU(),
            nn.Linear(dim_out, dim_out),
        )

    def forward(self, e):
        return self.net(e)


class SurrogateR(nn.Module):
    """Predicts cosine( MPNet(text), e ) approximately."""

    def __init__(
        self,
        vocab_size,
        d_text=512,
        n_head=8,
        n_layer=4,
        d_ff=2048,
        dim_in_e=768,
        temp=0.07,
        dropout=0.1,
    ):
        super().__init__()
        self.text = TextEncoder(vocab_size, d_text, n_head, n_layer, d_ff, dropout)
        self.emb = EmbProj(dim_in=dim_in_e, dim_out=d_text)
        self.temp = nn.Parameter(torch.tensor(temp))
        self.norm = nn.LayerNorm(d_text)

    def forward(self, input_ids, attention_mask, e):
        t = self.text(input_ids, attention_mask)
        v = self.emb(e)
        t = self.norm(t)
        v = self.norm(v)
        # cosine
        cos = torch.nn.functional.cosine_similarity(t, v, dim=-1)
        return cos
