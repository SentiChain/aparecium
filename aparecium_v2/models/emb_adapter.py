import torch, torch.nn as nn


class EmbAdapter(nn.Module):
    """Multi‑channel adapter: projects (B,768) into pseudo‑sequence (B,S_total,D)."""

    def __init__(self, d_in=768, D=384, S_list=(8, 16), dropout=0.1):
        super().__init__()
        self.proj = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_in, S * D),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(S * D, S * D),
                )
                for S in S_list
            ]
        )
        self.freq = nn.ParameterList(
            [nn.Parameter(torch.randn(S, D) * 0.02) for S in S_list]
        )
        self.norm = nn.LayerNorm(D)
        self.S_list, self.D = S_list, D

    def forward(self, e):  # e: (B, 768)
        seqs = []
        for mlp, f, S in zip(self.proj, self.freq, self.S_list):
            z = mlp(e)  # (B, S*D)
            z = z.view(-1, S, self.D) + f  # add frequency mixing
            z = self.norm(z)
            seqs.append(z)
        H = torch.cat(seqs, dim=1)  # (B, S_total, D)
        return H
