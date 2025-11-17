from dataclasses import dataclass


@dataclass
class Config:
    seed: int = 1234
    device: str = "cuda"
    precision: str = "bf16"
    # data
    max_len: int = 64
    vocab: str = "gpt2"
    # model sizes
    emb_adapter_heads: int = 2
    emb_adapter_S: list = (8, 16)
    emb_adapter_D: int = 384
    decoder_layers: int = 12
    decoder_heads: int = 8
    decoder_d_model: int = 768
    decoder_ff: int = 3072
    dropout: float = 0.1
    # surrogate r
    r_text_layers: int = 4
    r_text_heads: int = 8
    r_text_d: int = 512
    r_ff: int = 2048
    r_temp: float = 0.07
    # training
    s1_batch: int = 512
    s1_lr: float = 3e-4
    s1_wd: float = 0.01
    s1_warmup: int = 10000
    s1_steps: int = 300000
    s1_label_smoothing: float = 0.1
    s2_lr: float = 5e-5
    s2_steps: int = 100000
    beam_width: int = 5
    rerank_every: int = 8
    rl_alpha: float = 0.7
    rl_beta: float = 0.1
    rl_gamma: float = 0.2
    entropy_bonus: float = 1e-3
