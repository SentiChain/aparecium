# Aparecium v2 – Pooled MPNet Reverser

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Aparecium v2 is a Python package for **reconstructing crypto‑domain social‑media posts from a single pooled embedding vector**.
It is the pooled‑embedding counterpart to the original token‑level seq2seq model
[`SentiChain/aparecium-seq2seq-reverser`](https://huggingface.co/SentiChain/aparecium-seq2seq-reverser),
but with a stricter input contract:

- **Input**: one 768‑D pooled vector from `sentence-transformers/all-mpnet-base-v2` (not a token‑level matrix).
- **Output**: natural‑language text that matches the crypto market context of the embedding.

This package contains:

- Low‑level components: EmbAdapter, Sketcher, Decoder, surrogate similarity scorer `r(x, e)`.
- Training scripts for supervised S1 and optional SCST (S2) fine‑tuning.
- A high‑level `Aparecium` wrapper for easy use from PyPI and Hugging Face Hub.

---

## Features

- **Pooled‑only embedding reversal**: works directly from a single pooled MPNet embedding (size 768), no token‑level memory required.
- **Crypto‑domain specialization**: trained on synthetic crypto social‑media posts (markets, DeFi, L2s, MEV, NFTs, governance).
- **Modern architecture**:
  - Multi‑channel EmbAdapter (pooled → pseudo‑sequence memory).
  - Sketcher plan head (optional constraints from simple signals).
  - Transformer decoder, surrogate similarity scorer, and beam‑search reranking.
- **High‑level API**:
  - `Aparecium.invert_embedding(...)` for direct vector → text.
  - `Aparecium.invert_text(...)` for raw text → embed → invert (for diagnostics).
- **Service‑ready**:
  - FastAPI inference server with `/invert` endpoint, suitable for batch/online use.

## Limitations & Caveats

- Reconstruction is **not exact**: outputs preserve semantic gist and entities but may differ in wording or style.
- Quality depends on:
  - Encoder alignment (`sentence-transformers/all-mpnet-base-v2`),
  - Domain match (crypto / finance social‑media posts),
  - Decode settings (beam size, constraints, rerank weights).
- Data are **synthetic** crypto market posts, not real social‑media timelines; there may be domain‑shift in practice.
- Do **not** use this model to attempt to reconstruct sensitive or personally identifiable content from embeddings.

---

## Model Architecture (v2)

At a high level, Aparecium v2 reverses a pooled vector \( e \in \mathbb{R}^{768} \) as follows:

1. **EmbAdapter**: `e → H`
   - Takes a pooled MPNet embedding and produces a multi‑scale pseudo‑sequence memory `H ∈ R^{B × S × D}`.
2. **Sketcher** (optional at inference):
   - Predicts simple crypto‑domain signals, such as presence of URLs or basic plan fields.
3. **RealizerDecoder** (Transformer decoder):
   - GPT‑style transformer decoder with cross‑attention over `H`.
   - Typical configuration:
     - `d_model = 768`
     - `n_layer = 12`
     - `n_head = 8`
     - `d_ff = 3072`
4. **Surrogate scorer `r(x, e)`**:
   - Neural surrogate that approximates cosine similarity between the MPNet embedding of the generated text and the target embedding `e`.
   - Used for sequence‑level reranking.
5. **Decoding**:
   - Deterministic beam search or stochastic sampling.
   - Optional constraints (tickers/hashtags/amounts) and surrogate‑based rerank.

The v2 S1 checkpoint released on Hugging Face at
[`SentiChain/aparecium-v2-pooled-reverser`](https://huggingface.co/SentiChain/aparecium-v2-pooled-reverser)
contains the EmbAdapter, Sketcher, Decoder, tokenizer name, and (optionally) surrogate `r` state.

---

## Installation

### From PyPI

Once published as the new major version, you will be able to install with:

```bash
pip install aparecium
```

### From Source (this repo)

```bash
git clone https://github.com/SentiChain/aparecium.git
cd aparecium
pip install -e .
```

This installs the `aparecium` package (v2 pooled‑only variant) in editable mode for development and experiments.

---

## Quick Start (High‑Level API)

### 1. Invert a pooled embedding from Python

The HF v2 checkpoint lives at
[`SentiChain/aparecium-v2-pooled-reverser`](https://huggingface.co/SentiChain/aparecium-v2-pooled-reverser).  
The `Aparecium` wrapper downloads it automatically and exposes a simple interface:

```python
from aparecium import Aparecium
from sentence_transformers import SentenceTransformer

# 1) Embed a crypto-domain social-media post with pooled MPNet
encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
text = "Bitcoin ETF inflows hit a new weekly high as markets turn risk-on."
e = encoder.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]  # shape (768,)

# 2) Load Aparecium v2 (S1 baseline) from Hugging Face
model = Aparecium()  # defaults to SentiChain/aparecium-v2-pooled-reverser, aparecium_v2_s1.pt

# 3) Invert the pooled embedding
res = model.invert_embedding(e, beam=5, max_len=64)
print("Reconstruction:", res.text)
print("Candidates:", res.candidates)
```

### 2. End‑to‑end inversion from raw text

This is mostly useful for **diagnostics** (how much information is lost by pooling):

```python
from aparecium import Aparecium

model = Aparecium()
text = "Ethereum L2 blob fees spiked after EIP-4844; MEV still shapes order flow."
out = model.invert_text(text, beam=5, max_len=64)
print(out.text)
```

Internally this calls the same MPNet encoder you would use upstream and then runs the inversion pipeline.

### 3. CLI usage

You can also use the package as a simple CLI:

```bash
echo "Macro: DXY rallies while risk assets chop; crypto narratives rotate to AI tokens." | \
  python -m aparecium
```

The CLI uses the default HF repo and S1 checkpoint and prints one reconstructed text to stdout.

---

## Training & Pipeline (v2)

The training and data‑prep scripts live inside the package under `aparecium.aparecium`:

- `aparecium/aparecium/scripts/embed_mpnet.py` – embed raw posts into pooled MPNet vectors.
- `aparecium/aparecium/train/train_s1_supervised.py` – S1 supervised training.
- `aparecium/aparecium/train/train_surrogate_r.py` – surrogate `r` training.
- `aparecium/aparecium/train/train_s2_scst.py` – optional SCST RL fine‑tuning.
- `aparecium/aparecium/infer/service.py` – FastAPI inference service.
- `aparecium/aparecium/data/*.py` – dataset and crypto plan utilities.

Example S1 training command (from the `aparecium` project root):

```bash
python -m aparecium.aparecium.train.train_s1_supervised \
  --shards ./data/train \
  --val_shards ./data/val \
  --save_dir ./checkpoints \
  --batch_size 64 \
  --epochs 1 \
  --steps 6000 \
  --warmup_steps 1000 \
  --lr 3e-4 \
  --max_len 96 \
  --device cuda \
  --log_every 50
```

> Note: for most users of the PyPI package, you **do not** need to run training. You can simply use the HF checkpoint with `Aparecium`.

---

## Inference Service

For higher‑throughput use, you can run the FastAPI service:

```bash
python -m aparecium.aparecium.infer.service --ckpt checkpoints/aparecium_v2_s1.pt
```

Then POST to `/invert`:

```json
{
  "embedding": [0.123, 0.456, "...", 0.789],
  "deterministic": true,
  "beam": 5,
  "max_len": 64,
  "constraints": true,
  "final_mpnet": true
}
```

The response includes:

- `text`: top‑1 reconstruction,
- `candidates`: all beam candidates,
- `scores.lm_logp[]`: language‑model log‑prob scores,
- `scores.cos_mpnet[]` (if `final_mpnet=true`): MPNet cosine per candidate,
- `plan`: optional extracted or predicted plan information.

---

## Model Input Contract & Defaults

- Input to the v2 reverser is a **pooled MPNet vector** with shape `(768,)` (L2‑normalized recommended).
- Recommended encoder: `sentence-transformers/all-mpnet-base-v2`.
- Suggested decode defaults for general use:
  - Beam size: `beam=5`
  - Max length: `max_len≈64–96`
  - Determinism: `deterministic beam` (via `deterministic_beam_search`)
  - Rerank weight for surrogate `r`: `alpha≈1.0–1.5`
  - Optional: use constraints for tickers/hashtags/amounts if plans are available.

This differs from the v1 seq2seq model, which expects a token‑level `(seq_len, 768)` matrix and uses a slightly different decode configuration (see the v1 model card at
[`SentiChain/aparecium-seq2seq-reverser`](https://huggingface.co/SentiChain/aparecium-seq2seq-reverser)).

---

## Requirements

At runtime, Aparecium v2 depends on:

- Python ≥ 3.9
- PyTorch ≥ 2.0
- Transformers ≥ 4.40
- sentence-transformers ≥ 2.2
- huggingface-hub ≥ 0.25
- NumPy ≥ 1.23
- tqdm

GPU (CUDA) is auto‑detected when available; CPU works but is slower for training and beam‑search decoding.

---

## Project Structure (v2 subset)

```text
aparecium/
├── aparecium/              # v2 pooled-only Python package
│   ├── api.py              # High-level Aparecium wrapper
│   ├── __init__.py         # Package export
│   ├── __main__.py         # CLI entrypoint (python -m aparecium)
│   ├── config.py           # Config utilities
│   ├── data/               # Dataset + plan utilities
│   ├── infer/              # Decoding + FastAPI service
│   ├── models/             # EmbAdapter, Decoder, Sketcher, SurrogateR, Constraints
│   ├── scripts/            # Data prep, embedding, inspection
│   ├── train/              # S1/S2/r training scripts
│   └── utils/              # Common helpers, tokenization utilities
├── checkpoints/            # Local training outputs (S1/S2/r)
└── data/                   # Local data shards (train/val/test)
```

---

## License

This project is licensed under the MIT License – see the `LICENSE` file for details.

---

## Citation

If you use Aparecium v2 in research or production, please cite the project and, when relevant, also reference the v1 model card:

```bibtex
@software{apareciumv2_2025,
  author    = {SentiChain},
  title     = {Aparecium v2: Pooled MPNet Embedding Reversal for Crypto Social-Media Posts},
  year      = {2025},
  publisher = {Hugging Face},
  url       = {https://huggingface.co/SentiChain/aparecium-v2-pooled-reverser}
}
```

For the original token‑level seq2seq reverser, see:
[`SentiChain/aparecium-seq2seq-reverser`](https://huggingface.co/SentiChain/aparecium-seq2seq-reverser).
