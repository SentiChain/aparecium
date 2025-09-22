# Aparecium

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Aparecium is a Python package for revealing text from embedding vectors, particularly designed to work with SentiChain embeddings. Named after the Harry Potter spell that reveals hidden writing, Aparecium provides state-of-the-art tools for converting between text and vector representations, as well as reversing the embedding process to recover original text.

## Features

- **Text Vectorization**: Convert text into dense vector representations using pre-trained transformer models
- **Embedding Reversal**: Reconstruct original text from embedding vectors using a Transformer-based sequence-to-sequence architecture
- **Seamless Integration**: Works with SentiChain embeddings to reveal hidden content
- **Modern Architecture**: Built on PyTorch and Transformers for optimal performance
- **Extensible Design**: Easy to integrate with custom models and architectures

## Limitations & Caveats

- Complete reconstruction of original text from embeddings is not always guaranteed and depends heavily on the fidelity and nature of the embeddings.
- For best results, ensure that the model and embeddings are aligned (e.g., same tokenization and dimension).
- While Aparecium is designed for SentiChain embeddings, it can be adapted to other embedding pipelines if they provide a compatible dimensionality.

## Model Architecture

Aparecium employs a Transformer-based sequence-to-sequence architecture for text reconstruction. The model consists of:

- **Input Layer**: Processes embedding vectors of shape (sequence_length, d_model)
- **Embedding Layer**: Combines token and positional embeddings
- **Transformer Decoder Stack**: Multiple decoder layers with multi-head attention
- **Output Layer**: Projects decoder outputs to vocabulary space

```mermaid
graph TB
    subgraph InputLayer["Input Layer"]
        Input["Input Embeddings\n(seq_len × d_model)"]
    end

    subgraph EmbeddingLayer["Embedding Layer"]
        TokenEmb["Token Embedding\n(vocab_size → d_model)"]
        PosEmb["Positional Embedding\n(seq_len → d_model)"]
        Combined["Combined Embeddings\n(d_model)"]
        TokenEmb --> Combined
        PosEmb --> Combined
    end

    subgraph DecoderStack["Transformer Decoder Stack"]
        Dec1["Decoder Layer 1\n(nhead=8, dim_ff=2048)"]
        Dec2["Decoder Layer 2\n(nhead=8, dim_ff=2048)"]
        Dec1 --> Dec2
    end

    subgraph OutputLayer["Output Layer"]
        FC["Linear Projection\n(d_model → vocab_size)"]
        Output["Output Logits\n(seq_len × vocab_size)"]
        FC --> Output
    end

    Input --> Dec1
    Combined --> Dec1
    Dec2 --> FC
```

## Installation

### From PyPI

```bash
pip install aparecium
```

### From Source

```bash
git clone https://github.com/SentiChain/aparecium.git
cd aparecium
pip install -e .
```

## Quick Start

### Text to Vector Conversion

```python
from aparecium import Vectorizer

# Initialize the vectorizer with a pre-trained model
vectorizer = Vectorizer(model_name="sentence-transformers/all-mpnet-base-v2")

# Convert text to vector representation
text = "This is sample text to be vectorized."
embedding_vectors = vectorizer.encode(text)

# embedding_vectors shape: (sequence_length, embedding_dimension)
```

### Vector to Text Reconstruction (from Hugging Face Hub)

```python
from aparecium import Seq2SeqReverser

# Load the pre-trained model from Hugging Face Hub
reverser = Seq2SeqReverser.from_pretrained("SentiChain/aparecium-seq2seq-reverser")

# Reconstruct text from embedding vectors (expects token-level MPNet matrix)
text_or_text_info = reverser.generate_text(
    embedding_vectors,  # shape: (seq_len, 768)
    max_length=128,
    num_beams=8,
    deterministic=True,
    length_penalty_alpha=0.6,
    lambda_sim=0.6,
    rescore_every_k=4,
    rescore_top_m=8,
    beta=10.0,
    enable_constraints=True,
)
print(text_or_text_info)
```

Note: The pre-trained model is specifically trained on crypto market-related sentences. For best results, use it with similar content.

Alternatively, you can load the model from a local directory:

```python
from aparecium import Seq2SeqReverser

# Initialize the reverser
reverser = Seq2SeqReverser()

# Load the pre-trained model from a local directory
reverser.load_model("path/to/model/directory")

# Reconstruct text from embedding vectors
recovered_text = reverser.generate_text(embedding_vectors)
print(recovered_text)
```

## Pipeline

The `pipeline/` directory contains end-to-end scripts and documentation for data preparation, training, and evaluation. See `pipeline/README.md` for details, including CLI flags and examples.

Note: Pipeline scripts are repository-only and are not included in the PyPI package. If you installed via `pip install aparecium`, clone the repo to use the pipeline scripts.

- `pipeline/prepare_db.py`: Cache token-level MPNet embeddings and texts into an SQLite database
- `pipeline/train.py`: Train or resume the `Seq2SeqReverser` directly from cached embeddings
- `pipeline/evaluate.py`: Evaluate model checkpoints
- `pipeline/evaluate_prompts.py`: Evaluate prompts/decoding settings against cached data

## Project Structure

```
aparecium/
├── aparecium/         # Main package directory
├── pipeline/          # Training/evaluation pipeline & docs
├── tests/             # Unittest suite
├── data/              # Data directory
├── models/            # Model checkpoints and configurations
└── logs/              # Training and evaluation logs
```

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/SentiChain/aparecium.git
   cd aparecium
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install development dependencies:
   ```bash
   pip install -e .
   pip install pytest
   ```

4. Run tests:
   ```bash
   pytest
   ```

## Model Input Contract & Defaults

- Input to `Seq2SeqReverser.generate_text(...)` must be a token-level MPNet matrix with shape `(src_len, d_model)` (not a pooled vector). Use `Vectorizer.encode(text, max_length=384)` to produce it.
- Suggested defaults that “just work” for tweets:
  - beams: `num_beams=5`
  - length penalty: `length_penalty_alpha=0.6`
  - embedding fusion: `lambda_sim=0.3`
  - rescoring cadence/top-M: `rescore_every_k=4`, `rescore_top_m=8`
  - cosine scale: `beta=10.0`
  - target length: `max_length≈128`
  - determinism: `deterministic=True`
  - constraints: `enable_constraints=True`

### Confidence semantics
- If `return_confidence=True`, generation returns `(text, info)` where `info` includes:
  - `cosine`: final cosine similarity to the MPNet target vector (higher is better)
  - `score_norm`: length-penalized LM score
  - `fused_score`: fused value of LM score and cosine, used for ranking
  - `used_refinement_steps`: 0 by default (reserved for future optional refinement)

## Requirements

- Python ≥ 3.9
- PyTorch 2.5.1
- Transformers 4.47.1
- SentiChain ≥ 0.2.2
- NumPy 1.26.4
- huggingface-hub ≥ 0.24.0

Optional:
- openai == 1.58.1 (only needed for certain evaluation utilities)

Note: GPU (CUDA) is auto-detected when available; CPU works but will be slower for training and generation.

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use Aparecium in your research, please cite:

```bibtex
@software{aparecium2025,
  author = {Chen, Edward},
  title = {Aparecium: Text Reconstruction from Embedding Vectors},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/SentiChain/aparecium}
}
```

## Links

- [GitHub Repository](https://github.com/SentiChain/aparecium)
- [Issue Tracker](https://github.com/SentiChain/aparecium/issues)