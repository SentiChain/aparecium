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

### Vector to Text Reconstruction

```python
from aparecium import Seq2SeqReverser

# Load the pre-trained model from Hugging Face Hub
reverser = Seq2SeqReverser.from_pretrained("SentiChain/aparecium-seq2seq-reverser")

# Reconstruct text from embedding vectors
recovered_text = reverser.generate_text(embedding_vectors)
print(recovered_text)
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

## Examples

The `examples/` directory contains several comprehensive examples:

- `train_pipeline.py`: Complete training pipeline for the text reconstruction model
- `train_reverser.py`: Script for training the embedding reverser model
- `generate_sentences.py`: Example of generating text from embedding vectors
- `config.py`: Configuration management for training and inference

For detailed usage examples, please refer to the individual example files in the `examples/` directory.

## Project Structure

```
aparecium/
├── aparecium/         # Main package directory
├── examples/          # Example scripts
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
   pip install -e ".[dev]"
   ```

4. Run tests:
   ```bash
   pytest
   ```

## Requirements

- Python ≥ 3.7
- PyTorch 2.5.1
- Transformers 4.47.1
- SentiChain 0.2.0
- NumPy 1.26.4

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