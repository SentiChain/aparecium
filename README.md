# Aparecium

Aparecium (named after the Harry Potter spell to reveal hidden writing) is a Python package for revealing text from embedding vectors processed by SentiChain. It provides tools to convert between text and vector representations, as well as reverse the embedding process to recover original text.

## Installation

```bash
pip install aparecium
```

## Features

- **Text Vectorization**: Convert text into dense vector representations using pre-trained transformer models
- **Embedding Reversal**: Reconstruct original text from embedding vectors using a Transformer-based sequence-to-sequence architecture
- **Seamless Integration**: Works with SentiChain embeddings to reveal hidden content

## Usage

### Vectorizing Text

```python
from aparecium import Vectorizer

# Initialize the vectorizer with a pre-trained model
vectorizer = Vectorizer(model_name="sentence-transformers/all-mpnet-base-v2")

# Convert text to vector representation
text = "This is sample text to be vectorized."
embedding_vectors = vectorizer.encode(text)

# embedding_vectors is a 2D matrix of shape (sequence_length, embedding_dimension)
# where sequence_length is the number of tokens in the text
```

### Reversing Embeddings to Text

```python
from aparecium import Seq2SeqReverser

# Initialize the reverser
reverser = Seq2SeqReverser()

# If you have a trained model, load it
# reverser.load_model("path/to/model_directory")

# Reverse embedding vectors to text
# The source_rep should be a list of lists containing float values (embedding vectors)
recovered_text = reverser.generate_text(source_rep)
print(recovered_text)  # The text recovered from embeddings
```

### Training a Reverser Model

```python
from aparecium import Seq2SeqReverser, Vectorizer

# Initialize components
vectorizer = Vectorizer()
reverser = Seq2SeqReverser()

# Generate embedding vectors from text
text = "This is text to train with."
embeddings = vectorizer.encode(text)

# Train the reverser using the embeddings and original text
loss = reverser.train_step(embeddings, text)

# Save the trained model
reverser.save_model("path/to/save/model")
```

## Requirements

- Python ≥ 3.7
- PyTorch 2.5.1
- Transformers 4.47.1
- SentiChain 0.2.0

## License

MIT License

## Links

- [GitHub Repository](https://github.com/SentiChain/aparecium)
- [Issue Tracker](https://github.com/SentiChain/aparecium/issues)