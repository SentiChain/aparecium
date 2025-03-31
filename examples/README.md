# Aparecium Examples

This directory contains example scripts demonstrating how to use the Aparecium package.

## Important Disclaimers

### API Key Security
- The examples require an OpenAI API key to be configured in `config.py`
- Users are solely responsible for protecting their API keys and managing their security
- Never commit API keys to version control or share them publicly
- The Aparecium team is not responsible for any API key leakage or misuse

### Usage Costs
- These examples make API calls to OpenAI's services, which incur costs based on:
  - The model selected (e.g., GPT-4 is significantly more expensive than GPT-3.5)
  - The number of tokens used per request
  - The frequency of API calls
- Users should carefully review their OpenAI usage limits and pricing before running these examples
- Monitor your API usage through OpenAI's dashboard to avoid unexpected charges
- The Aparecium team is not responsible for any costs incurred through API usage

### General Disclaimer
- These examples are provided as-is without any warranties
- Users should exercise caution and verify all configurations before execution
- The Aparecium team assumes no responsibility for any issues arising from the use of these examples

## Training Pipeline

The `train_pipeline.py` script provides an integrated pipeline for:
1. Generating crypto-related sentences using OpenAI's API
2. Storing sentences in ApareciumDB
3. Training a Seq2SeqReverser model on the generated data

### Configuration

The pipeline uses `config.py` for all configuration settings. You can modify the following parameters:

```python
# OpenAI API settings
config.openai.api_key = "your-openai-api-key"
config.openai.model = "gpt-4o-mini"
config.openai.max_tokens = 500
config.openai.temperature = 0.7

# Database settings
config.database.path = "data/generated_sentences.db"
config.database.block_size = 20

# Sentence generation settings
config.generation.num_sentences = 1000
config.generation.batch_size = 10
config.generation.max_retries = 3
config.generation.retry_delay = 1

# Model training settings
config.training.device = "cuda"  # or "cpu"
config.training.epochs = 5
config.training.batch_size = 8
config.training.block_start = 1
config.training.block_end = 100
config.training.block_size = 50
config.training.model_save_dir = "models/seq2seqreverser"
config.training.model_name = "example"
config.training.vectorizer_model = "sentence-transformers/all-mpnet-base-v2"
config.training.d_model = 768
config.training.num_decoder_layers = 2
config.training.nhead = 8
config.training.dim_feedforward = 2048
config.training.learning_rate = 1e-4
```

### Usage

1. First, edit `config.py` to set your OpenAI API key and adjust any other parameters as needed.

2. Run the pipeline:
```bash
python train_pipeline.py
```

The pipeline will:
- Generate sentences using OpenAI's API
- Store them in the configured database
- Train the Seq2SeqReverser model
- Save model checkpoints periodically
- Test the model on sample sentences

### Output

The pipeline will create:
- A database file at the configured path (default: `data/generated_sentences.db`)
- Model checkpoints in the configured directory (default: `models/seq2seqreverser/example`)
- Log messages showing progress and any potential issues

### Other Examples

- `generate_sentences.py`: Standalone script for generating sentences using OpenAI's API
- `train_reverser.py`: Standalone script for training the Seq2SeqReverser model