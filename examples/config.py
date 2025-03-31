"""
Configuration settings for the Aparecium training pipeline.

This module provides configuration classes for all aspects of the training pipeline:
1. OpenAI API settings (model, tokens, temperature)
2. Database settings (path, block size)
3. Sentence generation settings (batch size, retries)
4. Model training settings (device, epochs, model architecture)

The configuration can be loaded from environment variables or a dictionary,
with sensible defaults provided for all settings.
"""

from dataclasses import dataclass, field


@dataclass
class OpenAIConfig:
    """
    Configuration settings for OpenAI API integration.

    Attributes:
        api_key (str): OpenAI API key for authentication. Defaults to placeholder.
        model (str): OpenAI model identifier. Defaults to "gpt-4o-mini".
        max_tokens (int): Maximum tokens per API request. Defaults to 500.
        temperature (float): Sampling temperature for text generation. Defaults to 0.7.
    """

    api_key: str = (
        "your-openai-api-key"  # Set via environment variable OPENAI_API_KEY or edit this file
    )
    model: str = "gpt-4o-mini"
    max_tokens: int = 500
    temperature: float = 0.7


@dataclass
class DatabaseConfig:
    """
    Configuration settings for database operations.

    Attributes:
        path (str): Path to the SQLite database file. Defaults to "data/generated_sentences.db".
        block_size (int): Number of sentences per database block. Defaults to 20.
        reset_database (bool): Whether to delete existing database on startup. Defaults to True.
    """

    path: str = "data/generated_sentences.db"
    block_size: int = 20
    reset_database: bool = (
        True  # Set to True to delete existing database and start fresh
    )


@dataclass
class GenerationConfig:
    """
    Configuration settings for sentence generation.

    Attributes:
        num_sentences (int): Total number of sentences to generate. Defaults to 10000.
        batch_size (int): Number of sentences per API batch. Defaults to 10.
        max_retries (int): Maximum number of API retry attempts. Defaults to 3.
        retry_delay (int): Delay between retries in seconds. Defaults to 1.
        skip_generation (bool): Whether to skip sentence generation. Defaults to False.
    """

    num_sentences: int = 10000
    batch_size: int = 10
    max_retries: int = 3
    retry_delay: int = 1
    skip_generation: bool = (
        False  # Set to True to skip sentence generation and use existing database
    )


@dataclass
class TrainingConfig:
    """
    Configuration settings for model training.

    Attributes:
        device (str): Computing device for training ("cpu" or "cuda"). Defaults to "cpu".
        epochs (int): Number of training epochs. Defaults to 5.
        batch_size (int): Number of samples per training batch. Defaults to 8.
        block_start (int): Starting block number for training. Defaults to 1.
        block_end (int): Ending block number for training. Defaults to 500.
        block_size (int): Number of sentences per training block. Defaults to 20.
        model_save_dir (str): Directory for saving model checkpoints. Defaults to "models/seq2seqreverser".
        model_name (str): Name of the trained model. Defaults to "example".
        vectorizer_model (str): Name of the sentence transformer model. Defaults to "sentence-transformers/all-mpnet-base-v2".
        d_model (int): Dimension of the model's hidden states. Defaults to 768.
        num_decoder_layers (int): Number of decoder layers. Defaults to 2.
        nhead (int): Number of attention heads. Defaults to 8.
        dim_feedforward (int): Dimension of the feed-forward network. Defaults to 2048.
        learning_rate (float): Learning rate for optimization. Defaults to 1e-4.
    """

    device: str = "cpu"  # or "cuda"
    epochs: int = 5
    batch_size: int = 8
    block_start: int = 1
    block_end: int = 500
    block_size: int = 20
    model_save_dir: str = "models/seq2seqreverser"
    model_name: str = "example"
    vectorizer_model: str = "sentence-transformers/all-mpnet-base-v2"
    d_model: int = 768
    num_decoder_layers: int = 2
    nhead: int = 8
    dim_feedforward: int = 2048
    learning_rate: float = 1e-4


@dataclass
class Config:
    """
    Main configuration class combining all subsystem settings.

    This class aggregates configuration settings for all components of the training pipeline:
    OpenAI API, database operations, sentence generation, and model training.

    Attributes:
        openai (OpenAIConfig): OpenAI API configuration settings.
        database (DatabaseConfig): Database operation settings.
        generation (GenerationConfig): Sentence generation settings.
        training (TrainingConfig): Model training settings.
    """

    openai: OpenAIConfig = field(default_factory=OpenAIConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "Config":
        """
        Creates a Config instance from a dictionary of settings.

        Args:
            config_dict (dict): Dictionary containing configuration settings for each subsystem.

        Returns:
            Config: A new Config instance with settings from the dictionary.
        """
        return cls(
            openai=OpenAIConfig(**config_dict.get("openai", {})),
            database=DatabaseConfig(**config_dict.get("database", {})),
            generation=GenerationConfig(**config_dict.get("generation", {})),
            training=TrainingConfig(**config_dict.get("training", {})),
        )


# Create a default configuration instance
config_default = Config()
