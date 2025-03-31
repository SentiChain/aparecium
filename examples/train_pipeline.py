"""
Aparecium Training Pipeline.

This script provides an integrated pipeline for:
1. Generating sentences using OpenAI's API (using gpt-4o-mini model)
2. Storing sentences in ApareciumDB
3. Training a Seq2SeqReverser model on the generated data

IMPORTANT DISCLAIMERS:
1. This script requires an OpenAI API key and will incur costs based on API usage.
2. Users are solely responsible for protecting their API keys and managing their security.
3. Never commit API keys to version control or share them publicly.
4. Monitor your API usage through OpenAI's dashboard to avoid unexpected charges.
5. The Aparecium team is not responsible for any costs or security issues arising from API usage.
"""

import os
import torch  # type: ignore

from aparecium import Vectorizer, Seq2SeqReverser  # type: ignore
from aparecium.db_utils import ApareciumDB  # type: ignore
from aparecium.logger import logger  # type: ignore
from examples.generate_sentences import generate_sentences  # type: ignore
from examples.train_reverser import prepare_train_data, test_sample  # type: ignore
from examples.config import Config, config_default  # type: ignore


def print_disclaimer() -> None:
    """
    Displays important disclaimers about API usage and costs.

    This function prints a comprehensive set of disclaimers and warnings about
    API usage, costs, and security considerations. It requires user acknowledgment
    before proceeding with the pipeline execution.
    """
    print("\n" + "=" * 80)
    print("IMPORTANT DISCLAIMERS")
    print("=" * 80)
    print(
        "1. This script requires an OpenAI API key and will incur costs based on API usage."
    )
    print(
        "2. Users are solely responsible for protecting their API keys and managing their security."
    )
    print("3. Never commit API keys to version control or share them publicly.")
    print(
        "4. Monitor your API usage through OpenAI's dashboard to avoid unexpected charges."
    )
    print(
        "5. The Aparecium project is not responsible for any costs or security issues arising from API usage."
    )
    print("=" * 80 + "\n")

    print("WARNING: By proceeding, you acknowledge that:")
    print("1. You understand the potential costs and risks involved")
    print("2. You are solely responsible for any API usage and associated costs")
    print("3. You have reviewed and accept all the disclaimers above")
    print("\nIf you do not wish to proceed, please close this process now.")
    print("\nPress any key to continue at your own risk...")
    input()  # Wait for any key press


def setup_directories(config: Config) -> None:
    """
    Creates necessary directories for data and model storage.

    Args:
        config (Config): Configuration object containing directory paths.
    """
    os.makedirs(os.path.dirname(config.database.path), exist_ok=True)
    os.makedirs(config.training.model_save_dir, exist_ok=True)


def reset_database(config: Config) -> None:
    """
    Resets the database by deleting the existing file if present.

    Args:
        config (Config): Configuration object containing database path.

    Raises:
        Exception: If database deletion fails.
    """
    db_path = config.database.path
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
            logger.info(f"Successfully deleted existing database at {db_path}")
        except Exception as e:
            logger.error(f"Failed to delete database: {str(e)}")
            raise
    else:
        logger.info("No existing database found to reset")


def get_openai_api_key() -> str:
    """
    Retrieves the OpenAI API key from environment variables or configuration.

    This function attempts to retrieve the API key from environment variables first,
    then falls back to the configuration file. If no key is found, it raises an error
    with clear instructions for setting up the key.

    Returns:
        str: The OpenAI API key.

    Raises:
        ValueError: If no API key is found in environment or configuration.
    """
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        api_key = config_default.openai.api_key

    if not api_key:
        print("\nERROR: OpenAI API key not found!")
        print("\nTo set up your API key, you can either:")
        print("1. Set the OPENAI_API_KEY environment variable:")
        print("   Windows: set OPENAI_API_KEY=your-api-key-here")
        print("   Linux/Mac: export OPENAI_API_KEY=your-api-key-here")
        print("\n2. Or edit config.py and set your API key in the OpenAIConfig class:")
        print("   config.openai.api_key = 'your-api-key-here'")
        print("\nPlease ensure you have a valid OpenAI API key before proceeding.")
        raise ValueError(
            "OpenAI API key not found. Please set it in environment variables or config.py"
        )

    return api_key


def generate_and_store_sentences(config: Config) -> None:
    """
    Generates sentences using OpenAI API and stores them in the database.

    This function orchestrates the sentence generation process and database storage,
    handling batching and error cases appropriately.

    Args:
        config (Config): Configuration object containing generation and database settings.
    """
    logger.info("Starting sentence generation...")

    api_key = get_openai_api_key()

    sentences = generate_sentences(
        api_key=api_key,
        num_sentences=config.generation.num_sentences,
        batch_size=config.generation.batch_size,
        max_retries=config.generation.max_retries,
        retry_delay=config.generation.retry_delay,
        openai_model=config.openai.model,
        max_tokens=config.openai.max_tokens,
        temperature=config.openai.temperature,
    )

    # Store sentences in database
    db = ApareciumDB(config.database.path)
    block_size = config.database.block_size

    for i in range(0, len(sentences), block_size):
        block_start = i // block_size
        block_end = block_start + 1
        block_sentences = sentences[i : i + block_size]

        db.store_batch(
            block_start, block_end, block_sentences, [None] * len(block_sentences)
        )
        logger.info(
            f"Stored block {block_start}-{block_end} with {len(block_sentences)} sentences"
        )

    db.close()
    logger.info("Sentence generation and storage completed")


def train_model(config: Config) -> None:
    """
    Trains the Seq2SeqReverser model on the generated sentences.

    This function handles the complete training process, including:
    1. Device setup (CPU/CUDA)
    2. Model initialization
    3. Training loop with batch processing
    4. Periodic model testing and checkpointing

    Args:
        config (Config): Configuration object containing training parameters.
    """
    logger.info("Starting model training...")

    # Setup device
    device = config.training.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"

    # Initialize database and models
    db = ApareciumDB(config.database.path)
    vectorizer = Vectorizer(
        model_name=config.training.vectorizer_model,
        device=device,
    )

    reverser = Seq2SeqReverser(
        model_name=config.training.vectorizer_model,
        d_model=config.training.d_model,
        num_decoder_layers=config.training.num_decoder_layers,
        nhead=config.training.nhead,
        dim_feedforward=config.training.dim_feedforward,
        lr=config.training.learning_rate,
        device=device,
    )

    # Training parameters
    epochs = config.training.epochs
    batch_size = config.training.batch_size
    block_start = config.training.block_start
    block_end = config.training.block_end
    block_size = config.training.block_size
    model_path = os.path.join(
        config.training.model_save_dir, config.training.model_name
    )

    # Training loop
    for epoch in range(epochs):
        total_loss = 0.0
        total_samples = 0

        for start in range(block_start, block_end, block_size):
            current_end = min(start + block_size - 1, block_end)
            logger.info(f"Processing blocks {start}-{current_end}")

            # Prepare training data
            train_sentences, train_matrices = prepare_train_data(
                start, current_end, vectorizer, db
            )

            if not train_sentences:
                logger.warning(
                    f"No samples found for blocks {start}-{current_end}, skipping"
                )
                continue

            # Train in batches
            for i in range(0, len(train_sentences), batch_size):
                batch_sents = train_sentences[i : i + batch_size]
                batch_mats = train_matrices[i : i + batch_size]

                loss = reverser.train_step_batch(
                    source_rep_batch=batch_mats,
                    target_text_batch=batch_sents,
                    max_source_length=256,
                    max_target_length=256,
                )

                effective_batch_size = len(batch_sents)
                total_loss += loss * effective_batch_size
                total_samples += effective_batch_size

                if i % (batch_size * 10) == 0 or i + batch_size >= len(train_sentences):
                    logger.info(
                        f"Epoch {epoch+1}/{epochs}, Block {start}-{current_end}, "
                        f"Batch {i//batch_size + 1}/{(len(train_sentences) + batch_size - 1)//batch_size}, "
                        f"Loss: {loss:.4f}"
                    )

            # Test model periodically
            test_sample(vectorizer, reverser)

            # Save checkpoint
            reverser.save_model(model_path)
            logger.info(f"Model saved to {model_path}")

        # Log epoch statistics
        avg_loss = (total_loss / total_samples) if total_samples > 0 else 0.0
        logger.info(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}")

    db.close()
    logger.info("Model training completed")


def validate_config(config: Config) -> None:
    """
    Validates configuration settings for consistency and potential issues.

    This function performs comprehensive validation of the configuration settings,
    checking for:
    1. Database block size consistency with training block size
    2. Block range coverage for sentence generation
    3. Potential data loss or overflow issues
    4. Configuration contradictions

    Args:
        config (Config): Configuration object to validate.

    Raises:
        ValueError: If configuration validation fails.
    """
    warnings = []
    errors = []

    # Check block size consistency
    if config.database.block_size != config.training.block_size:
        warnings.append(
            f"Database block size ({config.database.block_size}) differs from "
            f"training block size ({config.training.block_size}). This may cause "
            "inefficient data loading or storage."
        )

    # Calculate total blocks needed for sentence generation
    total_blocks_needed = (
        config.generation.num_sentences + config.database.block_size - 1
    ) // config.database.block_size
    total_blocks_available = config.training.block_end - config.training.block_start + 1

    # Check if we have enough blocks to store all sentences
    if total_blocks_needed > total_blocks_available:
        errors.append(
            f"Configuration error: Not enough blocks to store all sentences.\n"
            f"Need {total_blocks_needed} blocks for {config.generation.num_sentences} sentences "
            f"(with block_size={config.database.block_size}), but only have "
            f"{total_blocks_available} blocks in range {config.training.block_start}-{config.training.block_end}"
        )

    # Check if block range makes sense
    if config.training.block_start > config.training.block_end:
        errors.append(
            f"Configuration error: block_start ({config.training.block_start}) "
            f"is greater than block_end ({config.training.block_end})"
        )

    # Check if block size is reasonable
    if config.training.block_size <= 0:
        errors.append(
            f"Configuration error: block_size ({config.training.block_size}) "
            "must be greater than 0"
        )

    # Check if number of sentences is reasonable
    if config.generation.num_sentences <= 0:
        errors.append(
            f"Configuration error: num_sentences ({config.generation.num_sentences}) "
            "must be greater than 0"
        )

    # Check if batch sizes are reasonable
    if config.generation.batch_size <= 0:
        errors.append(
            f"Configuration error: generation.batch_size ({config.generation.batch_size}) "
            "must be greater than 0"
        )
    if config.training.batch_size <= 0:
        errors.append(
            f"Configuration error: training.batch_size ({config.training.batch_size}) "
            "must be greater than 0"
        )

    # Check if we're skipping generation but resetting database
    if config.generation.skip_generation and config.database.reset_database:
        warnings.append(
            "Warning: You are configured to skip sentence generation but also reset the database. "
            "This will result in an empty database and no training data."
        )

    # Print all warnings
    if warnings:
        print("\n" + "=" * 80)
        print("CONFIGURATION WARNINGS")
        print("=" * 80)
        for warning in warnings:
            print(f"⚠️  {warning}")
        print("=" * 80 + "\n")

    # Print all errors and raise exception if any
    if errors:
        print("\n" + "=" * 80)
        print("CONFIGURATION ERRORS")
        print("=" * 80)
        for error in errors:
            print(f"❌ {error}")
        print("=" * 80 + "\n")
        raise ValueError(
            "Configuration validation failed. Please fix the errors above."
        )


def main() -> None:
    """
    Main entry point for the training pipeline.

    This function orchestrates the complete training pipeline:
    1. Displays disclaimers and gets user confirmation
    2. Validates configuration settings
    3. Sets up necessary directories
    4. Resets database if configured
    5. Generates and stores sentences
    6. Trains the model

    Raises:
        Exception: If any step of the pipeline fails.
    """
    print_disclaimer()

    validate_config(config_default)

    setup_directories(config_default)

    # Reset database if configured
    if config_default.database.reset_database:
        reset_database(config_default)

    try:
        # Generate and store sentences if not skipped
        if not config_default.generation.skip_generation:
            generate_and_store_sentences(config_default)
        else:
            logger.info("Skipping sentence generation as configured")

        # Train the model
        train_model(config_default)

        logger.info("Pipeline completed successfully")

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
