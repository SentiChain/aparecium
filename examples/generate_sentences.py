"""
Sentence Generation Module for Aparecium.

This module provides functions for generating short, engaging sentences using OpenAI's API
and storing them in ApareciumDB for training the Seq2SeqReverser model. The generated sentences are
designed to be concise and natural, suitable for training purposes.

IMPORTANT DISCLAIMERS:
1. This module requires an OpenAI API key and will incur costs based on API usage.
2. Users are solely responsible for protecting their API keys and managing their security.
3. Never commit API keys to version control or share them publicly.
4. Monitor your API usage through OpenAI's dashboard to avoid unexpected charges.
5. The Aparecium team is not responsible for any costs or security issues arising from API usage.

Note: This module uses the gpt-4o-mini model by default. Please ensure you have access to this model
and understand its associated costs before proceeding.
"""

import time
from typing import List, Optional
import openai  # type: ignore

from aparecium.db_utils import ApareciumDB  # type: ignore
from aparecium.logger import logger  # type: ignore


def generate_sentences(
    api_key: str,
    num_sentences: int,
    batch_size: int,
    max_retries: int,
    retry_delay: int,
    openai_model: str,
    max_tokens: int,
    temperature: float,
) -> List[str]:
    """
    Generates cryptocurrency-related sentences using OpenAI's API.

    This function interfaces with OpenAI's API to generate a specified number of sentences
    related to cryptocurrency markets, news, and trends. It implements batch processing and
    retry logic for robust operation.

    Args:
        api_key (str): OpenAI API key for authentication.
        num_sentences (int): Total number of sentences to generate.
        batch_size (int): Number of sentences to generate in each API call.
        max_retries (int): Maximum number of retry attempts for failed API calls.
        retry_delay (int): Delay in seconds between retry attempts.
        openai_model (str): Identifier of the OpenAI model to use.
        max_tokens (int): Maximum number of tokens per API request.
        temperature (float): Sampling temperature for text generation (0.0 to 1.0).

    Returns:
        List[str]: A list of generated sentences, each between 5 and 40 words.

    Raises:
        Exception: If sentence generation fails after all retry attempts.
    """
    openai.api_key = api_key

    sentences = []
    num_batches = (num_sentences + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        current_batch_size = min(batch_size, num_sentences - len(sentences))
        batch_sentences = []
        batch_attempts = 0

        while (
            len(batch_sentences) < current_batch_size and batch_attempts < max_retries
        ):
            try:
                completion = openai.chat.completions.create(
                    model=openai_model,
                    store=True,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a knowledgeable assistant with expertise in cryptocurrency developments. "
                                "Provide concise, engaging updates on recent cryptocurrency news, incidents, market trends, and price movements. "
                                "Each update must be 5-40 words long, written naturally, and should not mention social media platforms or their features. "
                                "Vary the tone and format (e.g., headlines, questions, short calls to action, mild humor) to keep the updates diverse. "
                                "Focus on accuracy and clarity. "
                                "Include crypto symbols or tickers (e.g., $BTC, $ETH) in some of the updates."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Please generate exactly {current_batch_size} unique, informative updates. "
                                "Each update should be on its own line, remain between 5 and 40 words, "
                                "and cover different topics or angles in the crypto sphere. "
                                "Ensure that some updates include relevant crypto symbols."
                            ),
                        },
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                response = completion.choices[0].message.content.strip()  # type: ignore

                # Extract and clean sentences
                new_sentences = [
                    line.strip()
                    for line in response.split("\n")
                    if line.strip()
                    and not line.strip().startswith(("1.", "2.", "3.", "4.", "5."))
                ]

                # Only add sentences if not exceeded the batch size
                remaining_slots = current_batch_size - len(batch_sentences)
                batch_sentences.extend(new_sentences[:remaining_slots])

                if len(batch_sentences) < current_batch_size:
                    batch_attempts += 1
                    if batch_attempts < max_retries:
                        logger.warning(
                            f"Batch {batch_idx + 1} only got {len(batch_sentences)}/{current_batch_size} sentences. "
                            f"Retrying... (Attempt {batch_attempts + 1}/{max_retries})"
                        )
                        time.sleep(retry_delay)
                    else:
                        logger.error(
                            f"Failed to generate {current_batch_size} sentences after {max_retries} attempts. "
                            f"Got {len(batch_sentences)} sentences instead."
                        )
                        raise Exception(
                            f"Failed to generate required number of sentences for batch {batch_idx + 1}"
                        )

            except Exception as e:
                batch_attempts += 1
                if batch_attempts == max_retries:
                    logger.error(
                        f"Failed to generate sentences after {max_retries} attempts: {str(e)}"
                    )
                    raise
                logger.warning(
                    f"Attempt {batch_attempts} failed, retrying in {retry_delay} seconds..."
                )
                time.sleep(retry_delay)

        sentences.extend(batch_sentences)
        logger.info(
            f"Generated batch {batch_idx + 1}/{num_batches} with {len(batch_sentences)} sentences"
        )

    return sentences


def store_sentences(
    sentences: List[str],
    block_start: int,
    block_end: int,
    db: ApareciumDB,
    vectorizer: Optional[object] = None,
) -> None:
    """
    Stores generated sentences in the ApareciumDB database.

    This function stores a batch of sentences in the database, optionally generating
    and storing their vector embeddings if a vectorizer is provided.

    Args:
        sentences (List[str]): List of sentences to store in the database.
        block_start (int): Starting block number for storage.
        block_end (int): Ending block number for storage.
        db (ApareciumDB): Database instance for storing sentences and embeddings.
        vectorizer (Optional[object]): Vectorizer instance for generating embeddings.
            If None, only sentences are stored without embeddings.
    """
    if vectorizer:
        matrices = [vectorizer.vectorize(sentence) for sentence in sentences]
        db.store_batch(block_start, block_end, sentences, matrices)
    else:
        db.store_batch(block_start, block_end, sentences, [None] * len(sentences))

    logger.info(
        f"Stored {len(sentences)} sentences in blocks {block_start}-{block_end}"
    )
