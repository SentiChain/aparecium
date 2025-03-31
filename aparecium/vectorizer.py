"""
Text Vectorization Module

This module provides functionality for converting text into vector representations
using pre-trained transformer models from Hugging Face. These vector representations
can be used for semantic search, text similarity, or as input features for downstream
NLP tasks.

Example:
    >>> from vectorizer import Vectorizer
    >>> vectorizer = Vectorizer()
    >>> embeddings = vectorizer.encode("Hello world")
    >>> print(embeddings)  # A list of token embedding vectors
"""

from typing import List
import torch  # type: ignore
from transformers import AutoTokenizer, AutoModel  # type: ignore

from .logger import logger  # type: ignore
from .exceptions import (  # type: ignore
    VectorizationError,
    ConfigurationError,
    DataProcessingError,
)


class Vectorizer:
    """
    A class for converting text into dense, contextualized vector representations.

    This class uses a pre-trained transformer model from Hugging Face
    to convert text into token-level embeddings. It returns the full
    sequence of token embeddings instead of a single aggregate embedding.

    Attributes:
        tokenizer (transformers.PreTrainedTokenizer):
            The tokenizer used to preprocess text for the model.
        model (transformers.PreTrainedModel):
            The underlying transformer model used for generating embeddings.
        device (torch.device):
            The device (CPU or GPU) where computations are performed.
    """

    def __init__(
        self, model_name="sentence-transformers/all-mpnet-base-v2", device=None
    ):
        """
        Initialize the Vectorizer with a pre-trained model.

        Args:
            model_name (str):
                The name or path of the pre-trained model to use
                (Hugging Face model hub or local directory).
                Defaults to "sentence-transformers/all-mpnet-base-v2".
            device (Optional[str]):
                The device on which to run the model ('cpu', 'cuda', or None).
                If None, uses 'cuda' if available, otherwise 'cpu'.

        Raises:
            ConfigurationError: If model initialization fails due to invalid configuration.
            VectorizationError: If model or tokenizer loading fails.
        """
        try:
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = torch.device(device)
            logger.info(f"Initializing Vectorizer on device: {device}")

            logger.debug(f"Loading tokenizer from {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

            logger.debug(f"Loading model from {model_name}")
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self.model.eval()

            logger.info("Vectorizer initialized successfully")
        except OSError as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            raise ConfigurationError(f"Failed to initialize model: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during model initialization: {str(e)}")
            raise VectorizationError(
                f"Unexpected error during model initialization: {str(e)}"
            )

    def encode(self, text: str, max_length: int = 256) -> List[List[float]]:
        """
        Tokenize and encode text into a matrix of token embeddings.

        This method uses the pre-trained transformer model to convert
        the input text into a sequence of token embeddings (one vector per token).

        Args:
            text (str):
                The input text string to encode.
            max_length (int, optional):
                The maximum sequence length for tokenization.
                Sequences longer than this will be truncated.
                Defaults to 256.

        Returns:
            List[List[float]]:
                A 2D list (matrix) representing the token embeddings,
                with shape (sequence_length, embedding_dimension).
                The `sequence_length` is determined by the number of tokens in
                the input (up to `max_length`), and `embedding_dimension`
                depends on the pre-trained model.

        Raises:
            DataProcessingError: If input text is invalid or max_length is out of range.
            VectorizationError: If encoding fails due to model error.
        """
        try:
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Input text must be a non-empty string")
            if not isinstance(max_length, int) or max_length <= 0:
                raise ValueError(
                    f"max_length must be a positive integer, got {max_length}"
                )

            logger.debug(f"Encoding text with max_length={max_length}")
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=max_length
            )
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask)
                last_hidden_state = outputs.last_hidden_state

            matrix = last_hidden_state[0].cpu().tolist()
            logger.debug(
                f"Successfully encoded text into {len(matrix)} token embeddings"
            )
            return matrix
        except ValueError as e:
            logger.error(f"Invalid input parameters: {str(e)}")
            raise DataProcessingError(f"Invalid input parameters: {str(e)}")
        except Exception as e:
            logger.error(f"Text encoding failed: {str(e)}")
            raise VectorizationError(f"Text encoding failed: {str(e)}")
