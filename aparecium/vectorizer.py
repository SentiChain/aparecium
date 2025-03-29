"""
Text Vectorization Module

This module provides functionality for converting text into vector representations
using pre-trained transformer models. These vector representations can be used for
semantic search, text similarity, or as input features for downstream NLP tasks.
"""

from typing import List
import torch  # type: ignore
from transformers import AutoTokenizer, AutoModel  # type: ignore


class Vectorizer:
    """
    A class for converting text into dense vector representations.

    This class uses pre-trained transformer models from Hugging Face to convert
    text into contextualized embeddings. It returns the full sequence of token
    embeddings rather than a single sentence embedding.

    Attributes:
        tokenizer: The tokenizer used to preprocess text for the model
        model: The transformer model used for generating embeddings
        device: The device (CPU/GPU) where computations are performed
    """

    def __init__(
        self, model_name="sentence-transformers/all-mpnet-base-v2", device=None
    ):
        """
        Initialize the Vectorizer with a pre-trained model.

        Args:
            model_name (str): The name of the pre-trained model to use.
                Defaults to "sentence-transformers/all-mpnet-base-v2".
            device (str, optional): Device to run the model on ('cpu' or 'cuda').
                If None, will use CUDA if available, otherwise CPU.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def encode(self, text: str, max_length: int = 256) -> List[List[float]]:
        """
        Tokenize and encode text into a matrix of embeddings.

        This method converts the input text into token embeddings using the
        pre-trained transformer model. Each token is represented by a vector
        of floating point values.

        Args:
            text (str): The input text to encode
            max_length (int, optional): Maximum sequence length for tokenization.
                Longer sequences will be truncated. Defaults to 256.

        Returns:
            List[List[float]]: A matrix of token embeddings with shape
                (sequence_length, embedding_dimension), where sequence_length
                is the number of tokens in the input (capped at max_length)
                and embedding_dimension is determined by the model.
        """
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_length
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            last_hidden_state = outputs.last_hidden_state

        matrix = last_hidden_state[0].cpu().tolist()
        return matrix
