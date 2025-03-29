"""
Tests for the Vectorizer module.
"""

import unittest
from unittest.mock import patch, MagicMock
import torch  # type: ignore

from aparecium.vectorizer import Vectorizer  # type: ignore


class TestVectorizer(unittest.TestCase):
    """
    Tests for the Vectorizer class in aparecium/vectorizer.py
    """

    @patch("aparecium.vectorizer.AutoTokenizer")
    @patch("aparecium.vectorizer.AutoModel")
    def test_init(self, mock_auto_model, mock_auto_tokenizer):
        """
        Test Vectorizer initialization
        """
        # Setup mocks
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        mock_auto_model.from_pretrained.return_value = mock_model
        mock_model.to.return_value = mock_model

        # Test with default parameters
        vectorizer = Vectorizer()
        mock_auto_tokenizer.from_pretrained.assert_called_once_with(
            "sentence-transformers/all-mpnet-base-v2"
        )
        mock_auto_model.from_pretrained.assert_called_once_with(
            "sentence-transformers/all-mpnet-base-v2"
        )
        mock_model.to.assert_called_once()
        mock_model.eval.assert_called_once()

        # Test device assignment
        self.assertIsInstance(vectorizer.device, torch.device)

    @patch("aparecium.vectorizer.AutoTokenizer")
    @patch("aparecium.vectorizer.AutoModel")
    def test_encode(self, mock_auto_model, mock_auto_tokenizer):
        """
        Test the encode method
        """
        # Setup mocks
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        mock_auto_model.from_pretrained.return_value = mock_model
        mock_model.to.return_value = mock_model

        # Setup return values
        test_text = "This is a test sentence"
        mock_inputs = {
            "input_ids": torch.tensor([[101, 2023, 2003, 1037, 3231, 6251, 102]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 1]]),
        }
        mock_tokenizer.return_value = mock_inputs

        # Mock model output
        mock_output = MagicMock()
        embedding_dim = 768
        seq_len = 7
        mock_output.last_hidden_state = torch.rand(1, seq_len, embedding_dim)
        mock_model.return_value = mock_output

        # Initialize vectorizer and call encode
        vectorizer = Vectorizer()
        result = vectorizer.encode(test_text)

        # Verify tokenizer called with correct args
        mock_tokenizer.assert_called_once_with(
            test_text, return_tensors="pt", truncation=True, max_length=256
        )

        # Check result structure
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), seq_len)  # Should match sequence length
        self.assertIsInstance(result[0], list)  # Each row should be a list
        self.assertEqual(
            len(result[0]), embedding_dim
        )  # Each vector should have correct dimension

    @patch("aparecium.vectorizer.AutoTokenizer")
    @patch("aparecium.vectorizer.AutoModel")
    def test_encode_with_custom_length(self, mock_auto_model, mock_auto_tokenizer):
        """
        Test encode method with custom max_length
        """
        # Setup mocks
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        mock_auto_model.from_pretrained.return_value = mock_model
        mock_model.to.return_value = mock_model

        # Setup return values
        test_text = "This is a test sentence"
        custom_max_length = 128
        mock_inputs = {
            "input_ids": torch.tensor([[101, 2023, 2003, 1037, 3231, 6251, 102]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 1]]),
        }
        mock_tokenizer.return_value = mock_inputs

        # Mock model output
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.rand(1, 7, 768)
        mock_model.return_value = mock_output

        # Initialize vectorizer and call encode with custom max_length
        vectorizer = Vectorizer()
        vectorizer.encode(test_text, max_length=custom_max_length)

        # Verify tokenizer called with custom max_length
        mock_tokenizer.assert_called_once_with(
            test_text,
            return_tensors="pt",
            truncation=True,
            max_length=custom_max_length,
        )


if __name__ == "__main__":
    unittest.main()
