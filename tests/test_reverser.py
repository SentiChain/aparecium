"""
Tests for the Seq2Seq Reverser module.
"""

import sys
import os

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest
from unittest.mock import patch, MagicMock
import tempfile
import torch  # type: ignore

from aparecium.reverser import (  # type: ignore
    TransformerSeq2SeqModel,
    Seq2SeqReverser,
    generate_subsequent_mask,
)


class TestSubsequentMask(unittest.TestCase):
    """
    Tests for the generate_subsequent_mask function
    """

    def test_mask_generation(self):
        """
        Test that the mask is correctly generated
        """
        size = 5
        device = torch.device("cpu")
        mask = generate_subsequent_mask(size, device)

        # Check shape and type
        self.assertEqual(mask.shape, (size, size))
        self.assertEqual(mask.dtype, torch.bool)

        # Check content (upper triangular with diagonal=1 should be True)
        # Lower triangular and diagonal should be False (not masked)
        expected = torch.tensor(
            [
                [False, True, True, True, True],
                [False, False, True, True, True],
                [False, False, False, True, True],
                [False, False, False, False, True],
                [False, False, False, False, False],
            ]
        )
        self.assertTrue(torch.all(mask == expected))


class TestTransformerSeq2SeqModel(unittest.TestCase):
    """
    Tests for the TransformerSeq2SeqModel class
    """

    def test_init(self):
        """
        Test model initialization with default parameters
        """
        vocab_size = 1000
        model = TransformerSeq2SeqModel(vocab_size)

        # Check if all components are correctly initialized
        self.assertEqual(model.token_embedding.num_embeddings, vocab_size)
        self.assertEqual(model.token_embedding.embedding_dim, 768)
        self.assertEqual(model.pos_embedding.num_embeddings, 512)
        self.assertEqual(model.pos_embedding.embedding_dim, 768)
        self.assertEqual(len(model.transformer_decoder.layers), 2)
        self.assertEqual(model.fc_out.out_features, vocab_size)

    def test_forward(self):
        """
        Test the forward pass of the model
        """
        vocab_size = 1000
        d_model = 512
        batch_size = 2
        src_seq_len = 10
        tgt_seq_len = 5

        # Initialize model with smaller dimensions for testing
        model = TransformerSeq2SeqModel(
            vocab_size=vocab_size,
            d_model=d_model,
            num_decoder_layers=1,
            nhead=4,
            dim_feedforward=1024,
        )

        # Create sample inputs
        encoder_outputs = torch.rand(src_seq_len, batch_size, d_model)
        tgt_input_ids = torch.randint(0, vocab_size, (tgt_seq_len, batch_size))
        tgt_mask = generate_subsequent_mask(tgt_seq_len, device=torch.device("cpu"))

        # Run forward pass
        logits = model(encoder_outputs, tgt_input_ids, tgt_mask)

        # Check output shape
        self.assertEqual(logits.shape, (tgt_seq_len, batch_size, vocab_size))


class TestSeq2SeqReverser(unittest.TestCase):
    """
    Tests for the Seq2SeqReverser class
    """

    @patch("aparecium.reverser.optim.AdamW")
    @patch("aparecium.reverser.AutoTokenizer")
    @patch("aparecium.reverser.TransformerSeq2SeqModel")
    def test_init(self, mock_model, mock_tokenizer, mock_adamw):
        """Test Reverser initialization"""
        # Setup mocks
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.__len__.return_value = 1000
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        # Setup mock model with parameters
        mock_model_instance = MagicMock()
        mock_params = [torch.nn.Parameter(torch.randn(10, 10))]
        mock_model_instance.parameters.return_value = mock_params
        mock_model.return_value = mock_model_instance

        # Setup mock optimizer
        mock_optimizer = MagicMock()
        mock_adamw.return_value = mock_optimizer

        # Create a reverser
        reverser = Seq2SeqReverser()

        # Check if components are initialized correctly
        mock_tokenizer.from_pretrained.assert_called_once_with(
            "sentence-transformers/all-mpnet-base-v2"
        )
        mock_model.assert_called_once()
        mock_model.call_args.kwargs["vocab_size"] = 1000
        mock_adamw.assert_called_once()

        # Test config
        self.assertEqual(
            reverser.config["model_name"], "sentence-transformers/all-mpnet-base-v2"
        )
        self.assertEqual(reverser.config["d_model"], 768)
        self.assertEqual(reverser.config["num_decoder_layers"], 2)

    @patch("aparecium.reverser.optim.AdamW")
    @patch("aparecium.reverser.AutoTokenizer")
    @patch("aparecium.reverser.TransformerSeq2SeqModel")
    def test_train_step(self, mock_model_class, mock_tokenizer, mock_adamw):
        """
        Test the train_step method
        """
        # Setup mocks
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.__len__.return_value = 1000
        mock_tokenizer_instance.pad_token_id = 0
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        # Setup mock model with parameters
        mock_decoder = MagicMock()
        mock_decoder.train = MagicMock()
        mock_params = [torch.nn.Parameter(torch.randn(10, 10))]
        mock_decoder.parameters.return_value = mock_params

        # Setup model forward pass - return a real tensor with requires_grad=True
        logits = torch.rand(5, 1, 1000, requires_grad=True)
        mock_decoder.return_value = logits

        # Return the mock decoder when TransformerSeq2SeqModel is called
        mock_model_class.return_value = mock_decoder

        # Setup mock optimizer
        mock_optimizer = MagicMock()
        mock_adamw.return_value = mock_optimizer

        # Setup tokenizer encoding
        target_text = "Sample text for training"
        encoded_tokens = torch.tensor([101, 2023, 3231, 2005, 2367, 102])
        mock_tokenizer_instance.encode.return_value = encoded_tokens.unsqueeze(0)

        # Create reverser
        reverser = Seq2SeqReverser()

        # Mock the criterion to return a scalar tensor with requires_grad=True
        loss_tensor = torch.tensor(0.5, requires_grad=True)
        reverser.criterion = MagicMock(return_value=loss_tensor)

        # Create sample input
        source_rep = [[0.1, 0.2, 0.3] * 256 for _ in range(10)]

        # Patch the backward operation to avoid actual backprop
        with patch.object(torch.Tensor, "backward") as mock_backward:
            # Call train_step
            loss = reverser.train_step(source_rep, target_text)

            # Check that methods were called properly
            reverser.criterion.assert_called_once()
            reverser.optimizer.zero_grad.assert_called_once()
            mock_backward.assert_called_once()
            self.assertGreaterEqual(loss, 0.0)

    @patch("aparecium.reverser.optim.AdamW")
    @patch("aparecium.reverser.AutoTokenizer")
    @patch("aparecium.reverser.TransformerSeq2SeqModel")
    @patch("aparecium.reverser.torch.argmax")  # Patch argmax
    def test_generate_text(
        self, mock_argmax, mock_model_class, mock_tokenizer, mock_adamw
    ):
        """
        Test the generate_text method
        """
        # Setup mocks
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.__len__.return_value = 1000
        mock_tokenizer_instance.cls_token_id = 101
        mock_tokenizer_instance.sep_token_id = 102
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        # Setup mock model with parameters
        mock_decoder = MagicMock()
        mock_decoder.eval = MagicMock()
        mock_params = [torch.nn.Parameter(torch.randn(10, 10))]
        mock_decoder.parameters.return_value = mock_params

        # Make the model return a real tensor to avoid tensor operations on MagicMock
        # This will be used in log_softmax before argmax is called
        mock_decoder.return_value = torch.zeros(1, 1, 1000)
        mock_model_class.return_value = mock_decoder

        # Setup argmax to return a tensor with an item method
        token_tensor = torch.tensor(200)
        mock_argmax.return_value = token_tensor

        # Setup mock optimizer
        mock_optimizer = MagicMock()
        mock_adamw.return_value = mock_optimizer

        # Setup decoder to decode token ids to text
        mock_tokenizer_instance.decode.return_value = "Generated text"

        # Create reverser
        reverser = Seq2SeqReverser()

        # Create sample input
        source_rep = [[0.1, 0.2, 0.3] * 256 for _ in range(10)]

        # Call generate_text
        result = reverser.generate_text(source_rep, max_length=3)

        # Check result and that the expected methods were called
        mock_tokenizer_instance.decode.assert_called_once()
        self.assertEqual(result, "Generated text")

    @patch("aparecium.reverser.optim.AdamW")
    @patch("aparecium.reverser.os.makedirs")
    @patch("aparecium.reverser.torch.save")
    @patch("aparecium.reverser.AutoTokenizer")
    @patch("aparecium.reverser.TransformerSeq2SeqModel")
    def test_save_model(
        self, mock_model, mock_tokenizer, mock_torch_save, mock_makedirs, mock_adamw
    ):
        """
        Test saving the model
        """
        # Setup mocks
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.__len__.return_value = 1000
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        # Setup mock model with parameters
        mock_model_instance = MagicMock()
        mock_params = [torch.nn.Parameter(torch.randn(10, 10))]
        mock_model_instance.parameters.return_value = mock_params
        mock_model_instance.state_dict.return_value = {
            "layer1.weight": torch.rand(10, 10)
        }
        mock_model.return_value = mock_model_instance

        # Setup mock optimizer
        mock_optimizer = MagicMock()
        mock_adamw.return_value = mock_optimizer

        # Create reverser
        reverser = Seq2SeqReverser()
        reverser.decoder = mock_model_instance

        # Create a temp directory for testing
        with tempfile.TemporaryDirectory() as tmpdirname:
            # Call save_model
            reverser.save_model(tmpdirname)

            # Check that directories were created and torch.save was called
            mock_makedirs.assert_called_once_with(tmpdirname, exist_ok=True)
            mock_torch_save.assert_called_once()
            mock_tokenizer_instance.save_pretrained.assert_called_once_with(tmpdirname)

    @patch("aparecium.reverser.torch.load")
    @patch("aparecium.reverser.AutoTokenizer")
    @patch("aparecium.reverser.TransformerSeq2SeqModel")
    @patch("aparecium.reverser.optim.AdamW")
    def test_load_model(self, mock_adamw, mock_model, mock_tokenizer, mock_torch_load):
        """
        Test loading the model
        """
        # Setup mocks
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.__len__.return_value = 1000
        mock_tokenizer.from_pretrained.side_effect = [
            mock_tokenizer_instance,
            mock_tokenizer_instance,
        ]

        # Setup mock model with parameters
        mock_model_instance = MagicMock()
        mock_params = [torch.nn.Parameter(torch.randn(10, 10))]
        mock_model_instance.parameters.return_value = mock_params
        mock_model.return_value = mock_model_instance

        # Setup mock optimizer
        mock_optimizer = MagicMock()
        mock_adamw.return_value = mock_optimizer

        # Setup checkpoint
        mock_checkpoint = {
            "decoder_state_dict": {"layer1.weight": torch.rand(10, 10)},
            "config": {"model_name": "test-model", "d_model": 512, "lr": 1e-5},
        }
        mock_torch_load.return_value = mock_checkpoint

        # Create reverser
        reverser = Seq2SeqReverser()
        reverser.decoder = mock_model_instance

        # Create a temp directory for testing
        with tempfile.TemporaryDirectory() as tmpdirname:
            # Call load_model
            reverser.load_model(tmpdirname)

            # Check that model was loaded
            mock_torch_load.assert_called_once()
            mock_tokenizer.from_pretrained.assert_any_call(tmpdirname)
            mock_model_instance.load_state_dict.assert_called_once_with(
                mock_checkpoint["decoder_state_dict"]
            )
            # The model is moved to the device, we just check that to was called at least once
            self.assertTrue(mock_model_instance.to.called)

            # Check config was updated
            self.assertEqual(reverser.config["model_name"], "test-model")
            self.assertEqual(reverser.config["lr"], 1e-5)


if __name__ == "__main__":
    unittest.main()
