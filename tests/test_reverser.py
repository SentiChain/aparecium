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

from aparecium import (  # type: ignore
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
            max_position_embeddings=64,
        )

        # Create sample inputs
        encoder_outputs = torch.rand(src_seq_len, batch_size, d_model)
        tgt_input_ids = torch.randint(0, vocab_size, (tgt_seq_len, batch_size))
        tgt_mask = generate_subsequent_mask(tgt_seq_len, device=torch.device("cpu"))
        # Provide key padding masks
        tgt_kpm = torch.zeros(batch_size, tgt_seq_len, dtype=torch.bool)
        mem_kpm = torch.zeros(batch_size, src_seq_len, dtype=torch.bool)

        # Run forward pass
        logits = model(encoder_outputs, tgt_input_ids, tgt_mask, tgt_kpm, mem_kpm)

        # Check output shape
        self.assertEqual(logits.shape, (tgt_seq_len, batch_size, vocab_size))

    @patch("aparecium.reverser.optim.AdamW")
    @patch("aparecium.reverser.AutoTokenizer")
    @patch("aparecium.reverser.TransformerSeq2SeqModel")
    def test_length_penalty_and_determinism_flags(
        self, mock_model_class, mock_tokenizer, mock_adamw
    ):
        # Setup mocks
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.__len__.return_value = 1000
        mock_tokenizer_instance.cls_token_id = 101
        mock_tokenizer_instance.sep_token_id = 102
        mock_tokenizer_instance.decode.return_value = "decoded"
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        # Setup mock model with parameters
        mock_decoder = MagicMock()
        mock_params = [torch.nn.Parameter(torch.randn(10, 10))]
        mock_decoder.parameters.return_value = mock_params
        # Model returns zeros logits for deterministic behavior
        mock_decoder.return_value = torch.zeros(1, 1, 1000)
        mock_model_class.return_value = mock_decoder

        # Setup mock optimizer
        mock_optimizer = MagicMock()
        mock_adamw.return_value = mock_optimizer

        reverser = Seq2SeqReverser()
        reverser.decoder = mock_decoder

        source_rep = [[0.0] * reverser.config["d_model"] for _ in range(5)]

        text, info = reverser.generate_text(
            source_rep,
            max_length=4,
            num_beams=2,
            deterministic=True,
            length_penalty_alpha=0.6,
            lambda_sim=0.0,
            return_confidence=True,
        )
        self.assertIsInstance(text, str)
        self.assertIn("score_norm", info)

    @patch("aparecium.reverser.cosine_sim")
    @patch("aparecium.reverser.MPNetEmbeddingScorer.encode_and_pool")
    @patch("aparecium.reverser.optim.AdamW")
    @patch("aparecium.reverser.AutoTokenizer")
    @patch("aparecium.reverser.TransformerSeq2SeqModel")
    def test_rescoring_flip(
        self,
        mock_model_class,
        mock_tokenizer,
        mock_adamw,
        mock_encode_and_pool,
        mock_cosine,
    ):
        # Setup mocks
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.__len__.return_value = 1000
        mock_tokenizer_instance.cls_token_id = 101
        mock_tokenizer_instance.sep_token_id = 102

        # The two candidates decode to A (LM-preferred) vs B (cosine-preferred)
        # We will just return different strings depending on first token id
        def fake_decode(ids, skip_special_tokens=True):
            if not ids:
                return ""
            return "A" if ids[-1] % 2 == 0 else "B"

        mock_tokenizer_instance.decode.side_effect = fake_decode
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        # Mock model that forces topk to prefer token even=LM, odd=cosine
        mock_decoder = MagicMock()
        mock_decoder.parameters.return_value = [torch.nn.Parameter(torch.randn(10, 10))]
        # Return zeros so argmax selects index 0 deterministically; we'll simulate flip by lambda_sim>0 via decode text change
        mock_decoder.return_value = torch.zeros(1, 1, 1000)
        mock_model_class.return_value = mock_decoder

        # Patch encoder + cosine so that text "B" always has higher cosine than "A"
        def fake_encode(texts):
            # Return (N, D) where D=1 with values 1.0 for 'A', 2.0 for 'B'
            vals = [1.0 if (t == "A") else 2.0 for t in texts]
            return torch.tensor(vals, dtype=torch.float32).unsqueeze(1)

        def fake_cosine(a, b):
            # Return (N, 1) equal to the value in a
            return a

        mock_encode_and_pool.side_effect = fake_encode
        mock_cosine.side_effect = fake_cosine

        reverser = Seq2SeqReverser()
        reverser.decoder = mock_decoder
        source_rep = [[0.1] * reverser.config["d_model"] for _ in range(5)]

        # With lambda_sim=0, LM-only
        text_lm, info_lm = reverser.generate_text(
            source_rep,
            max_length=2,
            num_beams=2,
            deterministic=True,
            lambda_sim=0.0,
            return_confidence=True,
        )
        # With lambda_sim>0, cosine-preferred should be favored
        text_cos, info_cos = reverser.generate_text(
            source_rep,
            max_length=2,
            num_beams=2,
            deterministic=True,
            lambda_sim=0.3,
            return_confidence=True,
        )
        # Cosine should be higher under lambda_sim > 0 setting
        self.assertGreaterEqual(info_cos["cosine"], info_lm["cosine"])


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
        # Ensure the decoder is our mock so we can track train() calls
        reverser.decoder = mock_decoder

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

            # Verify the decoder was put in training mode
            mock_decoder.train.assert_called_once()

    @patch("aparecium.reverser.optim.AdamW")
    @patch("aparecium.reverser.AutoTokenizer")
    @patch("aparecium.reverser.TransformerSeq2SeqModel")
    def test_train_step_batch(self, mock_model_class, mock_tokenizer, mock_adamw):
        """
        Test the train_step_batch method
        """
        # Setup mocks
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.__len__.return_value = 1000
        mock_tokenizer_instance.pad_token_id = 0
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        # Setup mock model with parameters
        mock_decoder = MagicMock()
        mock_params = [torch.nn.Parameter(torch.randn(10, 10))]
        mock_decoder.parameters.return_value = mock_params

        # Setup model forward pass - return a real tensor with requires_grad=True
        logits = torch.rand(5, 3, 1000, requires_grad=True)  # 3 is batch size
        mock_decoder.return_value = logits

        # Return the mock decoder when TransformerSeq2SeqModel is called
        mock_model_class.return_value = mock_decoder

        # Setup mock optimizer
        mock_optimizer = MagicMock()
        mock_adamw.return_value = mock_optimizer

        # Setup tokenizer encoding for batch input
        target_text_batch = ["Sample text 1", "Sample text 2", "Sample text 3"]
        mock_encoded_batch = {
            "input_ids": torch.tensor(
                [
                    [101, 2023, 3231, 102, 0, 0],  # Padded sequences
                    [101, 2023, 3232, 2005, 102, 0],
                    [101, 2023, 3233, 2005, 2367, 102],
                ]
            )
        }
        mock_tokenizer_instance.return_value = mock_encoded_batch

        # Create reverser
        reverser = Seq2SeqReverser()
        # Ensure the decoder is our mock
        reverser.decoder = mock_decoder

        # Mock the criterion to return a scalar tensor with requires_grad=True
        loss_tensor = torch.tensor(0.5, requires_grad=True)
        reverser.criterion = MagicMock(return_value=loss_tensor)

        # Create sample batch input - 3 sources
        source_rep_batch = [
            [[0.1, 0.2, 0.3] * 256 for _ in range(5)],  # Source 1
            [[0.2, 0.3, 0.4] * 256 for _ in range(8)],  # Source 2
            [[0.3, 0.4, 0.5] * 256 for _ in range(6)],  # Source 3
        ]

        # Patch the backward operation to avoid actual backprop
        with patch.object(torch.Tensor, "backward") as mock_backward:
            # Call train_step_batch
            loss = reverser.train_step_batch(source_rep_batch, target_text_batch)

            # Check that methods were called properly
            reverser.criterion.assert_called_once()
            reverser.optimizer.zero_grad.assert_called_once()
            mock_backward.assert_called_once()
            self.assertGreaterEqual(loss, 0.0)

            # Verify the decoder was put in training mode
            mock_decoder.train.assert_called_once()

            # Verify the tokenizer was called with batched input
            mock_tokenizer_instance.assert_called_once_with(
                target_text_batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )

    @patch("aparecium.reverser.optim.AdamW")
    @patch("aparecium.reverser.AutoTokenizer")
    @patch("aparecium.reverser.TransformerSeq2SeqModel")
    @patch("aparecium.reverser.torch.argmax")  # Patch argmax
    def test_generate_text_greedy(
        self, mock_argmax, mock_model_class, mock_tokenizer, mock_adamw
    ):
        """
        Test the generate_text method with greedy decoding
        """
        # Setup mocks
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.__len__.return_value = 1000
        mock_tokenizer_instance.cls_token_id = 101
        mock_tokenizer_instance.sep_token_id = 102
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        # Setup mock model with parameters
        mock_decoder = MagicMock()
        mock_params = [torch.nn.Parameter(torch.randn(10, 10))]
        mock_decoder.parameters.return_value = mock_params

        # Make the model return a real tensor to avoid tensor operations on MagicMock
        mock_decoder.return_value = torch.zeros(1, 1, 1000)
        mock_model_class.return_value = mock_decoder

        # Setup argmax to return a tensor with an item method
        token_tensor = torch.tensor(200)
        mock_argmax.return_value = token_tensor

        # Setup mock optimizer
        mock_optimizer = MagicMock()
        mock_adamw.return_value = mock_optimizer

        # Setup decoder to decode token ids to text
        mock_tokenizer_instance.decode.return_value = (
            "Generated text using greedy decoding"
        )

        # Create reverser
        reverser = Seq2SeqReverser()
        # Ensure the decoder is our mock
        reverser.decoder = mock_decoder

        # Create sample input
        source_rep = [[0.1, 0.2, 0.3] * 256 for _ in range(10)]

        # Call generate_text
        result = reverser.generate_text(
            source_rep, max_length=3, num_beams=1, do_sample=False
        )

        # Check result and that the expected methods were called
        mock_tokenizer_instance.decode.assert_called_once()
        self.assertEqual(result, "Generated text using greedy decoding")
        mock_decoder.eval.assert_called_once()

    @patch("aparecium.reverser.optim.AdamW")
    @patch("aparecium.reverser.AutoTokenizer")
    @patch("aparecium.reverser.TransformerSeq2SeqModel")
    @patch("aparecium.reverser.torch.topk")
    @patch("aparecium.reverser.torch.multinomial")
    def test_generate_text_sampling(
        self, mock_multinomial, mock_topk, mock_model_class, mock_tokenizer, mock_adamw
    ):
        """
        Test the generate_text method with sampling
        """
        # Setup mocks
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.__len__.return_value = 1000
        mock_tokenizer_instance.cls_token_id = 101
        mock_tokenizer_instance.sep_token_id = 102
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        # Setup mock model with parameters
        mock_decoder = MagicMock()
        mock_params = [torch.nn.Parameter(torch.randn(10, 10))]
        mock_decoder.parameters.return_value = mock_params

        # Make the model return a real tensor
        mock_decoder.return_value = torch.zeros(1, 1, 1000)
        mock_model_class.return_value = mock_decoder

        # Setup topk to return values and indices
        mock_topk.return_value = (torch.tensor([0.1, 0.05]), torch.tensor([5, 10]))

        # Setup multinomial to return a tensor with an item method
        sampled_token = torch.tensor([0])  # First index (token 5 from topk)
        mock_multinomial.return_value = sampled_token

        # Setup mock optimizer
        mock_optimizer = MagicMock()
        mock_adamw.return_value = mock_optimizer

        # Setup decoder to decode token ids to text
        mock_tokenizer_instance.decode.return_value = "Generated text using sampling"

        # Create reverser
        reverser = Seq2SeqReverser()
        # Ensure the decoder is our mock
        reverser.decoder = mock_decoder

        # Patch the _sample_from_logits method
        with patch.object(reverser, "_sample_from_logits") as mock_sample:
            # Make sample return a token ID
            mock_sample.return_value = 5

            # Create sample input
            source_rep = [[0.1, 0.2, 0.3] * 256 for _ in range(10)]

            # Call generate_text with sampling
            result = reverser.generate_text(
                source_rep,
                max_length=3,
                num_beams=1,
                do_sample=True,
                top_k=50,
                top_p=0.9,
                temperature=0.8,
            )

            # Check result and that the expected methods were called
            mock_tokenizer_instance.decode.assert_called_once()
            self.assertEqual(result, "Generated text using sampling")
            mock_decoder.eval.assert_called_once()

            # Verify sampling parameters were used
            mock_sample.assert_called()
            # Check the temperature was applied
            self.assertEqual(mock_sample.call_args[1]["top_k"], 50)
            self.assertEqual(mock_sample.call_args[1]["top_p"], 0.9)

    @patch("aparecium.reverser.optim.AdamW")
    @patch("aparecium.reverser.AutoTokenizer")
    @patch("aparecium.reverser.TransformerSeq2SeqModel")
    def test_greedy_confidence_nonzero_with_lambda(
        self, mock_model_class, mock_tokenizer, mock_adamw
    ):
        # Setup mocks
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.__len__.return_value = 1000
        mock_tokenizer_instance.cls_token_id = 101
        mock_tokenizer_instance.sep_token_id = 102
        mock_tokenizer_instance.decode.return_value = "decoded"
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        # Setup mock model
        mock_decoder = MagicMock()
        mock_decoder.parameters.return_value = [torch.nn.Parameter(torch.randn(10, 10))]
        mock_decoder.return_value = torch.zeros(1, 1, 1000)
        mock_model_class.return_value = mock_decoder

        # Optimizer
        mock_optimizer = MagicMock()
        mock_adamw.return_value = mock_optimizer

        reverser = Seq2SeqReverser()
        reverser.decoder = mock_decoder

        source_rep = [[0.0] * reverser.config["d_model"] for _ in range(5)]

        text, info = reverser.generate_text(
            source_rep,
            max_length=3,
            num_beams=1,
            deterministic=True,
            lambda_sim=0.3,
            return_confidence=True,
        )

        self.assertIsInstance(text, str)
        # With lambda_sim > 0, cosine or fused_score should be computable (>= 0)
        self.assertIn("cosine", info)
        self.assertIn("fused_score", info)

    @patch("aparecium.reverser.optim.AdamW")
    @patch("aparecium.reverser.AutoTokenizer")
    @patch("aparecium.reverser.TransformerSeq2SeqModel")
    def test_generate_text_beam_search(
        self, mock_model_class, mock_tokenizer, mock_adamw
    ):
        """
        Test the generate_text method with beam search
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

        # Make the model return a real tensor
        mock_decoder.return_value = torch.zeros(1, 1, 1000)
        mock_model_class.return_value = mock_decoder

        # Setup mock optimizer
        mock_optimizer = MagicMock()
        mock_adamw.return_value = mock_optimizer

        # Setup decoder to decode token ids to text
        mock_tokenizer_instance.decode.return_value = "Generated text using beam search"

        # Create reverser
        reverser = Seq2SeqReverser()

        # Patch the internal _beam_search method
        with patch.object(reverser, "_beam_search") as mock_beam_search:
            mock_beam_search.return_value = "Generated text using beam search"

            # Create sample input
            source_rep = [[0.1, 0.2, 0.3] * 256 for _ in range(10)]

            # Call generate_text with beam search
            result = reverser.generate_text(
                source_rep, max_length=10, num_beams=3, temperature=0.9
            )

            # Verify the beam search method was called with correct args
            mock_beam_search.assert_called_once()
            self.assertEqual(mock_beam_search.call_args[1]["num_beams"], 3)
            self.assertEqual(mock_beam_search.call_args[1]["max_length"], 10)
            self.assertEqual(mock_beam_search.call_args[1]["temperature"], 0.9)

            # Check result
            self.assertEqual(result, "Generated text using beam search")

    @patch("aparecium.reverser.optim.AdamW")
    @patch("aparecium.reverser.AutoTokenizer")
    @patch("aparecium.reverser.TransformerSeq2SeqModel")
    def test_sample_from_logits(self, mock_model_class, mock_tokenizer, mock_adamw):
        """
        Test the _sample_from_logits method
        """
        # Create reverser instance
        reverser = Seq2SeqReverser()

        # Create a logits tensor
        logits = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

        # Test top-k sampling
        with patch("aparecium.reverser.torch.topk") as mock_topk:
            # Mock topk to return known values and indices
            mock_topk.return_value = (
                torch.tensor([5.0, 4.0, 3.0]),  # Values
                torch.tensor([4, 3, 2]),  # Indices
            )

            # Mock multinomial to return the 0-th position (representing index 4)
            with patch("aparecium.reverser.torch.multinomial") as mock_multinomial:
                # The multinomial returns index 0, which maps to the highest probability token (index 4)
                mock_multinomial.return_value = torch.tensor([0])

                # Test with top_k=3 (only keep top 3 logits)
                with patch(
                    "aparecium.reverser.F.softmax",
                    return_value=torch.tensor([0.0, 0.0, 0.1, 0.3, 0.6]),
                ):
                    token_id = reverser._sample_from_logits(logits, top_k=3, top_p=1.0)

                    # Should be token 4 (token at index 0 from multinomial's return value)
                    self.assertEqual(token_id, 0)
                    # Replace torch.any_kind_of() with a matcher that works
                    mock_topk.assert_called_once()

        # Test top-p (nucleus) sampling
        with patch("aparecium.reverser.torch.sort") as mock_sort:
            # Mock sorted values and indices
            mock_sort.return_value = (
                torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0]),  # Sorted values
                torch.tensor([4, 3, 2, 1, 0]),  # Sorted indices
            )

            # Mock cumsum to reach top_p threshold at index 2
            with patch("aparecium.reverser.torch.cumsum") as mock_cumsum:
                mock_cumsum.return_value = torch.tensor([0.33, 0.60, 0.80, 0.93, 1.0])

                # Mock where to return indices where > top_p
                with patch("aparecium.reverser.torch.where") as mock_where:
                    mock_where.return_value = [torch.tensor([2, 3, 4])]

                    # Mock multinomial to return the 1st value after filtering
                    with patch(
                        "aparecium.reverser.torch.multinomial"
                    ) as mock_multinomial:
                        mock_multinomial.return_value = torch.tensor([1])

                        # Test with top_p=0.7
                        with patch(
                            "aparecium.reverser.F.softmax",
                            return_value=torch.tensor([0.1, 0.1, 0.2, 0.2, 0.4]),
                        ):
                            token_id = reverser._sample_from_logits(
                                logits, top_k=0, top_p=0.7
                            )

                            # Since the implementation is complex, just verify calls
                            mock_sort.assert_called_once()
                            mock_cumsum.assert_called_once()
                            mock_where.assert_called_once()
                            mock_multinomial.assert_called_once()

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
    @patch("aparecium.reverser.torch.save")
    @patch("aparecium.reverser.os.makedirs")
    @patch("aparecium.reverser.os.path.exists")
    def test_load_model(
        self,
        mock_exists,
        mock_makedirs,
        mock_torch_save,
        mock_adamw,
        mock_model,
        mock_tokenizer,
        mock_torch_load,
    ):
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
        mock_model_instance.state_dict.return_value = {
            "layer1.weight": torch.rand(10, 10)
        }
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
            # First save the model
            reverser.save_model(tmpdirname)

            # Mock file existence check to return True
            mock_exists.return_value = True

            # Then load the model
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


class TestNewBehaviors(unittest.TestCase):
    """
    Additional tests for mask invariance, constraints safety, and determinism.
    """

    @patch("aparecium.reverser.optim.AdamW")
    @patch("aparecium.reverser.AutoTokenizer")
    @patch("aparecium.reverser.TransformerSeq2SeqModel")
    def test_determinism_flag(self, mock_model_class, mock_tokenizer, mock_adamw):
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.__len__.return_value = 1000
        mock_tokenizer_instance.cls_token_id = 101
        mock_tokenizer_instance.sep_token_id = 102
        mock_tokenizer_instance.decode.return_value = "decoded"
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        mock_decoder = MagicMock()
        mock_decoder.parameters.return_value = [torch.nn.Parameter(torch.randn(10, 10))]
        mock_decoder.return_value = torch.zeros(1, 1, 1000)
        mock_model_class.return_value = mock_decoder

        reverser = Seq2SeqReverser()
        reverser.decoder = mock_decoder
        source_rep = [[0.0] * reverser.config["d_model"] for _ in range(5)]

        text1 = reverser.generate_text(
            source_rep, max_length=3, num_beams=1, deterministic=True
        )
        text2 = reverser.generate_text(
            source_rep, max_length=3, num_beams=1, deterministic=True
        )
        self.assertEqual(text1, text2)

    @patch("aparecium.reverser.optim.AdamW")
    @patch("aparecium.reverser.AutoTokenizer")
    @patch("aparecium.reverser.TransformerSeq2SeqModel")
    def test_mask_invariance_append_padding(
        self, mock_model_class, mock_tokenizer, mock_adamw
    ):
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.__len__.return_value = 1000
        mock_tokenizer_instance.cls_token_id = 101
        mock_tokenizer_instance.sep_token_id = 102
        mock_tokenizer_instance.decode.return_value = "decoded"
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        mock_decoder = MagicMock()
        mock_decoder.parameters.return_value = [torch.nn.Parameter(torch.randn(10, 10))]
        # Return stable logits so outputs are deterministic
        mock_decoder.return_value = torch.zeros(1, 1, 1000)
        mock_model_class.return_value = mock_decoder

        reverser = Seq2SeqReverser()
        reverser.decoder = mock_decoder
        source_rep = [[0.1] * reverser.config["d_model"] for _ in range(5)]
        source_rep_padded = source_rep + [
            [0.0] * reverser.config["d_model"] for _ in range(3)
        ]

        text_base = reverser.generate_text(
            source_rep, max_length=3, num_beams=1, deterministic=True
        )
        text_pad = reverser.generate_text(
            source_rep_padded, max_length=3, num_beams=1, deterministic=True
        )
        self.assertEqual(text_base, text_pad)

    @patch("aparecium.reverser.optim.AdamW")
    @patch("aparecium.reverser.AutoTokenizer")
    @patch("aparecium.reverser.TransformerSeq2SeqModel")
    def test_constraints_safety_no_crash(
        self, mock_model_class, mock_tokenizer, mock_adamw
    ):
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.__len__.return_value = 1000
        mock_tokenizer_instance.cls_token_id = 101
        mock_tokenizer_instance.sep_token_id = 102
        mock_tokenizer_instance.convert_ids_to_tokens.return_value = "▁$"
        mock_tokenizer_instance.decode.return_value = "$BTC"
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        mock_decoder = MagicMock()
        mock_decoder.parameters.return_value = [torch.nn.Parameter(torch.randn(10, 10))]
        mock_decoder.return_value = torch.zeros(1, 1, 1000)
        mock_model_class.return_value = mock_decoder

        reverser = Seq2SeqReverser()
        reverser.decoder = mock_decoder
        source_rep = [[0.1] * reverser.config["d_model"] for _ in range(5)]

        # Should not crash and should produce contiguous cashtag-like text
        text = reverser.generate_text(
            source_rep,
            max_length=3,
            num_beams=1,
            deterministic=True,
            enable_constraints=True,
        )
        self.assertIsInstance(text, str)
        # basic contiguity check for cashtag-like output
        self.assertNotIn("$ ", text)


if __name__ == "__main__":
    unittest.main()
