"""
Basic tests for the high-level Aparecium v2 API.

These tests are lightweight and do not touch the network:
- We mock `hf_hub_download` so no real Hugging Face calls are made.
- We mock `SentenceTransformer` so no real model is downloaded.
- We mock `load_models` and `deterministic_beam_search` to avoid heavy
  model instantiation and decoding.
"""

import unittest
from unittest.mock import MagicMock, patch


class TestApareciumAPI(unittest.TestCase):
    @patch("aparecium.api.SentenceTransformer")
    @patch("aparecium.api.deterministic_beam_search")
    @patch("aparecium.api.load_models")
    @patch("aparecium.api.hf_hub_download")
    def test_invert_embedding_uses_decoder_and_returns_top_candidate(
        self,
        mock_hf_hub_download,
        mock_load_models,
        mock_det_beam,
        mock_sentence_transformer,
    ):
        # Arrange: mock checkpoint paths and model components
        mock_hf_hub_download.side_effect = ["ckpt_path.pt", "r_ckpt_path.pt"]

        tokenizer = MagicMock()
        adapter = MagicMock()
        sketcher = MagicMock()
        decoder = MagicMock()
        rnet = MagicMock()
        mock_load_models.return_value = (
            tokenizer,
            adapter,
            sketcher,
            decoder,
            rnet,
        )

        # determinisitc_beam_search will return a fixed set of candidates
        mock_det_beam.return_value = {"texts": [["c0", "c1", "c2"]]}

        from aparecium import Aparecium

        # Act
        model = Aparecium(device="cpu")
        result = model.invert_embedding([0.1, 0.2, 0.3], beam=3, max_len=32)

        # Assert
        self.assertEqual(result.text, "c0")
        self.assertEqual(result.candidates, ["c0", "c1", "c2"])
        mock_hf_hub_download.assert_called()  # at least once for the main ckpt
        mock_load_models.assert_called_once()
        mock_det_beam.assert_called_once()

    @patch("aparecium.api.SentenceTransformer")
    @patch("aparecium.api.deterministic_beam_search")
    @patch("aparecium.api.load_models")
    @patch("aparecium.api.hf_hub_download")
    def test_invert_text_embeds_then_inverts(
        self,
        mock_hf_hub_download,
        mock_load_models,
        mock_det_beam,
        mock_sentence_transformer,
    ):
        # Arrange
        mock_hf_hub_download.side_effect = ["ckpt_path.pt", "r_ckpt_path.pt"]

        tokenizer = MagicMock()
        adapter = MagicMock()
        sketcher = MagicMock()
        decoder = MagicMock()
        rnet = MagicMock()
        mock_load_models.return_value = (
            tokenizer,
            adapter,
            sketcher,
            decoder,
            rnet,
        )

        # Mock SentenceTransformer.encode to produce a fake pooled vector
        st_instance = MagicMock()
        st_instance.encode.return_value = [[0.1, 0.2, 0.3]]
        mock_sentence_transformer.return_value = st_instance

        mock_det_beam.return_value = {"texts": [["decoded"]]}

        from aparecium import Aparecium

        model = Aparecium(device="cpu")

        # Act
        out = model.invert_text("some crypto post", beam=2, max_len=16)

        # Assert
        self.assertEqual(out.text, "decoded")
        st_instance.encode.assert_called_once()
        mock_det_beam.assert_called_once()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
