"""Network-free checks of public reconstruction behavior."""

from __future__ import annotations

import concurrent.futures
from contextlib import ExitStack
import threading
import time
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from aparecium.api import Aparecium, MODEL_FILES


def vector(index: int = 0, scale: float = 1.0) -> np.ndarray:
    result = np.zeros(768, dtype=np.float32)
    result[index] = scale
    return result


class TestApareciumAPI(unittest.TestCase):
    def setUp(self) -> None:
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.download = self.stack.enter_context(patch("aparecium.api.snapshot_download", return_value="/models/aparecium"))
        self.load = self.stack.enter_context(patch("aparecium.api.PrefixDecoder.load"))
        self.encoder_class = self.stack.enter_context(patch("aparecium.api.SentenceTransformer"))
        self.decoder = self.load.return_value
        self.decoder.config = SimpleNamespace(
            embedding_dim=768,
            encoder_model="sentence-transformers/all-mpnet-base-v2",
            encoder_revision="0123456789abcdef",
            max_length=64,
        )
        self.encoder = self.encoder_class.return_value
        self.encoder.encode.side_effect = lambda texts, **kwargs: np.stack([vector() for _ in texts])
        self.decoder.generate.side_effect = lambda vectors, num_candidates, **kwargs: [
            {"candidates": [{"text": f"Candidate sentence number {i}", "stopped": True}
                            for i in range(num_candidates)]}
            for _ in vectors
        ]
        self.model = Aparecium(device="cpu")

    def test_download_uses_default_repository_and_only_runtime_files(self) -> None:
        self.download.assert_called_once_with(
            repo_id="SentiChain/aparecium", revision=None,
            allow_patterns=list(MODEL_FILES), local_files_only=False,
        )
        self.load.assert_called_once_with(Path("/models/aparecium"), device="cpu", local_files_only=False)
        self.encoder_class.assert_not_called()

    def test_explicit_model_revision_is_forwarded(self) -> None:
        self.download.reset_mock()
        Aparecium(model_id="example/model", revision="0123456789abcdef", device="cpu")
        self.download.assert_called_once_with(
            repo_id="example/model", revision="0123456789abcdef",
            allow_patterns=list(MODEL_FILES), local_files_only=False,
        )

    def test_local_model_and_offline_mode_propagate_to_dependencies(self) -> None:
        self.download.reset_mock()
        model = Aparecium(model_path="/local/model", device="cpu", local_files_only=True)
        model.invert_embedding(vector())
        self.download.assert_not_called()
        self.load.assert_called_with(Path("/local/model"), device="cpu", local_files_only=True)
        self.encoder_class.assert_called_once_with(
            self.decoder.config.encoder_model, revision=self.decoder.config.encoder_revision,
            device="cpu", local_files_only=True,
        )
        self.encoder.float.assert_called_once()

    def test_scores_align_with_generation_order_and_fluency_filter(self) -> None:
        self.decoder.generate.side_effect = None
        self.decoder.generate.return_value = [{"candidates": [
            {"text": "Short", "stopped": True},
            {"text": "The roads reopened", "stopped": False},
            {"text": "A dog barked", "stopped": True},
        ]}]
        self.encoder.encode.side_effect = None
        self.encoder.encode.return_value = np.stack([vector(), vector() * 0.6 + vector(1) * 0.8, vector(1)])
        result = self.model.invert_embedding(vector(scale=7), num_candidates=3, max_length=24, seed=17)
        self.assertEqual(result.text, "The roads reopened")
        self.assertEqual(result.candidates, ["Short", "The roads reopened", "A dog barked"])
        np.testing.assert_allclose(result.scores, [1, 0.6, 0], atol=1e-6)
        self.assertEqual(result.selected_index, 1)
        self.assertFalse(result.stopped)
        call = self.decoder.generate.call_args
        np.testing.assert_allclose(call.args[0].cpu().numpy(), [vector()])
        self.assertEqual(call.kwargs, {"num_candidates": 3, "max_length": 24, "seed": 17})

    def test_all_candidates_are_eligible_when_none_pass_fluency_floor(self) -> None:
        self.decoder.generate.side_effect = None
        self.decoder.generate.return_value = [{"candidates": [
            {"text": "", "stopped": True}, {"text": "Rain", "stopped": True},
        ]}]
        self.encoder.encode.side_effect = None
        self.encoder.encode.return_value = np.stack([vector(1), vector()])
        result = self.model.invert_embedding(vector(), num_candidates=2)
        self.assertEqual(result.selected_index, 1)
        self.assertEqual(result.text, "Rain")

    def test_batch_order_and_chunk_sizes_are_preserved(self) -> None:
        def generate(vectors, num_candidates, **kwargs):
            return [{"candidates": [{"text": f"Sentence for vector {int(v.argmax())}", "stopped": True}]
                     } for v in vectors]

        self.decoder.generate.side_effect = generate
        results = self.model.invert_embeddings(np.stack([vector(i) for i in range(5)]),
                                              num_candidates=1, batch_size=2)
        self.assertEqual([item.text for item in results], [f"Sentence for vector {i}" for i in range(5)])
        self.assertEqual([len(call.args[0]) for call in self.decoder.generate.call_args_list], [2, 2, 1])
        self.assertTrue(all(call.kwargs["batch_size"] == 32 for call in self.encoder.encode.call_args_list))

    def test_embed_texts_uses_stock_encoder_and_preserves_order(self) -> None:
        results = self.model.invert_texts(["The train arrived.", "The doors opened.", "Passengers left."], batch_size=2)
        self.assertEqual(len(results), 3)
        calls = self.encoder.encode.call_args_list
        self.assertEqual(calls[0].args[0], ["The train arrived.", "The doors opened."])
        self.assertEqual(calls[2].args[0], ["Passengers left."])
        for call in calls:
            self.assertTrue(call.kwargs["normalize_embeddings"])
            self.assertTrue(call.kwargs["convert_to_numpy"])
            self.assertFalse(call.kwargs["show_progress_bar"])
        self.encoder_class.assert_called_once()

    def test_single_text_returns_one_result(self) -> None:
        result = self.model.invert_text("A train arrived.", num_candidates=2, max_length=32, seed=5)
        self.assertEqual(result.selected_index, 0)
        self.assertEqual(len(result.candidates), 2)
        self.assertEqual(self.encoder.encode.call_args_list[0].args[0], ["A train arrived."])

    def test_empty_batches_do_not_run_models(self) -> None:
        for empty in ([], np.empty((0, 768)), torch.empty(0, 768)):
            self.assertEqual(self.model.invert_embeddings(empty), [])
        self.assertEqual(self.model.invert_texts([]), [])
        self.decoder.generate.assert_not_called()
        self.encoder.encode.assert_not_called()

    def test_invalid_vectors_are_rejected_before_generation(self) -> None:
        invalid = [
            [0.1, 0.2], np.zeros(768), np.full(768, np.nan), np.full(768, np.inf),
            [True] + [0.0] * 767, ["1"] + [0.0] * 767,
            np.ones(768, dtype=complex), np.ones(768, dtype=bool),
            torch.ones(768, dtype=torch.bool), torch.ones(768, dtype=torch.complex64),
            np.ones((1, 768)), None,
        ]
        for value in invalid:
            with self.subTest(value_type=type(value), shape=getattr(value, "shape", None)):
                with self.assertRaises(ValueError):
                    self.model.invert_embedding(value)
        self.decoder.generate.assert_not_called()

    def test_invalid_batch_shapes_are_rejected(self) -> None:
        for value in (vector(), [[1], [2, 3]], np.empty((0, 767)), np.empty((1, 0)), [[0] * 768]):
            with self.subTest(shape=getattr(value, "shape", None)):
                with self.assertRaises(ValueError):
                    self.model.invert_embeddings(value)

    def test_extreme_finite_magnitudes_normalize_without_overflow(self) -> None:
        for scale in (1e308, 1e-308, 1e30, 1e-30):
            value = np.zeros(768, dtype=np.float64)
            value[0] = scale
            self.model.invert_embedding(value)
            np.testing.assert_allclose(self.decoder.generate.call_args.args[0].cpu().numpy(), [vector()])

    def test_ordinary_vectors_use_exact_float32_tensor_normalization(self) -> None:
        values = np.random.default_rng(11).normal(size=(8, 768)).astype(np.float32)
        expected = torch.nn.functional.normalize(torch.tensor(values), dim=-1)
        self.model.invert_embeddings(values)
        actual = self.decoder.generate.call_args.args[0]
        self.assertEqual(actual.dtype, torch.float32)
        self.assertTrue(torch.equal(actual, expected))

    def test_extreme_vector_protection_does_not_change_ordinary_batch_rows(self) -> None:
        values = np.random.default_rng(12).normal(size=(3, 768))
        values[1] *= 1e300
        expected = torch.nn.functional.normalize(torch.tensor(values[[0, 2]], dtype=torch.float32), dim=-1)
        self.model.invert_embeddings(values)
        actual = self.decoder.generate.call_args.args[0]
        self.assertTrue(torch.equal(actual[[0, 2]], expected))
        self.assertTrue(torch.isfinite(actual).all().item())
        torch.testing.assert_close(actual.norm(dim=-1), torch.ones(3))

    def test_reranking_preserves_encoder_normalized_vectors_without_rescaling(self) -> None:
        rng = np.random.default_rng(13)
        values = rng.normal(size=(1, 768)).astype(np.float32)
        candidates = torch.nn.functional.normalize(torch.tensor(rng.normal(size=(5, 768)), dtype=torch.float32), dim=-1).numpy()
        self.encoder.encode.side_effect = None
        self.encoder.encode.return_value = candidates
        actual = self.model.invert_embedding(values[0])
        target = torch.nn.functional.normalize(torch.tensor(values), dim=-1).numpy()[0]
        np.testing.assert_array_equal(actual.scores, candidates @ target)
        self.encoder.encode.assert_called_once()
        self.assertEqual(self.encoder.encode.call_args.kwargs["batch_size"], 32)

    def test_tensors_are_detached_before_conversion(self) -> None:
        value = torch.zeros(768, requires_grad=True)
        with torch.no_grad():
            value[0] = 1
        self.model.invert_embedding(value)
        self.assertFalse(self.decoder.generate.call_args.args[0].requires_grad)

    def test_invalid_options_and_texts_fail_without_generation(self) -> None:
        for kwargs in ({"num_candidates": 0}, {"num_candidates": True}, {"num_candidates": 1.5},
                       {"num_candidates": 33}, {"max_length": 0}, {"max_length": 65},
                       {"seed": -1}, {"seed": True}, {"seed": 2**63}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.model.invert_embedding(vector(), **kwargs)
        for batch_size in (0, -1, True, 1.5):
            with self.subTest(batch_size=batch_size), self.assertRaises(ValueError):
                self.model.invert_embeddings([vector()], batch_size=batch_size)
        for texts in ("one text", [""], ["   "], [1], [None]):
            with self.subTest(texts=texts), self.assertRaises(ValueError):
                self.model.invert_texts(texts)
        self.decoder.generate.assert_not_called()

    def test_numpy_integer_options_are_passed_as_python_integers(self) -> None:
        self.model.invert_embedding(vector(), num_candidates=np.int64(1), max_length=np.int64(24), seed=np.int64(7))
        self.assertTrue(all(type(value) is int for value in self.decoder.generate.call_args.kwargs.values()))

    def test_decoder_and_encoder_cardinality_errors_are_explicit(self) -> None:
        self.decoder.generate.side_effect = None
        for drafts in ([], [{"candidates": []}]):
            self.decoder.generate.return_value = drafts
            with self.assertRaisesRegex(RuntimeError, "unexpected number"):
                self.model.invert_embedding(vector())
        self.decoder.generate.return_value = [{"candidates": [{"text": "A train arrived", "stopped": True}]}]
        self.encoder.encode.side_effect = None
        self.encoder.encode.return_value = np.stack([vector(), vector()])
        with self.assertRaisesRegex(RuntimeError, "unexpected number"):
            self.model.invert_embedding(vector(), num_candidates=1)

    def test_calls_on_one_instance_are_serialized(self) -> None:
        state = {"active": 0, "peak": 0}
        state_lock = threading.Lock()

        def generate(vectors, **kwargs):
            with state_lock:
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
            time.sleep(0.02)
            with state_lock:
                state["active"] -= 1
            return [{"candidates": [{"text": "A train arrived", "stopped": True}]}]

        self.decoder.generate.side_effect = generate
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            results = list(executor.map(lambda _: self.model.invert_embedding(vector(), num_candidates=1), range(3)))
        self.assertEqual(len(results), 3)
        self.assertEqual(state["peak"], 1)

    def test_default_device_is_cpu_without_cuda_even_if_mps_exists(self) -> None:
        with patch("torch.cuda.is_available", return_value=False), patch("torch.backends.mps.is_available", return_value=True):
            self.assertEqual(str(Aparecium().device), "cpu")

    def test_unavailable_devices_fail_before_downloading(self) -> None:
        self.download.reset_mock()
        with patch("torch.cuda.is_available", return_value=False), patch("torch.backends.mps.is_available", return_value=False):
            for device in ("cuda", "mps", "cpu:0", "mps:0", "meta", "invalid"):
                with self.subTest(device=device), self.assertRaises(ValueError):
                    Aparecium(device=device)
        with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.device_count", return_value=1):
            with self.assertRaises(ValueError):
                Aparecium(device="cuda:1")
        self.download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
