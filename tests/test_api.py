"""Matrix-only inference, strict inputs, and native token-state encoding."""

import concurrent.futures
import threading
import time
import unittest
from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from aparecium import Aparecium, InvertResult
from aparecium.api import DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION, MODEL_FILES, _validate_matrix
from aparecium.config import ENCODER_MODEL_ID, ENCODER_REVISION


_PREVIOUS_THREADS = None


def setUpModule():
    global _PREVIOUS_THREADS
    _PREVIOUS_THREADS = torch.get_num_threads()
    torch.set_num_threads(4)


def tearDownModule():
    torch.set_num_threads(_PREVIOUS_THREADS)


def matrix(ids=(0, 7, 8, 2)):
    values = np.zeros((len(ids), 768), dtype=np.float32)
    values[:, 0] = ids
    return values


class EncoderTokenizer:
    def __init__(self):
        self.calls = []
        self.long = False

    def __call__(self, texts, **kwargs):
        self.calls.append((texts, kwargs))
        return {"input_ids": [[0] + ([3] * 255 if self.long else list(range(3, 3 + len(text.split())))) + [2]
                              for text in texts]}

    def pad(self, encoded, **kwargs):
        ids = encoded["input_ids"]
        size = max(map(len, ids))
        return {"input_ids": torch.tensor([value + [1] * (size - len(value)) for value in ids]),
                "attention_mask": torch.tensor([[1] * len(value) + [0] * (size - len(value)) for value in ids])}


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.decoder = Mock()
        self.decoder.config = SimpleNamespace(
            encoder_model_id=ENCODER_MODEL_ID, encoder_revision=ENCODER_REVISION,
            tokenizer_revision=ENCODER_REVISION, bos_token_id=0, eos_token_id=2,
        )
        self.decoder.predict_token_ids.side_effect = lambda states, **kwargs: states[:, 0].to(torch.int64).tolist()
        self.decoder.tokenizer.decode.side_effect = lambda ids, **kwargs: " ".join(str(i) for i in ids if i not in (0, 1, 2))
        self.load_patch = patch("aparecium.api.MatrixDecoder.load", return_value=self.decoder)
        self.download_patch = patch("aparecium.api.snapshot_download", return_value="/model")
        self.load = self.load_patch.start()
        self.download = self.download_patch.start()
        self.addCleanup(self.load_patch.stop)
        self.addCleanup(self.download_patch.stop)
        self.api = Aparecium()

    def test_default_revision_and_bundle_download_are_pinned(self):
        self.download.assert_called_once_with(repo_id=DEFAULT_MODEL_ID, revision=DEFAULT_MODEL_REVISION,
                                              allow_patterns=list(MODEL_FILES), local_files_only=False)
        self.load.assert_called_once_with("/model", device=torch.device("cpu"))
        self.assertEqual(str(self.api.device), "cpu")
        self.assertEqual([field.name for field in fields(InvertResult)], ["text", "token_ids", "input_rows"])
        self.assertFalse(hasattr(self.api, "invert_embedding"))
        self.assertFalse(hasattr(self.api, "invert_embeddings"))

    def test_local_bundle_and_custom_revision_have_no_download(self):
        self.download.reset_mock()
        api = Aparecium(model_path="/local", revision="a" * 40, local_files_only=True, token_batch_size=7)
        self.download.assert_not_called()
        self.assertEqual(api.revision, "a" * 40)
        self.assertEqual(api.token_batch_size, 7)
        api.invert_matrix(matrix())
        self.assertEqual(self.decoder.predict_token_ids.call_args.kwargs, {"token_batch_size": 7})

    def test_custom_repository_without_revision_uses_its_default_branch(self):
        self.download.reset_mock()
        api = Aparecium(model_id="example/matrix")
        self.assertIsNone(api.revision)
        self.download.assert_called_once_with(repo_id="example/matrix", revision=None,
                                              allow_patterns=list(MODEL_FILES), local_files_only=False)

    def test_explicit_revision_overrides_default_pin_for_each_repository(self):
        for model_id in (DEFAULT_MODEL_ID, "example/matrix"):
            self.download.reset_mock()
            api = Aparecium(model_id=model_id, revision="a" * 40)
            self.assertEqual(api.revision, "a" * 40)
            self.download.assert_called_once_with(repo_id=model_id, revision="a" * 40,
                                                  allow_patterns=list(MODEL_FILES), local_files_only=False)

    def test_matrix_decoding_drops_only_framing_preserves_special_ids_and_uses_no_encoder(self):
        with patch("transformers.AutoModel.from_pretrained", side_effect=AssertionError("encoder loaded")), patch(
            "transformers.AutoTokenizer.from_pretrained", side_effect=AssertionError("encoder tokenizer loaded")
        ):
            result = self.api.invert_matrix(matrix((99, 7, 2, 8, 99)))
        self.assertEqual(result, InvertResult(text="7 8", token_ids=[7, 2, 8], input_rows=5))
        self.assertEqual(self.decoder.predict_token_ids.call_args.args[0][:, 0].tolist(), [7, 2, 8])
        self.decoder.tokenizer.decode.assert_called_once_with([7, 2, 8], skip_special_tokens=True,
                                                              clean_up_tokenization_spaces=False)
        self.assertIsNone(self.api._encoder)

    def test_right_padding_and_varied_length_batch_preserve_order(self):
        results = self.api.invert_matrices(
            [matrix((0, 4, 2, 99, 99)), matrix((0, 5, 6, 7, 2)), matrix((0, 8, 2))],
            [[1, 1, 1, 0, 0], [True] * 5, [1.0] * 3],
        )
        self.assertEqual([result.token_ids for result in results], [[4], [5, 6, 7], [8]])
        self.assertEqual([result.input_rows for result in results], [3, 5, 3])
        self.assertEqual([len(call.args[0]) for call in self.decoder.predict_token_ids.call_args_list], [1, 3, 1])
        self.assertEqual(len(self.api.invert_matrices(np.stack([matrix(), matrix()]))), 2)

    def test_float32_conversion_detachment_and_no_normalization(self):
        values = matrix().astype(np.float64) * 3.25
        tensor, rows = _validate_matrix(values)
        self.assertEqual(tensor.dtype, torch.float32)
        np.testing.assert_array_equal(tensor.numpy(), values.astype(np.float32))
        self.assertEqual(rows, 4)
        value = torch.tensor(matrix(), requires_grad=True)
        self.api.invert_matrix(value)
        self.assertFalse(self.decoder.predict_token_ids.call_args.args[0].requires_grad)

    def test_invalid_matrix_values_fail_before_inference(self):
        invalid = [np.zeros(768), np.ones((1, 768)), np.ones((2, 768)), np.ones((257, 768)),
                   np.ones((3, 767)), np.ones((1, 3, 768)), [[1], [2, 3]], None,
                   np.full((3, 768), np.nan), np.full((3, 768), np.inf), np.full((3, 768), 1e300),
                   np.ones((3, 768), dtype=bool), np.ones((3, 768), dtype=complex),
                   torch.ones(3, 768, dtype=torch.bool), torch.ones(3, 768, dtype=torch.complex64),
                   [[True] + [0] * 767] * 3, [["1"] + [0] * 767] * 3]
        for value in invalid:
            with self.subTest(type=type(value), shape=getattr(value, "shape", None)), self.assertRaises(ValueError):
                self.api.invert_matrix(value)
        self.decoder.predict_token_ids.assert_not_called()

    def test_invalid_masks_fail_before_inference_including_padding_nonfinite_states(self):
        for mask in ([1, 1, 1], [1, 0, 1, 0], [0, 1, 1, 1], [1, 1, 0, 0], [2] * 4,
                     ["1"] * 4, [1, 1, 1, float("nan")], [[1] * 4]):
            with self.subTest(mask=mask), self.assertRaises(ValueError):
                self.api.invert_matrix(matrix(), mask)
        values = matrix()
        values[-1] = np.nan
        with self.assertRaises(ValueError):
            self.api.invert_matrix(values, [1, 1, 1, 0])
        self.decoder.predict_token_ids.assert_not_called()

    def test_entire_batch_validates_before_any_forward(self):
        for values, masks in (([matrix(), np.ones(768)], None), ([matrix()] * 33, None),
                              (matrix(), None), ([matrix()], []), ([matrix()], [[1] * 3]),
                              ("matrix", None), ([matrix()], "mask"), (np.empty((0, 3, 767)), None)):
            with self.subTest(type=type(values)), self.assertRaises(ValueError):
                self.api.invert_matrices(values, masks)
        self.decoder.predict_token_ids.assert_not_called()

    def test_empty_batches_do_not_run_encoder_or_decoder(self):
        self.assertEqual(self.api.invert_matrices([]), [])
        self.assertEqual(self.api.invert_matrices(np.empty((0, 3, 768))), [])
        self.assertEqual(self.api.invert_texts([]), [])
        self.decoder.predict_token_ids.assert_not_called()
        self.assertIsNone(self.api._encoder)

    def test_constructor_options_fail_before_download_or_load(self):
        self.download.reset_mock()
        self.load.reset_mock()
        for kwargs in ({"device": "cuda"}, {"device": "cpu:0"}, {"device": "invalid"},
                       {"token_batch_size": 0}, {"token_batch_size": 257}, {"token_batch_size": True},
                       {"model_id": ""}, {"revision": ""}, {"local_files_only": 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Aparecium(**kwargs)
        self.download.assert_not_called()
        self.load.assert_not_called()

    def test_decoder_cardinality_error_is_explicit(self):
        self.decoder.predict_token_ids.side_effect = None
        self.decoder.predict_token_ids.return_value = []
        with self.assertRaisesRegex(RuntimeError, "unexpected number"):
            self.api.invert_matrix(matrix())

    def install_encoder(self):
        tokenizer = EncoderTokenizer()
        encoder = Mock()
        encoder.side_effect = lambda **inputs: SimpleNamespace(
            last_hidden_state=inputs["input_ids"].unsqueeze(-1).expand(-1, -1, 768).to(torch.float32)
        )
        self.api._encoder = encoder
        self.api._encoder_tokenizer = tokenizer
        return tokenizer, encoder

    def test_text_convenience_uses_native_unpooled_states_and_right_padding(self):
        tokenizer, encoder = self.install_encoder()
        result = self.api.invert_texts(["a b c", "a", "a b"], batch_size=2)
        self.assertEqual([item.token_ids for item in result], [[3, 4, 5], [3], [3, 4]])
        self.assertEqual([item.input_rows for item in result], [5, 3, 4])
        self.assertEqual([call.kwargs["input_ids"].shape[0] for call in encoder.call_args_list], [2, 1])
        self.assertEqual(tokenizer.calls[0][1], {"add_special_tokens": True, "padding": False, "truncation": False})
        self.assertEqual(self.api.invert_text("a b").token_ids, [3, 4])

    def test_pinned_encoder_load_is_lazy_safe_and_cached(self):
        encoder = Mock()
        encoder.to.return_value = encoder
        encoder.eval.return_value = encoder
        encoder.requires_grad_.return_value = encoder
        with patch("transformers.AutoModel.from_pretrained", return_value=encoder) as load_encoder, patch(
            "transformers.AutoTokenizer.from_pretrained", return_value=EncoderTokenizer()
        ) as load_tokenizer:
            tokenizer, actual = self.api._ensure_encoder()
            self.assertIs(actual, encoder)
            self.assertEqual(self.api._ensure_encoder(), (tokenizer, encoder))
        load_encoder.assert_called_once_with(ENCODER_MODEL_ID, revision=ENCODER_REVISION,
                                             local_files_only=False, trust_remote_code=False,
                                             use_safetensors=True, dtype=torch.float32)
        load_tokenizer.assert_called_once_with(ENCODER_MODEL_ID, revision=ENCODER_REVISION,
                                               local_files_only=False, trust_remote_code=False, use_fast=True)
        encoder.requires_grad_.assert_called_once_with(False)

    def test_overlong_text_rejected_before_encoder_forward_without_truncation(self):
        tokenizer, encoder = self.install_encoder()
        tokenizer.long = True
        with self.assertRaisesRegex(ValueError, "3..256"):
            self.api.invert_text("a")
        encoder.assert_not_called()
        self.decoder.predict_token_ids.assert_not_called()

    def test_invalid_text_batches_fail_before_encoder_forward(self):
        _, encoder = self.install_encoder()
        for values in ("text", [""], ["   "], [None], [1], ["a"] * 33):
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.api.invert_texts(values)
        for size in (0, True, 33, 1.5):
            with self.subTest(size=size), self.assertRaises(ValueError):
                self.api.invert_texts(["a"], batch_size=size)
        encoder.assert_not_called()

    def test_bad_encoder_contract_does_not_reach_decoder(self):
        _, encoder = self.install_encoder()
        encoder.side_effect = lambda **inputs: SimpleNamespace(last_hidden_state=torch.zeros(1, 3, 768, dtype=torch.float16))
        with self.assertRaisesRegex(RuntimeError, "float32"):
            self.api.invert_text("a")
        self.decoder.predict_token_ids.assert_not_called()

    def test_calls_on_one_instance_are_serialized(self):
        state = {"active": 0, "peak": 0}
        state_lock = threading.Lock()

        def predict(values, **kwargs):
            with state_lock:
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
            time.sleep(0.02)
            with state_lock:
                state["active"] -= 1
            return [3] * len(values)

        self.decoder.predict_token_ids.side_effect = predict
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            results = list(executor.map(lambda _: self.api.invert_matrix(matrix()), range(3)))
        self.assertEqual(len(results), 3)
        self.assertEqual(state["peak"], 1)


if __name__ == "__main__":
    unittest.main()
