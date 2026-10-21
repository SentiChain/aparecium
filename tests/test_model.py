"""Configuration integrity, portable loading, and decoder mathematics."""

import hashlib
import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from unittest.mock import Mock, patch

import torch
from safetensors.torch import save_file
from torch.nn import functional as F

from aparecium.config import ModelConfig, TENSOR_SHAPES, TOKENIZER_FILES
from aparecium.model import MatrixDecoder, _get_device


_PREVIOUS_THREADS = None


def setUpModule():
    global _PREVIOUS_THREADS
    _PREVIOUS_THREADS = torch.get_num_threads()
    torch.set_num_threads(4)


def tearDownModule():
    torch.set_num_threads(_PREVIOUS_THREADS)


def configuration():
    return ModelConfig(
        weights_sha256="a" * 64,
        source_checkpoint_sha256="b" * 64,
        vocabulary_array_sha256="c" * 64,
        tensor_sha256={name: "c" * 64 for name in TENSOR_SHAPES},
        tokenizer_files_sha256={name: "d" * 64 for name in TOKENIZER_FILES},
    )


def tensors():
    values = {name: torch.zeros(shape, dtype=torch.float32) for name, shape in TENSOR_SHAPES.items()}
    values["dense.weight"].copy_(torch.eye(768))
    values["layer_norm.weight"].fill_(1)
    for index in range(8):
        values["vocabulary"][index, index] = 1
    return values


class ConfigTests(unittest.TestCase):
    def test_complete_round_trip_and_immutable_hashes(self):
        config = configuration()
        self.assertEqual(ModelConfig.from_dict(config.to_dict()), config)
        with self.assertRaises(FrozenInstanceError):
            config.state_width = 4
        with self.assertRaises(TypeError):
            config.tensor_sha256["vocabulary"] = "e" * 64
        values = config.to_dict()
        values["tensor_sha256"]["vocabulary"] = "e" * 64
        self.assertEqual(config.tensor_sha256["vocabulary"], "c" * 64)

    def test_missing_unknown_and_nonobject_fields_are_rejected(self):
        values = configuration().to_dict()
        values.pop("encoder_revision")
        values["unknown"] = 1
        with self.assertRaisesRegex(ValueError, "missing fields: encoder_revision; unknown fields: unknown"):
            ModelConfig.from_dict(values)
        with self.assertRaisesRegex(ValueError, "JSON object"):
            ModelConfig.from_dict([])

    def test_architecture_revisions_types_paths_and_hashes_are_strict(self):
        for change in (
            {"schema_version": True}, {"state_width": 768.0}, {"vocabulary_size": 30526},
            {"architecture": "other"}, {"model_type": "other"}, {"max_sequence_length": 512},
            {"encoder_model_id": "other/model"}, {"encoder_revision": "main"},
            {"tokenizer_revision": "main"}, {"pad_token_id": False}, {"layer_norm_eps": 0.001},
            {"weights_file": "../model.safetensors"}, {"weights_sha256": "A" * 64},
            {"source_checkpoint_sha256": "b" * 63}, {"vocabulary_array_sha256": "e" * 64},
            {"tensor_sha256": {}}, {"tokenizer_files_sha256": {}},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(configuration(), **change)

    def test_bounded_json_and_duplicate_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(configuration().to_dict()), encoding="utf-8")
            self.assertEqual(ModelConfig.load(path), configuration())
            path.write_text('{"schema_version": 1, "schema_version": 1}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                ModelConfig.load(path)
            path.write_bytes(b" " * 65537)
            with self.assertRaisesRegex(ValueError, "file size"):
                ModelConfig.load(path)


class DecoderMathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = MatrixDecoder(configuration(), tensors(), Mock())

    def test_exact_dense_gelu_layernorm_projection_and_no_gradients(self):
        rows = torch.zeros(5, 768)
        rows[torch.arange(5), torch.arange(3, 8)] = 3
        expected = F.linear(
            F.layer_norm(F.gelu(F.linear(rows, self.model.dense.weight, self.model.dense.bias)),
                         (768,), self.model.layer_norm.weight, self.model.layer_norm.bias, 1e-5),
            self.model.vocabulary, self.model.vocabulary_bias,
        )
        actual = self.model(rows.requires_grad_())
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(actual.dtype, torch.float32)
        self.assertFalse(actual.requires_grad)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            mixed_context = self.model(rows)
        self.assertEqual(mixed_context.dtype, torch.float32)
        torch.testing.assert_close(mixed_context, expected, rtol=0, atol=0)
        self.assertFalse(self.model.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in self.model.parameters()))
        self.assertEqual(set(self.model.state_dict()), set(TENSOR_SHAPES))
        with self.assertRaisesRegex(ValueError, "inference only"):
            self.model.train()

    def test_row_chunking_keeps_native_argmax_and_sequence_order(self):
        rows = torch.zeros(7, 768)
        rows[torch.arange(7), torch.tensor([3, 4, 5, 6, 7, 3, 4])] = 4
        with patch.object(self.model, "forward", wraps=self.model.forward) as forward:
            actual = self.model.predict_token_ids(rows, token_batch_size=3)
            self.assertEqual([len(call.args[0]) for call in forward.call_args_list], [3, 3, 1])
        self.assertEqual(actual, [3, 4, 5, 6, 7, 3, 4])
        for chunk in (1, 7, 256):
            self.assertEqual(self.model.predict_token_ids(rows, chunk), actual)

    def test_invalid_states_and_chunk_sizes_fail_before_forward(self):
        for values in (torch.ones(768), torch.ones(2, 767), torch.ones(0, 768), torch.ones(257, 768),
                       torch.ones(3, 768, dtype=torch.float64), torch.full((3, 768), float("nan"))):
            with self.subTest(shape=values.shape), self.assertRaises(ValueError):
                self.model.predict_token_ids(values)
        for size in (0, 257, True, 1.5):
            with self.subTest(size=size), self.assertRaises(ValueError):
                self.model.predict_token_ids(torch.ones(3, 768), size)


class LoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        values = tensors()
        save_file(values, str(cls.root / "model.safetensors"))
        digests = {name: hashlib.sha256(value.numpy().tobytes()).hexdigest() for name, value in values.items()}
        token_digests = {}
        for name in TOKENIZER_FILES:
            data = ("tokenizer fixture " + name).encode()
            (cls.root / name).write_bytes(data)
            token_digests[name] = hashlib.sha256(data).hexdigest()
        cls.config = replace(
            configuration(), weights_sha256=hashlib.sha256((cls.root / "model.safetensors").read_bytes()).hexdigest(),
            vocabulary_array_sha256=digests["vocabulary"], tensor_sha256=digests,
            tokenizer_files_sha256=token_digests,
        )
        cls.write_config(cls.config)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    @classmethod
    def write_config(cls, config):
        (cls.root / "config.json").write_text(json.dumps(config.to_dict()), encoding="utf-8")

    def setUp(self):
        self.tokenizer = type("TestTokenizer", (), dict(is_fast=True, pad_token_id=1, bos_token_id=0,
                              eos_token_id=2, unk_token_id=104, __len__=lambda self: 30527))()
        self.patch = patch("transformers.MPNetTokenizer.from_pretrained", return_value=self.tokenizer)
        self.loader = self.patch.start()
        self.addCleanup(self.patch.stop)
        self.addCleanup(self.write_config, self.config)

    def test_safe_complete_local_bundle_loads_without_encoder_or_pickle(self):
        with patch("torch.load", side_effect=AssertionError("pickle load")), patch(
            "transformers.AutoModel.from_pretrained", side_effect=AssertionError("encoder load")
        ):
            model = MatrixDecoder.load(self.root)
        self.assertFalse(model.training)
        self.assertEqual(model.predict_token_ids(torch.eye(768)[:4]), [0, 1, 2, 3])
        self.loader.assert_called_once_with(str(self.root), use_fast=True, local_files_only=True, trust_remote_code=False)

    def test_file_tensor_and_tokenizer_hash_mismatches_are_rejected(self):
        for config in (
            replace(self.config, weights_sha256="0" * 64),
            replace(self.config, tensor_sha256={**self.config.tensor_sha256, "dense.bias": "0" * 64}),
            replace(self.config, tokenizer_files_sha256={**self.config.tokenizer_files_sha256, "vocab.txt": "0" * 64}),
        ):
            self.write_config(config)
            with self.subTest(config=config), self.assertRaisesRegex(ValueError, "checksum mismatch"):
                MatrixDecoder.load(self.root)
        self.loader.assert_not_called()

    def test_missing_or_oversized_tokenizer_files_are_rejected(self):
        path = self.root / "special_tokens_map.json"
        content = path.read_bytes()
        try:
            path.unlink()
            with self.assertRaises(FileNotFoundError):
                MatrixDecoder.load(self.root)
            path.write_bytes(b" " * 65537)
            with self.assertRaisesRegex(ValueError, "file size"):
                MatrixDecoder.load(self.root)
        finally:
            path.write_bytes(content)
        self.loader.assert_not_called()

    def test_tokenizer_special_id_and_fast_contract_are_checked(self):
        for name, value in (("is_fast", False), ("pad_token_id", 0), ("unk_token_id", None)):
            previous = getattr(self.tokenizer, name)
            setattr(self.tokenizer, name, value)
            try:
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, "Tokenizer"):
                    MatrixDecoder.load(self.root)
            finally:
                setattr(self.tokenizer, name, previous)

    def test_tensor_shape_dtype_and_finiteness_are_checked_before_loading_tokenizer(self):
        real_open = __import__("safetensors").safe_open
        for shape, dtype in (((768, 30527), "F32"), ((30527, 768), "F16")):
            view = Mock()
            view.get_shape.return_value = shape
            view.get_dtype.return_value = dtype
            saved = Mock()
            saved.keys.return_value = list(TENSOR_SHAPES)
            saved.get_slice.return_value = view
            context = Mock(__enter__=Mock(return_value=saved), __exit__=Mock(return_value=False))
            with patch("safetensors.safe_open", return_value=context), self.assertRaisesRegex(ValueError, "shape or dtype"):
                MatrixDecoder.load(self.root)
        with real_open(str(self.root / "model.safetensors"), framework="pt", device="cpu") as saved:
            actual = {name: saved.get_tensor(name) for name in TENSOR_SHAPES}
        actual["dense.bias"] = torch.full((768,), float("nan"))
        saved = Mock()
        saved.keys.return_value = list(TENSOR_SHAPES)
        saved.get_slice.side_effect = lambda name: Mock(get_shape=lambda: TENSOR_SHAPES[name], get_dtype=lambda: "F32")
        saved.get_tensor.side_effect = actual.__getitem__
        context = Mock(__enter__=Mock(return_value=saved), __exit__=Mock(return_value=False))
        with patch("safetensors.safe_open", return_value=context), self.assertRaisesRegex(ValueError, "non-finite"):
            MatrixDecoder.load(self.root)
        self.loader.assert_not_called()

    def test_cpu_default_and_explicit_available_mps(self):
        self.assertEqual(str(_get_device(None)), "cpu")
        with patch("torch.backends.mps.is_available", return_value=True):
            self.assertEqual(str(_get_device("mps")), "mps")
        with patch("torch.backends.mps.is_available", return_value=False):
            for device in ("mps", "cuda", "cuda:0", "cpu:0", "mps:0", "meta", "invalid"):
                with self.subTest(device=device), self.assertRaises(ValueError):
                    _get_device(device)


if __name__ == "__main__":
    unittest.main()
