"""Unit tests for model configuration, decoding, and checkpoint loading."""

import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from unittest.mock import Mock, patch

import torch
from safetensors.torch import save_file
from torch import nn

from aparecium.config import ModelConfig
from aparecium.model import PrefixDecoder, PrefixMapper, _seeded_sampling, is_fluent


class Tokenizer:
    eos_token_id = 0

    def decode(self, tokens, skip_special_tokens):
        assert skip_special_tokens
        return " " + " ".join(str(token) for token in tokens) + " "


class LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(100, 6)
        self.calls = []

    def get_input_embeddings(self):
        return self.embedding

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        batch_size = kwargs["inputs_embeds"].shape[0]
        if kwargs["do_sample"]:
            return torch.randint(1, 90, (batch_size * kwargs["num_return_sequences"], 4))
        return torch.tensor([[1, 2, 0, 7]] * batch_size)


def small_config():
    return replace(ModelConfig(), prefix_length=2, mapper_hidden=4)


class ConfigTests(unittest.TestCase):
    def test_complete_configuration_round_trip_and_immutability(self):
        config = ModelConfig()
        self.assertEqual(ModelConfig.from_dict(config.to_dict()), config)
        with self.assertRaises(FrozenInstanceError):
            config.embedding_dim = 12
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
            self.assertEqual(ModelConfig.load(path), config)

    def test_missing_or_unknown_fields_are_rejected(self):
        values = ModelConfig().to_dict()
        values.pop("encoder_revision")
        values["typo"] = 1
        with self.assertRaisesRegex(ValueError, "missing fields: encoder_revision; unknown fields: typo"):
            ModelConfig.from_dict(values)
        with self.assertRaisesRegex(ValueError, "JSON object"):
            ModelConfig.from_dict([])

    def test_unsupported_or_invalid_values_are_rejected(self):
        invalid = [
            {"architecture": "different"},
            {"base_model": "other/model"},
            {"encoder_model": "other/encoder"},
            {"base_revision": "main"},
            {"encoder_revision": None},
            {"embedding_dim": 384},
            {"prefix_length": 0},
            {"mapper_hidden": True},
            {"lora_rank": 769},
            {"max_length": 1024},
        ]
        for changed in invalid:
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                replace(ModelConfig(), **changed)


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.lm = LanguageModel()
        self.model = PrefixDecoder(small_config(), self.lm, Tokenizer()).eval()
        self.vectors = torch.ones(2, 768)

    def test_mapper_shape_and_weight_names(self):
        mapper = PrefixMapper(768, 4, 2, 6)
        output = mapper(self.vectors)
        self.assertEqual(output.shape, (2, 2, 6))
        self.assertEqual(set(mapper.state_dict()), {"net.0.weight", "net.0.bias", "net.2.weight", "net.2.bias"})
        expected = mapper.net(self.vectors).reshape(2, 2, 6)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)

    def test_greedy_and_sampled_candidate_order_and_eos_handling(self):
        greedy = torch.tensor([[1, 2, 0, 99], [3, 4, 5, 6]])
        sampled = torch.tensor([[11, 0, 8, 8], [12, 13, 0, 8], [21, 22, 23, 24], [0, 31, 32, 33]])
        with patch.object(self.lm, "generate", side_effect=[greedy, sampled]) as generate:
            results = self.model.generate(self.vectors, num_candidates=3)
        self.assertEqual(
            results,
            [
                {"candidates": [
                    {"text": "1 2", "stopped": True},
                    {"text": "11", "stopped": True},
                    {"text": "12 13", "stopped": True},
                ]},
                {"candidates": [
                    {"text": "3 4 5 6", "stopped": False},
                    {"text": "21 22 23 24", "stopped": False},
                    {"text": "", "stopped": True},
                ]},
            ],
        )
        first = generate.call_args_list[0].kwargs
        second = generate.call_args_list[1].kwargs
        self.assertFalse(first["do_sample"])
        self.assertTrue(second["do_sample"])
        self.assertEqual(first["max_new_tokens"], 64)
        self.assertEqual(second["num_return_sequences"], 2)
        self.assertEqual(second["temperature"], 0.8)
        self.assertEqual(second["top_p"], 0.9)
        self.assertEqual(first["inputs_embeds"].shape, (2, 2, 6))
        torch.testing.assert_close(first["attention_mask"], torch.ones(2, 2, dtype=torch.long))

    def test_default_candidate_count_and_rng_restoration(self):
        before = torch.random.get_rng_state().clone()
        first = self.model.generate(self.vectors, seed=42)
        torch.testing.assert_close(torch.random.get_rng_state(), before, rtol=0, atol=0)
        second = self.model.generate(self.vectors, seed=42)
        different = self.model.generate(self.vectors, seed=43)
        self.assertEqual(first, second)
        self.assertNotEqual(first, different)
        self.assertEqual([len(item["candidates"]) for item in first], [5, 5])
        self.assertEqual(first[0]["candidates"][0], different[0]["candidates"][0])
        with _seeded_sampling(42, torch.device("cpu")):
            expected = torch.randint(1, 90, (8, 4))
        self.assertEqual(first[0]["candidates"][1]["text"], " ".join(map(str, expected[0].tolist())))

    def test_rng_restored_after_sampling_failure(self):
        before = torch.random.get_rng_state().clone()

        def fail(**kwargs):
            if kwargs["do_sample"]:
                torch.rand(20)
                raise RuntimeError("sampling failed")
            return torch.tensor([[1, 0], [2, 0]])

        with patch.object(self.lm, "generate", side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, "sampling failed"):
                self.model.generate(self.vectors)
        torch.testing.assert_close(torch.random.get_rng_state(), before, rtol=0, atol=0)

    def test_single_candidate_and_empty_batch_do_not_sample(self):
        results = self.model.generate(self.vectors, num_candidates=1, max_length=8)
        self.assertEqual(len(self.lm.calls), 1)
        self.assertEqual(self.lm.calls[0]["max_new_tokens"], 8)
        self.assertEqual([len(item["candidates"]) for item in results], [1, 1])
        self.assertEqual(self.model.generate(torch.empty(0, 768)), [])
        self.assertEqual(len(self.lm.calls), 1)

    def test_invalid_controls_are_rejected_before_generation(self):
        invalid = [
            {"num_candidates": 0}, {"num_candidates": 33}, {"num_candidates": True},
            {"max_length": 0}, {"max_length": 65}, {"max_length": 2.0},
            {"seed": -1}, {"seed": True}, {"seed": 2**63},
            {"temperature": 0}, {"temperature": float("nan")}, {"temperature": True},
            {"top_p": 0}, {"top_p": 1.01}, {"top_p": float("inf")},
        ]
        for changed in invalid:
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                self.model.generate(self.vectors, **changed)
        self.assertEqual(self.lm.calls, [])

    def test_invalid_vectors_are_rejected_before_generation(self):
        invalid = [
            [[0.0] * 768], torch.ones(768), torch.ones(2, 767),
            torch.ones(2, 768, dtype=torch.long), torch.full((1, 768), float("nan")),
            torch.full((1, 768), 1e300, dtype=torch.float64),
        ]
        for vectors in invalid:
            with self.subTest(vectors_type=type(vectors)), self.assertRaises(ValueError):
                self.model.generate(vectors)
        self.assertEqual(self.lm.calls, [])

    def test_fluency_filter(self):
        self.assertTrue(is_fluent("Heavy rain closed several roads."))
        self.assertFalse(is_fluent("Heavy rain"))
        self.assertFalse(is_fluent("the market rose the market rose the market rose"))


class LoadingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)
        self.config = small_config()
        (self.path / "config.json").write_text(json.dumps(self.config.to_dict()), encoding="utf-8")
        (self.path / "adapter").mkdir()
        adapter = {
            "peft_type": "LORA", "r": self.config.lora_rank,
            "base_model_name_or_path": self.config.base_model, "revision": self.config.base_revision,
        }
        (self.path / "adapter" / "adapter_config.json").write_text(json.dumps(adapter), encoding="utf-8")
        save_file({"weight": torch.zeros(1)}, str(self.path / "adapter" / "adapter_model.safetensors"))
        self.lm = LanguageModel()
        mapper = PrefixMapper(768, self.config.mapper_hidden, self.config.prefix_length, 6)
        self.weights = mapper.state_dict()
        save_file(self.weights, str(self.path / "mapper.safetensors"))

    def test_load_pins_dependencies_and_uses_safe_weights(self):
        with (
            patch("transformers.AutoTokenizer.from_pretrained", return_value=Tokenizer()) as tokenizer,
            patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=self.lm) as base,
            patch("peft.PeftModel.from_pretrained", return_value=self.lm) as adapter,
            patch("torch.load", side_effect=AssertionError("Unsafe tensor loading")),
        ):
            model = PrefixDecoder.load(self.path, local_files_only=True)
        tokenizer.assert_called_once_with(
            self.config.base_model, revision=self.config.base_revision, local_files_only=True
        )
        base.assert_called_once_with(
            self.config.base_model, revision=self.config.base_revision, dtype=torch.float32,
            use_safetensors=True, local_files_only=True,
        )
        adapter.assert_called_once_with(self.lm, self.path / "adapter", is_trainable=False, local_files_only=True)
        self.assertFalse(model.training)
        self.assertFalse(model.lm.training)
        for name, tensor in model.mapper.state_dict().items():
            torch.testing.assert_close(tensor, self.weights[name], rtol=0, atol=0)

    def test_missing_safe_adapter_weights_are_rejected_before_download(self):
        (self.path / "adapter" / "adapter_model.safetensors").unlink()
        with patch("transformers.AutoModelForCausalLM.from_pretrained") as base:
            with self.assertRaisesRegex(FileNotFoundError, "adapter_model.safetensors"):
                PrefixDecoder.load(self.path)
        base.assert_not_called()

    def test_mismatched_adapter_is_rejected_before_download(self):
        path = self.path / "adapter" / "adapter_config.json"
        values = json.loads(path.read_text(encoding="utf-8"))
        values["revision"] = "main"
        path.write_text(json.dumps(values), encoding="utf-8")
        with patch("transformers.AutoModelForCausalLM.from_pretrained") as base:
            with self.assertRaisesRegex(ValueError, "Adapter configuration"):
                PrefixDecoder.load(self.path)
        base.assert_not_called()

    def test_mapper_weights_are_loaded_strictly(self):
        save_file({"net.0.bias": self.weights["net.0.bias"]}, str(self.path / "mapper.safetensors"))
        with (
            patch("transformers.AutoTokenizer.from_pretrained", return_value=Tokenizer()),
            patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=self.lm),
            patch("peft.PeftModel.from_pretrained", return_value=self.lm),
        ):
            with self.assertRaisesRegex(RuntimeError, "Missing key"):
                PrefixDecoder.load(self.path)


class AcceleratorRngTests(unittest.TestCase):
    def test_mps_state_is_restored_when_sampling_raises(self):
        cpu_before = torch.random.get_rng_state().clone()
        accelerator_before = torch.tensor([1, 2, 3], dtype=torch.uint8)
        with (
            patch("torch.mps.get_rng_state", return_value=accelerator_before),
            patch("torch.mps.manual_seed") as seed,
            patch("torch.mps.set_rng_state") as restore,
        ):
            with self.assertRaisesRegex(RuntimeError, "sampling failed"):
                with _seeded_sampling(27, torch.device("mps")):
                    torch.rand(10)
                    raise RuntimeError("sampling failed")
        seed.assert_called_once_with(27)
        restore.assert_called_once_with(accelerator_before)
        torch.testing.assert_close(torch.random.get_rng_state(), cpu_before, rtol=0, atol=0)

    def test_cuda_seeds_and_restores_only_the_selected_device(self):
        cpu_before = torch.random.get_rng_state().clone()
        accelerator_before = torch.tensor([4, 5, 6], dtype=torch.uint8)
        generators = (Mock(), Mock())
        with (
            patch("torch.cuda.get_rng_state", return_value=accelerator_before) as snapshot,
            patch("torch.cuda.default_generators", generators),
            patch("torch.cuda.set_rng_state") as restore,
        ):
            with _seeded_sampling(28, torch.device("cuda:1")):
                torch.rand(10)
        snapshot.assert_called_once_with(1)
        generators[0].manual_seed.assert_not_called()
        generators[1].manual_seed.assert_called_once_with(28)
        restore.assert_called_once_with(accelerator_before, 1)
        torch.testing.assert_close(torch.random.get_rng_state(), cpu_before, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
