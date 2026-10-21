"""Export a verified token-matrix checkpoint to a portable model directory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer
from tokenizers import AddedToken


TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def tensor_digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def special_token_value(value):
    if isinstance(value, AddedToken):
        return value.__getstate__()
    if isinstance(value, list):
        return [special_token_value(item) for item in value]
    return value


def export(checkpoint: Path, manifest: Path, manifest_sha256: str, output: Path) -> dict:
    if output.exists():
        raise FileExistsError("output directory already exists")
    if digest(manifest) != manifest_sha256:
        raise ValueError("manifest checksum does not match")
    specification = json.loads(manifest.read_text(encoding="utf-8"))
    expected = specification["checkpoint"]["sha256"]
    if digest(checkpoint) != expected:
        raise ValueError("checkpoint checksum does not match")
    if checkpoint.stat().st_size > 128 * 1024**2:
        raise ValueError("checkpoint exceeds the supported size")
    torch.set_num_threads(4)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (saved.get("schema_version") != 1 or saved.get("model_type") != "matrix_token_nonlinear"
            or saved.get("architecture") != "dense_gelu_layernorm_frozen_original_vocabulary_plus_bias"
            or saved.get("config") != specification["model_config"]):
        raise ValueError("checkpoint configuration does not match")
    tensors = saved["state_dict"]
    inventory = specification["state_dict_tensor_inventory"]
    if set(tensors) != set(inventory):
        raise ValueError("checkpoint tensor inventory does not match")
    for name, tensor in tensors.items():
        item = inventory[name]
        if (tensor.dtype != torch.float32 or list(tensor.shape) != item["shape"]
                or not bool(torch.isfinite(tensor).all()) or tensor_digest(tensor) != item["sha256"]):
            raise ValueError(f"invalid checkpoint tensor: {name}")
    config = saved["config"]
    tokenizer = AutoTokenizer.from_pretrained(config["encoder_model_id"],
        revision=config["tokenizer_revision"], local_files_only=True, use_fast=True,
        trust_remote_code=False)
    if not tokenizer.is_fast or len(tokenizer) != config["vocabulary_size"]:
        raise ValueError("tokenizer vocabulary does not match")
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free - 256 * 1024**2 < 40 * 1024**3:
        raise OSError("export would breach the free-disk reserve")
    temporary = Path(tempfile.mkdtemp(prefix=".model-export-", dir=output.parent))
    try:
        tokenizer.save_pretrained(temporary)
        vocabulary = tokenizer.get_vocab()
        if set(vocabulary.values()) != set(range(len(tokenizer))):
            raise ValueError("tokenizer vocabulary IDs must be dense and unique")
        ordered = sorted(vocabulary, key=vocabulary.__getitem__)
        if any("\n" in token or "\r" in token for token in ordered):
            raise ValueError("vocabulary tokens cannot contain line delimiters")
        (temporary / "vocab.txt").write_text("\n".join(ordered) + "\n", encoding="utf-8")
        mapping = {}
        for name, value in tokenizer.special_tokens_map.items():
            if not isinstance(value, str):
                raise ValueError("special token entries must identify a single token")
            index = tokenizer.convert_tokens_to_ids(value)
            descriptor = tokenizer.added_tokens_decoder[index]
            if str(descriptor) != value:
                raise ValueError("special token metadata does not match its vocabulary ID")
            mapping[name] = special_token_value(descriptor)
        (temporary / "special_tokens_map.json").write_text(
            json.dumps(mapping, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        actual = {path.name for path in temporary.iterdir()}
        if actual != set(TOKENIZER_FILES):
            raise ValueError(f"unexpected tokenizer file inventory: {sorted(actual)}")
        restored_tokenizer = AutoTokenizer.from_pretrained(temporary, local_files_only=True,
            use_fast=True, trust_remote_code=False)
        probes = ["BTC 2,000 versus 2,000,000", "$250,000 by 2030", "<mask>",
            "Ethereum's price", "Gold \u00a34,200", " \u0009hello\nworld", "\U0001f680 BTC"]
        if (restored_tokenizer.get_vocab() != vocabulary
                or any(restored_tokenizer.encode(value) != tokenizer.encode(value) for value in probes)):
            raise ValueError("export changed tokenizer behavior")
        tokenizer_hashes = {name: digest(temporary / name) for name in TOKENIZER_FILES}
        weights = temporary / "model.safetensors"
        save_file({name: tensor.detach().cpu().contiguous() for name, tensor in tensors.items()},
            str(weights), metadata={"format": "pt", "source_checkpoint_sha256": expected})
        restored = load_file(str(weights), device="cpu")
        if set(restored) != set(tensors) or any(
                tensor_digest(restored[name]) != inventory[name]["sha256"] for name in tensors):
            raise ValueError("export changed tensor content")
        result = {"schema_version": 1, "model_type": "aparecium_matrix",
            "architecture": "dense_gelu_layernorm_frozen_vocabulary", **config,
            "max_sequence_length": 256, "layer_norm_eps": 1e-5,
            "weights_file": "model.safetensors", "weights_sha256": digest(weights),
            "source_checkpoint_sha256": expected,
            "vocabulary_array_sha256": saved["vocabulary_array_sha256"],
            "tensor_sha256": {name: item["sha256"] for name, item in inventory.items()},
            "tokenizer_files_sha256": tokenizer_hashes}
        (temporary / "config.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        if digest(checkpoint) != expected or digest(manifest) != manifest_sha256:
            raise ValueError("source files changed during export")
        temporary.rename(output)
        return result
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = export(arguments.checkpoint, arguments.manifest, arguments.manifest_sha256, arguments.output)
    print(json.dumps({"weights_sha256": result["weights_sha256"],
        "source_checkpoint_sha256": result["source_checkpoint_sha256"],
        "tensor_count": len(result["tensor_sha256"])}))


if __name__ == "__main__":
    main()
