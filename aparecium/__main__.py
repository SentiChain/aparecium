"""Reconstruct text from MPNet token matrices or an explicit text round trip."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import io
import json
from pathlib import Path
import sys
from typing import Sequence

import numpy as np

from .api import DEFAULT_MODEL_ID, _validate_matrices, _validate_matrix


MAX_INPUT_BYTES = 16 * 1024 * 1024
MAX_NPY_HEADER_BYTES = 10_000


def _positive_integer(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be an integer between 1 and 256") from None
    if not 1 <= number <= 256:
        raise argparse.ArgumentTypeError("must be an integer between 1 and 256")
    return number


def _reject_constant(value: str) -> None:
    raise ValueError("JSON numbers must be finite")


def _validate_npy_header(contents: bytes, *, batch: bool) -> None:
    stream = io.BytesIO(contents)
    version = np.lib.format.read_magic(stream)
    readers = {
        (1, 0): (2, np.lib.format.read_array_header_1_0),
        (2, 0): (4, np.lib.format.read_array_header_2_0),
    }
    if version not in readers:
        raise ValueError("unsupported NPY format")
    length_bytes, reader = readers[version]
    length_position = stream.tell()
    encoded_length = stream.read(length_bytes)
    if len(encoded_length) != length_bytes:
        raise ValueError("incomplete NPY header")
    header_length = int.from_bytes(encoded_length, byteorder="little")
    if not 1 <= header_length <= MAX_NPY_HEADER_BYTES:
        raise ValueError("NPY header exceeds its size limit")
    payload_position = stream.tell() + header_length
    if payload_position > len(contents):
        raise ValueError("incomplete NPY header")
    stream.seek(length_position)
    shape, _, dtype = reader(stream, max_header_size=MAX_NPY_HEADER_BYTES)
    if stream.tell() != payload_position or any(type(dimension) is not int for dimension in shape):
        raise ValueError("invalid NPY header")
    if len(shape) != (3 if batch else 2):
        raise ValueError("invalid NPY dimensions")
    if shape[-1] != 768 or not 3 <= shape[-2] <= 256:
        raise ValueError("invalid NPY matrix shape")
    if batch and not 1 <= shape[0] <= 32:
        raise ValueError("invalid NPY batch size")
    if dtype.hasobject or dtype.fields is not None or dtype.subdtype is not None or dtype.kind not in "iuf":
        raise ValueError("NPY states must use a real numerical dtype")
    elements = shape[-2] * shape[-1] * (shape[0] if batch else 1)
    if dtype.itemsize < 1 or payload_position + elements * dtype.itemsize != len(contents):
        raise ValueError("NPY payload does not match its declared shape and dtype")


def _matrix_file(path: str, *, batch: bool) -> tuple[object, object | None]:
    source = Path(path).expanduser()
    if source.suffix.lower() not in {".npy", ".json"}:
        raise ValueError("matrix files must use the .npy or .json extension")
    try:
        with source.open("rb") as stream:
            contents = stream.read(MAX_INPUT_BYTES + 1)
    except OSError:
        raise ValueError("unable to read the matrix file") from None
    if len(contents) > MAX_INPUT_BYTES:
        raise ValueError("matrix files must not exceed 16 MiB")
    try:
        if source.suffix.lower() == ".npy":
            _validate_npy_header(contents, batch=batch)
            values = np.load(io.BytesIO(contents), allow_pickle=False)
            if not isinstance(values, np.ndarray):
                values.close()
                raise ValueError
            if values.ndim != (3 if batch else 2):
                raise ValueError
            masks = None
        else:
            values = json.loads(contents.decode("utf-8"), parse_constant=_reject_constant)
            masks = None
            if isinstance(values, dict):
                key, mask_key = ("matrices", "attention_masks") if batch else ("matrix", "attention_mask")
                if key not in values or set(values) - {key, mask_key}:
                    raise ValueError
                masks = values.get(mask_key)
                values = values[key]
        if batch:
            if not len(values):
                raise ValueError
            _validate_matrices(values, masks)
        else:
            _validate_matrix(values, masks)
    except (ValueError, TypeError, OverflowError, UnicodeError, RecursionError, EOFError):
        contract = "a nonempty batch of up to 32 matrices" if batch else "one matrix"
        raise ValueError(
            f"matrix file must contain {contract} with 3 to 256 rows, 768 finite real values per row, "
            "and any mask must be an aligned binary prefix with at least 3 visible rows"
        ) from None
    return values, masks


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reconstruct text from an MPNet token matrix, or round-trip explicit text."
    )
    parser.add_argument("text", nargs="?", help="text to round-trip (default: standard input)")
    files = parser.add_mutually_exclusive_group()
    files.add_argument("--matrix-file", metavar="PATH", help="one token matrix in a .npy or .json file")
    files.add_argument("--matrices-file", metavar="PATH", help="a batch of token matrices in a .npy or .json file")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Hugging Face model repository")
    parser.add_argument("--revision", help="model repository revision")
    parser.add_argument("--model-path", help="local model directory")
    parser.add_argument("--device", choices=("cpu", "mps"), help="execution device (default: cpu)")
    parser.add_argument("--local-files-only", action="store_true", help="use only locally available model files")
    parser.add_argument("--token-batch-size", type=_positive_integer, default=256,
                        help="token rows decoded per batch, 1 to 256 (default: 256)")
    parser.add_argument("--json", action="store_true", help="print result fields as JSON")
    args = parser.parse_args(argv)
    if args.text is not None and (args.matrix_file or args.matrices_file):
        parser.error("provide exactly one of text, --matrix-file, or --matrices-file")

    values = masks = None
    text = args.text
    if args.matrix_file or args.matrices_file:
        try:
            values, masks = _matrix_file(args.matrices_file or args.matrix_file, batch=bool(args.matrices_file))
        except ValueError as exc:
            parser.error(str(exc))
    else:
        if text is None:
            if sys.stdin.isatty():
                parser.error("provide text, a matrix file, or text through standard input")
            text = sys.stdin.read(MAX_INPUT_BYTES + 1)
            if len(text.encode("utf-8")) > MAX_INPUT_BYTES:
                parser.error("standard input must not exceed 16 MiB")
            text = text.strip()
        if not text.strip():
            parser.error("text must not be empty")

    try:
        from .api import Aparecium

        model = Aparecium(
            model_id=args.model_id,
            revision=args.revision,
            model_path=args.model_path,
            device=args.device,
            local_files_only=args.local_files_only,
            token_batch_size=args.token_batch_size,
        )
        if args.matrices_file:
            results = model.invert_matrices(values, attention_masks=masks)
            output = json.dumps({"results": [asdict(result) for result in results]}, ensure_ascii=False) if args.json else "\n".join(result.text for result in results)
        else:
            result = model.invert_matrix(values, attention_mask=masks) if args.matrix_file else model.invert_text(text)
            output = json.dumps(asdict(result), ensure_ascii=False) if args.json else result.text
        print(output)
    except Exception:
        print("aparecium: reconstruction is unavailable", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
