"""Exercise command-line matrix inputs without model loading or network access."""

from __future__ import annotations

from contextlib import ExitStack, redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from aparecium.__main__ import MAX_NPY_HEADER_BYTES, main
from aparecium.api import DEFAULT_MODEL_ID, InvertResult


def matrix(rows: int = 3) -> list[list[float]]:
    return [[float(index + 1)] * 768 for index in range(rows)]


def result(text: str = "The train arrived.", rows: int = 3) -> InvertResult:
    return InvertResult(text=text, token_ids=[17], input_rows=rows)


class TestCLI(unittest.TestCase):
    def setUp(self) -> None:
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.directory = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.constructor = self.stack.enter_context(patch("aparecium.api.Aparecium"))
        self.model = self.constructor.return_value
        self.model.invert_text.return_value = result()
        self.model.invert_matrix.return_value = result()
        self.model.invert_matrices.return_value = [result("First."), result("Second.", 4)]
        self.stdout = self.stack.enter_context(redirect_stdout(io.StringIO()))
        self.stderr = self.stack.enter_context(redirect_stderr(io.StringIO()))

    def json_file(self, payload: object, name: str = "states.json") -> str:
        path = self.directory / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return str(path)

    def numpy_file(self, values: np.ndarray, name: str = "states.npy") -> str:
        path = self.directory / name
        np.save(path, values)
        return str(path)

    def numpy_header(self, shape: tuple[int, ...], dtype: np.dtype | str = "<f4", *,
                     version: tuple[int, int] = (1, 0), fortran_order: bool = False) -> bytes:
        stream = io.BytesIO()
        writer = np.lib.format.write_array_header_1_0 if version == (1, 0) else np.lib.format.write_array_header_2_0
        writer(stream, {"descr": np.lib.format.dtype_to_descr(np.dtype(dtype)),
                        "fortran_order": fortran_order, "shape": shape})
        return stream.getvalue()

    def bytes_file(self, contents: bytes) -> str:
        path = self.directory / "states.npy"
        path.write_bytes(contents)
        return str(path)

    def assert_usage_error(self, arguments: list[str]) -> None:
        with self.assertRaises(SystemExit) as raised:
            main(arguments)
        self.assertEqual(raised.exception.code, 2)
        self.constructor.assert_not_called()

    def test_text_output_and_defaults(self) -> None:
        self.assertEqual(main(["The train arrived."]), 0)
        self.assertEqual(self.stdout.getvalue(), "The train arrived.\n")
        self.assertEqual(self.stderr.getvalue(), "")
        self.constructor.assert_called_once_with(
            model_id=DEFAULT_MODEL_ID, revision=None, model_path=None,
            device=None, local_files_only=False, token_batch_size=256,
        )
        self.model.invert_text.assert_called_once_with("The train arrived.")
        self.model.invert_matrix.assert_not_called()

    def test_json_results_and_shared_model_options(self) -> None:
        self.assertEqual(main([
            "The train arrived.", "--json", "--model-id", "example/model", "--revision", "pinned",
            "--model-path", "/models/model", "--device", "cpu", "--local-files-only",
            "--token-batch-size", "64",
        ]), 0)
        self.assertEqual(json.loads(self.stdout.getvalue()),
                         {"text": "The train arrived.", "token_ids": [17], "input_rows": 3})
        self.constructor.assert_called_once_with(
            model_id="example/model", revision="pinned", model_path="/models/model",
            device="cpu", local_files_only=True, token_batch_size=64,
        )

    def test_standard_input_is_an_explicit_text_round_trip(self) -> None:
        with patch("sys.stdin", io.StringIO("The train arrived.\n")):
            self.assertEqual(main([]), 0)
        self.model.invert_text.assert_called_once_with("The train arrived.")

    def test_json_matrix_array(self) -> None:
        values = matrix()
        self.assertEqual(main(["--matrix-file", self.json_file(values)]), 0)
        self.model.invert_matrix.assert_called_once_with(values, attention_mask=None)
        self.model.invert_text.assert_not_called()

    def test_json_matrix_object_preserves_aligned_mask(self) -> None:
        values = matrix(5)
        mask = [True, True, True, False, False]
        path = self.json_file({"matrix": values, "attention_mask": mask})
        self.assertEqual(main(["--matrix-file", path, "--json"]), 0)
        self.model.invert_matrix.assert_called_once_with(values, attention_mask=mask)
        self.assertEqual(set(json.loads(self.stdout.getvalue())), {"text", "token_ids", "input_rows"})

    def test_numpy_matrix(self) -> None:
        values = np.asarray(matrix(), dtype=np.float32)
        self.assertEqual(main(["--matrix-file", self.numpy_file(values)]), 0)
        supplied = self.model.invert_matrix.call_args
        np.testing.assert_array_equal(supplied.args[0], values)
        self.assertEqual(supplied.kwargs, {"attention_mask": None})

    def test_json_batch_preserves_varied_lengths_and_masks(self) -> None:
        values = [matrix(), matrix(4)]
        masks = [[1, 1, 1], [1, 1, 1, 0]]
        path = self.json_file({"matrices": values, "attention_masks": masks})
        self.assertEqual(main(["--matrices-file", path, "--json"]), 0)
        self.model.invert_matrices.assert_called_once_with(values, attention_masks=masks)
        body = json.loads(self.stdout.getvalue())
        self.assertEqual([item["text"] for item in body["results"]], ["First.", "Second."])
        self.assertEqual([item["input_rows"] for item in body["results"]], [3, 4])
        self.assertEqual([item["token_ids"] for item in body["results"]], [[17], [17]])

    def test_numpy_batch_and_plain_text_output(self) -> None:
        values = np.asarray([matrix(), matrix()], dtype=np.float64)
        self.assertEqual(main(["--matrices-file", self.numpy_file(values)]), 0)
        np.testing.assert_array_equal(self.model.invert_matrices.call_args.args[0], values)
        self.assertEqual(self.stdout.getvalue(), "First.\nSecond.\n")

    def test_json_batch_array(self) -> None:
        values = [matrix(), matrix(4)]
        self.assertEqual(main(["--matrices-file", self.json_file(values)]), 0)
        self.model.invert_matrices.assert_called_once_with(values, attention_masks=None)

    def test_input_modes_are_exclusive(self) -> None:
        path = self.json_file(matrix())
        for arguments in (["text", "--matrix-file", path], ["text", "--matrices-file", path],
                          ["--matrix-file", path, "--matrices-file", path]):
            with self.subTest(arguments=arguments):
                self.assert_usage_error(arguments)

    def test_help_does_not_load_model_or_advertise_removed_options(self) -> None:
        with self.assertRaises(SystemExit) as raised:
            main(["--help"])
        self.assertEqual(raised.exception.code, 0)
        help_text = self.stdout.getvalue()
        self.assertIn("--matrix-file", help_text)
        self.assertIn("--matrices-file", help_text)
        for option in ("--candidates", "--max-length", "--seed", "--embedding"):
            self.assertNotIn(option, help_text)
        self.constructor.assert_not_called()

    def test_invalid_and_removed_arguments_fail_before_model_loading(self) -> None:
        invalid = [[""], ["text", "--token-batch-size", "0"], ["text", "--token-batch-size", "-1"],
                   ["text", "--token-batch-size", "257"],
                   ["text", "--token-batch-size", "1.5"], ["text", "--device", "cuda"],
                   ["text", "--candidates", "1"], ["text", "--seed", "0"],
                   ["text", "--max-length", "24"], ["--embedding", "anything"]]
        for arguments in invalid:
            with self.subTest(arguments=arguments):
                self.assert_usage_error(arguments)

    def test_empty_or_missing_stdin_does_not_load_model(self) -> None:
        with patch("sys.stdin", io.StringIO(" \n")):
            self.assert_usage_error([])
        with patch("sys.stdin.isatty", return_value=True):
            self.assert_usage_error([])

    def test_invalid_matrix_shapes_and_values_fail_before_model_loading(self) -> None:
        invalid = [[1.0] * 768, matrix(2), matrix(257), [[1] * 767] * 3,
                   [[1] * 769] * 3, matrix() + [["secret-coordinate"] * 768],
                   [[True] * 768] * 3, [[float("inf")] * 768] * 3, [[1e100] * 768] * 3]
        for values in invalid:
            with self.subTest(rows=len(values)):
                self.assert_usage_error(["--matrix-file", self.json_file(values)])
        self.assertNotIn("secret-coordinate", self.stderr.getvalue())

    def test_invalid_masks_fail_before_model_loading(self) -> None:
        for mask in ([1, 1, 0], [0, 1, 1, 1], [1, 1, 1, 0, 1], [1, 1, 1],
                     [1, 1, 1, 0.5], [1, 1, 1, "0"]):
            with self.subTest(mask=mask):
                path = self.json_file({"matrix": matrix(4), "attention_mask": mask})
                self.assert_usage_error(["--matrix-file", path])

    def test_invalid_batches_fail_before_model_loading(self) -> None:
        invalid = [[], matrix(), [matrix()] * 33, [matrix(), [1] * 768],
                   {"matrices": [matrix()], "attention_masks": []},
                   {"matrices": [matrix(), matrix(4)], "attention_masks": [[1, 1, 1], [1, 1, 1]]}]
        for values in invalid:
            with self.subTest(kind=type(values).__name__):
                self.assert_usage_error(["--matrices-file", self.json_file(values)])

    def test_unsupported_keys_and_unreadable_files_are_safe(self) -> None:
        for payload in ({"embedding": [1] * 768}, {"matrix": matrix(), "seed": 3},
                        {"matrix": matrix(), "attention_masks": [[1, 1, 1]]}):
            self.assert_usage_error(["--matrix-file", self.json_file(payload)])
        self.assert_usage_error(["--matrix-file", str(self.directory / "missing.json")])
        self.assert_usage_error(["--matrix-file", str(self.directory / "states.txt")])
        broken = self.directory / "broken.json"
        broken.write_text('{"secret-source-content":', encoding="utf-8")
        self.assert_usage_error(["--matrix-file", str(broken)])
        self.assertNotIn("secret-source-content", self.stderr.getvalue())

    def test_numpy_rejects_pickle_pooled_shapes_and_wrong_mode(self) -> None:
        for values in (np.asarray([{"secret": 1}], dtype=object), np.ones(768), np.ones((2, 3, 768))):
            self.assert_usage_error(["--matrix-file", self.numpy_file(values)])
        self.assert_usage_error(["--matrices-file", self.numpy_file(np.ones((3, 768)))])

    def test_numpy_huge_declared_shapes_never_reach_loader(self) -> None:
        cases = [((2**50, 768), "--matrix-file"),
                 ((3, 2**50), "--matrix-file"),
                 ((2**50, 3, 768), "--matrices-file"),
                 ((1, 2**50, 768), "--matrices-file")]
        with patch("aparecium.__main__.np.load") as load:
            for shape, option in cases:
                with self.subTest(shape=shape):
                    contents = self.numpy_header(shape)
                    self.assertLess(len(contents), 256)
                    self.assert_usage_error([option, self.bytes_file(contents)])
            load.assert_not_called()

    def test_numpy_invalid_declared_dimensions_never_reach_loader(self) -> None:
        cases = [((768,), "--matrix-file"), ((2, 768), "--matrix-file"),
                 ((257, 768), "--matrix-file"), ((3, 767), "--matrix-file"),
                 ((3, 769), "--matrix-file"), ((1, 3, 768), "--matrix-file"),
                 ((3, 768), "--matrices-file"), ((0, 3, 768), "--matrices-file"),
                 ((33, 3, 768), "--matrices-file"), ((1, 2, 768), "--matrices-file"),
                 ((1, 257, 768), "--matrices-file"), ((1, 3, 767), "--matrices-file")]
        with patch("aparecium.__main__.np.load") as load:
            for shape, option in cases:
                with self.subTest(shape=shape):
                    self.assert_usage_error([option, self.bytes_file(self.numpy_header(shape))])
            load.assert_not_called()

    def test_numpy_nonreal_dtypes_never_reach_loader(self) -> None:
        dtypes = ["O", "?", "c8", "c16", "U1", "S1", "V4", "M8[ns]", "m8[ns]",
                  np.dtype([("coordinate", "f4")]), np.dtype(("f4", (2,)))]
        with patch("aparecium.__main__.np.load") as load:
            for dtype in dtypes:
                with self.subTest(dtype=dtype):
                    dtype = np.dtype(dtype)
                    contents = self.numpy_header((3, 768), dtype) + bytes(3 * 768 * dtype.itemsize)
                    self.assert_usage_error(["--matrix-file", self.bytes_file(contents)])
            load.assert_not_called()

    def test_numpy_payload_must_exactly_match_declared_shape(self) -> None:
        header = self.numpy_header((3, 768))
        payload = np.asarray(matrix(), dtype="<f4").tobytes()
        with patch("aparecium.__main__.np.load") as load:
            for contents in (header, header + payload[:-1], header + payload + b"trailing-content"):
                with self.subTest(size=len(contents)):
                    self.assert_usage_error(["--matrix-file", self.bytes_file(contents)])
            load.assert_not_called()

    def test_numpy_malformed_unsupported_and_oversized_headers_never_reach_loader(self) -> None:
        valid = self.numpy_header((3, 768))
        invalid = [b"", b"not-npy", valid[:8], valid[:9], valid[:20],
                   np.lib.format.magic(3, 0) + valid[8:], np.lib.format.magic(4, 0) + valid[8:],
                   np.lib.format.magic(1, 0) + (MAX_NPY_HEADER_BYTES + 1).to_bytes(2, "little"),
                   np.lib.format.magic(2, 0) + (2**32 - 1).to_bytes(4, "little"),
                   np.lib.format.magic(1, 0) + (3).to_bytes(2, "little") + b"bad"]
        with patch("aparecium.__main__.np.load") as load:
            for contents in invalid:
                with self.subTest(prefix=contents[:10]):
                    self.assert_usage_error(["--matrix-file", self.bytes_file(contents)])
            load.assert_not_called()

    def test_numpy_supported_headers_orders_and_real_dtypes(self) -> None:
        cases = [((1, 0), "<f4", False), ((2, 0), ">f8", True),
                 ((1, 0), "<i8", False), ((2, 0), "<u2", False),
                 ((1, 0), "<f2", True)]
        original_load = np.load
        with patch("aparecium.__main__.np.load", wraps=original_load) as load:
            for version, dtype, fortran in cases:
                with self.subTest(version=version, dtype=dtype, fortran=fortran):
                    values = np.asarray(matrix(), dtype=dtype)
                    contents = self.numpy_header(values.shape, values.dtype, version=version,
                                                 fortran_order=fortran)
                    contents += values.tobytes(order="F" if fortran else "C")
                    self.assertEqual(main(["--matrix-file", self.bytes_file(contents)]), 0)
                    np.testing.assert_array_equal(self.model.invert_matrix.call_args.args[0], values)
                    self.assertEqual(load.call_args.kwargs, {"allow_pickle": False})
            self.assertEqual(load.call_count, len(cases))

    def test_file_and_stdin_size_limits(self) -> None:
        path = self.directory / "states.json"
        path.write_bytes(b"x" * 65)
        with patch("aparecium.__main__.MAX_INPUT_BYTES", 64):
            self.assert_usage_error(["--matrix-file", str(path)])
            with patch("sys.stdin", io.StringIO("x" * 65)):
                self.assert_usage_error([])

    def test_runtime_errors_do_not_expose_exception_or_input(self) -> None:
        self.constructor.side_effect = OSError("secret-source-content")
        self.assertEqual(main(["The train arrived."]), 2)
        self.assertEqual(self.stdout.getvalue(), "")
        self.assertEqual(self.stderr.getvalue(), "aparecium: reconstruction is unavailable\n")


if __name__ == "__main__":
    unittest.main()
