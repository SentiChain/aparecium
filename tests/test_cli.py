"""Command-line checks without model loading or network access."""

from __future__ import annotations

from contextlib import ExitStack, redirect_stderr, redirect_stdout
import io
import json
import unittest
from unittest.mock import patch

from aparecium.__main__ import main
from aparecium.api import InvertResult


class TestCLI(unittest.TestCase):
    def setUp(self) -> None:
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.constructor = self.stack.enter_context(patch("aparecium.api.Aparecium"))
        self.constructor.return_value.invert_text.return_value = InvertResult(
            text="The train arrived.", candidates=["A train came.", "The train arrived."],
            scores=[0.7, 0.9], selected_index=1, stopped=True,
        )
        self.stdout = self.stack.enter_context(redirect_stdout(io.StringIO()))
        self.stderr = self.stack.enter_context(redirect_stderr(io.StringIO()))

    def test_text_output_and_defaults(self) -> None:
        self.assertEqual(main(["The train arrived on time."]), 0)
        self.assertEqual(self.stdout.getvalue(), "The train arrived.\n")
        self.assertEqual(self.stderr.getvalue(), "")
        self.constructor.assert_called_once_with(
            model_id="SentiChain/aparecium", revision=None, model_path=None,
            device=None, local_files_only=False,
        )

    def test_json_contains_aligned_candidate_details(self) -> None:
        self.assertEqual(main(["The train arrived.", "--json", "--candidates", "2", "--seed", "7",
                               "--max-length", "24", "--model-id", "example/model", "--revision", "0123456789abcdef",
                               "--model-path", "/models/model", "--device", "cpu", "--local-files-only"]), 0)
        result = json.loads(self.stdout.getvalue())
        self.assertEqual(result["text"], result["candidates"][result["selected_index"]])
        self.assertEqual(result["scores"][result["selected_index"]], 0.9)
        self.constructor.assert_called_once_with(model_id="example/model", revision="0123456789abcdef",
                                                model_path="/models/model", device="cpu", local_files_only=True)
        self.constructor.return_value.invert_text.assert_called_once_with(
            "The train arrived.", num_candidates=2, max_length=24, seed=7,
        )

    def test_standard_input(self) -> None:
        with patch("sys.stdin", io.StringIO("A train arrived.\n")):
            self.assertEqual(main([]), 0)
        self.constructor.return_value.invert_text.assert_called_once_with(
            "A train arrived.", num_candidates=5, max_length=None, seed=0,
        )

    def test_help_does_not_load_model(self) -> None:
        with self.assertRaises(SystemExit) as raised:
            main(["--help"])
        self.assertEqual(raised.exception.code, 0)
        self.assertIn("--candidates", self.stdout.getvalue())
        self.constructor.assert_not_called()

    def test_invalid_arguments_fail_before_model_loading(self) -> None:
        for arguments in ([""], ["text", "--candidates", "0"], ["text", "--candidates", "33"], ["text", "--max-length", "0"],
                          ["text", "--seed", "-1"], ["text", "--seed", str(2**63)]):
            with self.subTest(arguments=arguments), self.assertRaises(SystemExit) as raised:
                main(arguments)
            self.assertEqual(raised.exception.code, 2)
        self.constructor.assert_not_called()

    def test_empty_stdin_has_actionable_error(self) -> None:
        with patch("sys.stdin", io.StringIO(" \n")), self.assertRaises(SystemExit) as raised:
            main([])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("text must not be empty", self.stderr.getvalue())
        self.constructor.assert_not_called()

    def test_missing_interactive_input_does_not_block(self) -> None:
        with patch("sys.stdin.isatty", return_value=True), self.assertRaises(SystemExit) as raised:
            main([])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("provide text", self.stderr.getvalue())
        self.constructor.assert_not_called()

    def test_runtime_error_is_reported_only_on_stderr(self) -> None:
        self.constructor.side_effect = OSError("Model files are unavailable")
        self.assertEqual(main(["A train arrived."]), 2)
        self.assertEqual(self.stdout.getvalue(), "")
        self.assertEqual(self.stderr.getvalue(), "aparecium: Model files are unavailable\n")


if __name__ == "__main__":
    unittest.main()
