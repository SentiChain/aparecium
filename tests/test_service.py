"""Exercise HTTP matrix validation and lifecycle without external model access."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from aparecium.api import DEFAULT_MODEL_ID, InvertResult
from aparecium.service import INPUT_CONTRACT, create_app, main


def matrix(rows: int = 3) -> list[list[float]]:
    return [[float(index + 1)] * 768 for index in range(rows)]


def result(text: str = "The train arrived.", rows: int = 3) -> InvertResult:
    return InvertResult(text=text, token_ids=[17], input_rows=rows)


class TestHTTPService(unittest.TestCase):
    def setUp(self) -> None:
        self.model = MagicMock()
        self.model.invert_matrix.return_value = result()
        self.model.invert_matrices.side_effect = lambda values, **kwargs: [
            result(f"Reconstruction number {index}", len(value)) for index, value in enumerate(values)
        ]
        self.factory = MagicMock(return_value=self.model)
        self.app = create_app(self.factory)

    def assert_no_inversion(self) -> None:
        self.model.invert_matrix.assert_not_called()
        self.model.invert_matrices.assert_not_called()

    def test_lifespan_loads_one_model_and_health_describes_contract(self) -> None:
        self.factory.assert_not_called()
        with TestClient(self.app) as client:
            for _ in range(2):
                response = client.get("/health")
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json(), {"ready": True, "input_contract": INPUT_CONTRACT})
            self.assertEqual(client.post("/invert", json={"matrix": matrix()}).status_code, 200)
            self.factory.assert_called_once_with()
        self.assertIsNone(self.app.state.model)

    def test_openapi_uses_installed_package_metadata_and_matrix_fields(self) -> None:
        with patch("aparecium.service.metadata.version", return_value="package-release") as version:
            app = create_app(self.factory)
        version.assert_called_once_with("aparecium")
        schema = app.openapi()
        self.assertEqual(schema["info"]["version"], "package-release")
        properties = schema["components"]["schemas"]["InvertRequest"]["properties"]
        self.assertEqual(set(properties), {"matrix", "matrices", "attention_mask", "attention_masks"})

    def test_health_and_inversion_require_startup(self) -> None:
        with TestClient(self.app, raise_server_exceptions=False) as started:
            self.assertEqual(started.get("/health").status_code, 200)
        client = TestClient(self.app)
        response = client.get("/health")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json(), {"ready": False, "input_contract": INPUT_CONTRACT})
        response = client.post("/invert", json={"matrix": matrix()})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json(), {"detail": "Model is not ready"})
        self.assert_no_inversion()

    def test_initialization_failure_is_safe_and_reports_unready(self) -> None:
        self.factory.side_effect = OSError("secret-model-location")
        with self.assertLogs("aparecium.service", level="ERROR") as logs:
            with TestClient(self.app) as client:
                self.assertEqual(client.get("/health").status_code, 503)
                response = client.post("/invert", json={"matrix": matrix()})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json(), {"detail": "Model is not ready"})
        self.factory.assert_called_once_with()
        self.assertNotIn("secret-model-location", " ".join(logs.output))
        self.assert_no_inversion()

    def test_single_matrix_preserves_exact_result_fields(self) -> None:
        values = matrix()
        with TestClient(self.app) as client:
            response = client.post("/invert", json={"matrix": values})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"text": "The train arrived.", "token_ids": [17], "input_rows": 3})
        self.model.invert_matrix.assert_called_once_with(values, attention_mask=None)
        self.model.invert_matrices.assert_not_called()

    def test_masked_matrix_preserves_supplied_rows_and_mask(self) -> None:
        values = matrix(5)
        mask = [True, True, True, False, False]
        self.model.invert_matrix.return_value = result(rows=3)
        with TestClient(self.app) as client:
            response = client.post("/invert", json={"matrix": values, "attention_mask": mask})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["input_rows"], 3)
        self.model.invert_matrix.assert_called_once_with(values, attention_mask=mask)

    def test_varied_length_batch_preserves_order_and_aligned_masks(self) -> None:
        values = [matrix(), matrix(5)]
        masks = [[1, 1, 1], [1, 1, 1, 1, 0]]
        with TestClient(self.app) as client:
            response = client.post("/invert", json={"matrices": values, "attention_masks": masks})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"results": [
            {"text": "Reconstruction number 0", "token_ids": [17], "input_rows": 3},
            {"text": "Reconstruction number 1", "token_ids": [17], "input_rows": 5},
        ]})
        self.model.invert_matrices.assert_called_once_with(values, attention_masks=masks)
        self.model.invert_matrix.assert_not_called()

    def test_unmasked_batch_and_integer_coordinates(self) -> None:
        values = [[[1] * 768] * 3]
        with TestClient(self.app) as client:
            self.assertEqual(client.post("/invert", json={"matrices": values}).status_code, 200)
        self.model.invert_matrices.assert_called_once_with(values, attention_masks=None)

    def test_maximum_batch_and_row_bounds_are_accepted(self) -> None:
        with TestClient(self.app) as client:
            self.assertEqual(client.post("/invert", json={"matrices": [matrix()] * 32}).status_code, 200)
            self.assertEqual(client.post("/invert", json={"matrix": matrix(256)}).status_code, 200)
        self.assertEqual(len(self.model.invert_matrices.call_args.args[0]), 32)
        self.assertEqual(len(self.model.invert_matrix.call_args.args[0]), 256)

    def test_exclusive_inputs_shapes_and_removed_keys_fail_before_dispatch(self) -> None:
        invalid = [
            {}, {"matrix": None}, {"matrices": None},
            {"matrix": matrix(), "matrices": [matrix()]},
            {"matrix": matrix(), "matrices": None},
            {"matrix": [1] * 768}, {"matrix": matrix(2)}, {"matrix": matrix(257)},
            {"matrix": [[1] * 767] * 3}, {"matrix": [[1] * 769] * 3},
            {"matrices": []}, {"matrices": [matrix()] * 33},
            {"matrices": matrix()}, {"matrices": [matrix(), [1] * 768]},
            {"embedding": [1] * 768}, {"embeddings": [[1] * 768]},
            {"matrix": matrix(), "embedding": [1] * 768},
            {"matrix": matrix(), "num_candidates": 1}, {"matrix": matrix(), "seed": 0},
            {"matrix": matrix(), "max_length": 64}, {"matrix": matrix(), "rerank": False},
        ]
        with TestClient(self.app) as client:
            for payload in invalid:
                with self.subTest(keys=list(payload)):
                    response = client.post("/invert", json=payload)
                    self.assertEqual(response.status_code, 422)
                    self.assertEqual(response.json(), {"detail": "Invalid matrix request"})
        self.assert_no_inversion()

    def test_coordinates_are_strict_finite_real_numbers(self) -> None:
        with TestClient(self.app) as client:
            for bad in (True, False, "0.5", None, float("nan"), float("inf"), -float("inf"), 1e100):
                for field in ("matrix", "matrices"):
                    values = matrix()
                    values[1][1] = bad
                    payload = {field: values if field == "matrix" else [values]}
                    with self.subTest(value=bad, field=field):
                        response = client.post("/invert", content=json.dumps(payload),
                                               headers={"content-type": "application/json"})
                        self.assertEqual(response.status_code, 422)
        self.assert_no_inversion()

    def test_zero_coordinates_are_valid_without_normalization(self) -> None:
        values = [[0] * 768] * 3
        with TestClient(self.app) as client:
            self.assertEqual(client.post("/invert", json={"matrix": values}).status_code, 200)
        self.model.invert_matrix.assert_called_once_with(values, attention_mask=None)

    def test_mask_shape_prefix_and_alignment_errors_fail_before_dispatch(self) -> None:
        invalid = [
            {"matrix": matrix(4), "attention_mask": [1, 1, 1]},
            {"matrix": matrix(4), "attention_mask": [1, 1, 0, 0]},
            {"matrix": matrix(4), "attention_mask": [0, 1, 1, 1]},
            {"matrix": matrix(5), "attention_mask": [1, 1, 1, 0, 1]},
            {"matrix": matrix(4), "attention_mask": [1, 1, 1, 0.5]},
            {"matrix": matrix(4), "attention_mask": [1, 1, 1, "0"]},
            {"matrix": matrix(4), "attention_mask": [1, 1, 1, None]},
            {"matrix": matrix(), "attention_masks": [[1, 1, 1]]},
            {"matrix": matrix(), "attention_masks": None},
            {"matrices": [matrix()], "attention_mask": [1, 1, 1]},
            {"matrices": [matrix()], "attention_mask": None},
            {"matrices": [matrix()], "attention_masks": []},
            {"matrices": [matrix()], "attention_masks": [[1, 1, 1], [1, 1, 1]]},
            {"matrices": [matrix(), matrix(4)], "attention_masks": [[1, 1, 1], [1, 1, 1]]},
        ]
        with TestClient(self.app) as client:
            for payload in invalid:
                with self.subTest(keys=list(payload)):
                    self.assertEqual(client.post("/invert", json=payload).status_code, 422)
        self.assert_no_inversion()

    def test_floating_point_binary_mask_is_accepted(self) -> None:
        mask = [1.0, 1.0, 1.0, 0.0]
        with TestClient(self.app) as client:
            response = client.post("/invert", json={"matrix": matrix(4), "attention_mask": mask})
        self.assertEqual(response.status_code, 200)
        self.model.invert_matrix.assert_called_once_with(matrix(4), attention_mask=mask)

    def test_nonfinite_mask_and_padding_coordinates_are_rejected(self) -> None:
        values = matrix(4)
        values[-1][0] = float("nan")
        invalid = [{"matrix": values, "attention_mask": [1, 1, 1, 0]},
                   {"matrix": matrix(4), "attention_mask": [1, 1, 1, float("inf")]}]
        with TestClient(self.app) as client:
            for payload in invalid:
                response = client.post("/invert", content=json.dumps(payload),
                                       headers={"content-type": "application/json"})
                self.assertEqual(response.status_code, 422)
        self.assert_no_inversion()

    def test_invalid_json_and_validation_errors_do_not_echo_body(self) -> None:
        with TestClient(self.app) as client:
            for content in ('{"secret-source-content":', '{"secret-source-content": 17}',
                            '{"matrix": "secret-source-content"}'):
                response = client.post("/invert", content=content,
                                       headers={"content-type": "application/json"})
                self.assertEqual(response.status_code, 422)
                self.assertNotIn("secret-source-content", response.text)
        self.assert_no_inversion()

    def test_unparseable_encoding_numbers_and_deep_json_are_invalid_input(self) -> None:
        invalid = [b"\xff", b"[" * 2000 + b"]" * 2000,
                   b'{"matrix":' + b"1" * 5000 + b"}"]
        with TestClient(self.app) as client:
            for content in invalid:
                response = client.post("/invert", content=content,
                                       headers={"content-type": "application/json"})
                self.assertEqual(response.status_code, 422)
                self.assertEqual(response.json(), {"detail": "Invalid matrix request"})
        self.assert_no_inversion()

    def test_model_failures_are_503_and_do_not_expose_exception(self) -> None:
        for field, method in (("matrix", self.model.invert_matrix), ("matrices", self.model.invert_matrices)):
            method.side_effect = RuntimeError("secret-source-content")
            payload = {field: matrix() if field == "matrix" else [matrix()]}
            with self.subTest(field=field), self.assertLogs("aparecium.service", level="ERROR") as logs:
                with TestClient(self.app) as client:
                    response = client.post("/invert", json=payload)
            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.json(), {"detail": "Matrix reconstruction is unavailable"})
            self.assertNotIn("secret-source-content", " ".join(logs.output))

    def test_body_limit_checks_declared_and_actual_size(self) -> None:
        with patch("aparecium.service.MAX_REQUEST_BYTES", 64):
            app = create_app(self.factory)
        with TestClient(app) as client:
            response = client.post("/invert", content=b"x" * 65)
            self.assertEqual(response.status_code, 413)
            self.assertEqual(response.json(), {"detail": "Request body is too large"})
            response = client.post("/invert", content=b"x" * 65, headers={"content-length": "1"})
            self.assertEqual(response.status_code, 413)
            response = client.post("/invert", content=iter([b"x" * 40, b"x" * 40]))
            self.assertEqual(response.status_code, 413)
        self.assert_no_inversion()


class TestServiceCommand(unittest.TestCase):
    def setUp(self) -> None:
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_startup_defaults_include_cpu_safe_and_online_settings(self) -> None:
        with patch("uvicorn.run") as run, patch("aparecium.api.Aparecium") as constructor:
            self.assertEqual(main([]), 0)
            self.assertEqual(run.call_args.kwargs, {"host": "127.0.0.1", "port": 8000})
            with TestClient(run.call_args.args[0]):
                constructor.assert_called_once_with(
                    model_id=DEFAULT_MODEL_ID, revision=None, model_path=None,
                    device=None, local_files_only=False, token_batch_size=256,
                )

    def test_startup_uses_environment_and_explicit_overrides(self) -> None:
        settings = {"APARECIUM_MODEL_ID": "example/environment", "APARECIUM_MODEL_REVISION": "environment",
                    "APARECIUM_MODEL_PATH": "/environment/model", "APARECIUM_DEVICE": "mps",
                    "APARECIUM_LOCAL_FILES_ONLY": "true", "APARECIUM_TOKEN_BATCH_SIZE": "128"}
        with patch.dict(os.environ, settings), patch("uvicorn.run") as run, patch("aparecium.api.Aparecium") as constructor:
            self.assertEqual(main([
                "--host", "0.0.0.0", "--port", "8080", "--model-id", "example/argument",
                "--revision", "argument", "--model-path", "/models/model", "--device", "cpu",
                "--token-batch-size", "64",
            ]), 0)
            self.assertEqual(run.call_args.kwargs, {"host": "0.0.0.0", "port": 8080})
            with TestClient(run.call_args.args[0]):
                constructor.assert_called_once_with(
                    model_id="example/argument", revision="argument", model_path="/models/model",
                    device="cpu", local_files_only=True, token_batch_size=64,
                )

    def test_explicit_offline_flag_is_forwarded(self) -> None:
        with patch("uvicorn.run") as run, patch("aparecium.api.Aparecium") as constructor:
            self.assertEqual(main(["--local-files-only"]), 0)
            with TestClient(run.call_args.args[0]):
                self.assertTrue(constructor.call_args.kwargs["local_files_only"])

    def test_application_factory_reads_offline_environment_at_startup(self) -> None:
        with patch.dict(os.environ, {"APARECIUM_LOCAL_FILES_ONLY": "1"}), patch("aparecium.api.Aparecium") as constructor:
            app = create_app()
            constructor.assert_not_called()
            with TestClient(app):
                self.assertTrue(constructor.call_args.kwargs["local_files_only"])

    def test_help_does_not_start_server(self) -> None:
        with patch("uvicorn.run") as run, redirect_stdout(io.StringIO()) as output, self.assertRaises(SystemExit) as raised:
            main(["--help"])
        self.assertEqual(raised.exception.code, 0)
        self.assertIn("--token-batch-size", output.getvalue())
        self.assertIn("--local-files-only", output.getvalue())
        run.assert_not_called()

    def test_invalid_arguments_do_not_start_server(self) -> None:
        invalid = [["--port", port] for port in ("0", "65536", "-1", "text")]
        invalid += [["--device", "cuda"], ["--token-batch-size", "0"],
                    ["--token-batch-size", "257"], ["--seed", "1"]]
        with patch("uvicorn.run") as run, redirect_stderr(io.StringIO()):
            for arguments in invalid:
                with self.subTest(arguments=arguments), self.assertRaises(SystemExit) as raised:
                    main(arguments)
                self.assertEqual(raised.exception.code, 2)
        run.assert_not_called()

    def test_invalid_environment_does_not_start_server_or_echo_values(self) -> None:
        invalid = {"APARECIUM_DEVICE": "secret-value", "APARECIUM_LOCAL_FILES_ONLY": "secret-value",
                   "APARECIUM_TOKEN_BATCH_SIZE": "secret-value"}
        with patch("uvicorn.run") as run, redirect_stderr(io.StringIO()) as output:
            for name, value in invalid.items():
                with patch.dict(os.environ, {name: value}), self.subTest(name=name), self.assertRaises(SystemExit) as raised:
                    main([])
                self.assertEqual(raised.exception.code, 2)
        self.assertNotIn("secret-value", output.getvalue())
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
