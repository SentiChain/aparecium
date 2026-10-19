"""HTTP validation and reconstruction checks without network access."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from aparecium.api import InvertResult
from aparecium.service import create_app, main


def vector(index: int = 0) -> list[float]:
    values = [0.0] * 768
    values[index] = 1.0
    return values


def result(text: str = "The train arrived.") -> InvertResult:
    return InvertResult(text=text, candidates=["A train came.", text], scores=[0.7, 0.9],
                        selected_index=1, stopped=True)


class TestHTTPService(unittest.TestCase):
    def setUp(self) -> None:
        self.model = MagicMock()
        self.model.invert_embedding.return_value = result()
        self.model.invert_embeddings.side_effect = lambda values, **kwargs: [
            result(f"Reconstruction number {i}") for i, _ in enumerate(values)
        ]
        self.factory = MagicMock(return_value=self.model)
        self.app = create_app(self.factory)

    def test_lifespan_loads_model_once_and_health_is_ready(self) -> None:
        self.factory.assert_not_called()
        with TestClient(self.app) as client:
            for _ in range(2):
                response = client.get("/health")
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json(), {"status": "ready"})
            client.post("/invert", json={"embedding": vector()})
            self.factory.assert_called_once_with()
        self.assertIsNone(self.app.state.model)

    def test_openapi_version_matches_installed_package_metadata(self) -> None:
        with patch("aparecium.service.metadata.version", return_value="package-release") as version:
            app = create_app(self.factory)
        version.assert_called_once_with("aparecium")
        self.assertEqual(app.openapi()["info"]["version"], "package-release")

    def test_health_and_inversion_require_startup(self) -> None:
        client = TestClient(self.app)
        self.assertEqual(client.get("/health").status_code, 503)
        response = client.post("/invert", json={"embedding": vector()})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json(), {"detail": "Model is not ready"})
        self.model.invert_embedding.assert_not_called()

    def test_single_embedding_preserves_result_fields_and_options(self) -> None:
        payload = {"embedding": vector(), "num_candidates": 2, "max_length": 24, "seed": 7}
        with TestClient(self.app) as client:
            response = client.post("/invert", json=payload)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["text"], body["candidates"][body["selected_index"]])
        self.assertEqual(body["scores"], [0.7, 0.9])
        self.assertTrue(body["stopped"])
        self.model.invert_embedding.assert_called_once_with(vector(), num_candidates=2, max_length=24, seed=7)
        self.model.invert_embeddings.assert_not_called()

    def test_batch_preserves_order_and_defaults(self) -> None:
        values = [vector(0), vector(1)]
        with TestClient(self.app) as client:
            response = client.post("/invert", json={"embeddings": values})
        self.assertEqual(response.status_code, 200)
        self.assertEqual([item["text"] for item in response.json()["results"]],
                         ["Reconstruction number 0", "Reconstruction number 1"])
        self.model.invert_embeddings.assert_called_once_with(values, num_candidates=5, max_length=64, seed=0)
        self.model.invert_embedding.assert_not_called()

    def test_integer_coordinates_are_valid_real_numbers(self) -> None:
        values = [1] + [0] * 767
        with TestClient(self.app) as client:
            response = client.post("/invert", json={"embedding": values})
        self.assertEqual(response.status_code, 200)

    def test_exclusive_inputs_dimensions_and_extra_fields_are_validated(self) -> None:
        invalid = [
            {}, {"embedding": None}, {"embeddings": None},
            {"embedding": vector(), "embeddings": [vector()]},
            {"embedding": vector(), "embeddings": None},
            {"embedding": [1] * 767}, {"embedding": [1] * 769},
            {"embeddings": []}, {"embeddings": [vector()] * 33},
            {"embeddings": [[1] * 767]}, {"embedding": [0] * 768},
            {"embeddings": [vector(), [0] * 768]},
            {"embedding": vector(), "unexpected": True},
        ]
        with TestClient(self.app) as client:
            for payload in invalid:
                with self.subTest(keys=list(payload)):
                    self.assertEqual(client.post("/invert", json=payload).status_code, 422)
        self.model.invert_embedding.assert_not_called()
        self.model.invert_embeddings.assert_not_called()

    def test_numbers_are_strict_and_finite_in_single_and_batch_requests(self) -> None:
        with TestClient(self.app) as client:
            for bad in (True, False, "0.5", None, float("nan"), float("inf"), -float("inf")):
                for field in ("embedding", "embeddings"):
                    values = vector()
                    values[1] = bad
                    payload = {field: values if field == "embedding" else [values]}
                    with self.subTest(value=bad, field=field):
                        response = client.post("/invert", content=json.dumps(payload),
                                               headers={"Content-Type": "application/json"})
                        self.assertEqual(response.status_code, 422)
                        self.assertTrue(response.json()["detail"])
                        self.assertNotIn("input", response.json()["detail"][0])
        self.model.invert_embedding.assert_not_called()
        self.model.invert_embeddings.assert_not_called()

    def test_generation_options_reject_coercion_and_out_of_range_values(self) -> None:
        invalid = {
            "num_candidates": [True, 1.0, "2", 0, 33, None],
            "max_length": [False, 1.0, "2", 0, 65, None],
            "seed": [True, 0.0, "2", -1, 2**63, None],
        }
        with TestClient(self.app) as client:
            for field, values in invalid.items():
                for value in values:
                    with self.subTest(field=field, value=value):
                        response = client.post("/invert", json={"embedding": vector(), field: value})
                        self.assertEqual(response.status_code, 422)
        self.model.invert_embedding.assert_not_called()

    def test_supported_upper_bounds_are_accepted(self) -> None:
        values = [vector()] * 32
        with TestClient(self.app) as client:
            response = client.post("/invert", json={"embeddings": values, "num_candidates": 32,
                                                    "max_length": 64, "seed": 2**63 - 1})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()["results"]), 32)

    def test_runtime_failure_is_logged_and_redacted(self) -> None:
        self.model.invert_embedding.side_effect = RuntimeError("private model path and diagnostic details")
        with TestClient(self.app) as client, self.assertLogs("aparecium.service", level="ERROR") as logs:
            response = client.post("/invert", json={"embedding": vector()})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json(), {"detail": "Text reconstruction is unavailable"})
        self.assertNotIn("private", response.text)
        self.assertIn("private model path", "\n".join(logs.output))

    def test_invalid_output_is_redacted(self) -> None:
        self.model.invert_embedding.return_value = InvertResult(
            text="The train arrived.", candidates=["The train arrived."], scores=[float("nan")],
            selected_index=0, stopped=True,
        )
        with TestClient(self.app) as client, self.assertLogs("aparecium.service", level="ERROR"):
            response = client.post("/invert", json={"embedding": vector()})
        self.assertEqual(response.status_code, 503)

    def test_environment_settings_are_used_without_loading_at_creation(self) -> None:
        settings = {"APARECIUM_MODEL_ID": "example/model", "APARECIUM_MODEL_REVISION": "0123456789abcdef",
                    "APARECIUM_MODEL_PATH": "/models/model", "APARECIUM_DEVICE": "cpu"}
        with patch.dict(os.environ, settings, clear=True), patch("aparecium.api.Aparecium") as constructor:
            app = create_app()
            constructor.assert_not_called()
            with TestClient(app):
                constructor.assert_called_once_with(model_id="example/model", revision="0123456789abcdef",
                                                    model_path="/models/model", device="cpu")


class TestServiceCLI(unittest.TestCase):
    def test_defaults_use_loopback_and_standard_port(self) -> None:
        with patch.dict(os.environ, {}, clear=True), patch("uvicorn.run") as run, patch("aparecium.api.Aparecium") as constructor:
            self.assertEqual(main([]), 0)
            constructor.assert_not_called()
            self.assertEqual(run.call_args.kwargs, {"host": "127.0.0.1", "port": 8000})
            with TestClient(run.call_args.args[0]):
                constructor.assert_called_once_with(model_id="SentiChain/aparecium", revision=None,
                                                    model_path=None, device=None)

    def test_explicit_settings_override_environment(self) -> None:
        with patch.dict(os.environ, {"APARECIUM_MODEL_ID": "example/environment"}, clear=True), \
                patch("uvicorn.run") as run, patch("aparecium.api.Aparecium") as constructor:
            self.assertEqual(main(["--host", "0.0.0.0", "--port", "8080", "--model-id", "example/argument",
                                   "--revision", "0123456789abcdef", "--model-path", "/models/model", "--device", "cpu"]), 0)
            self.assertEqual(run.call_args.kwargs, {"host": "0.0.0.0", "port": 8080})
            with TestClient(run.call_args.args[0]):
                constructor.assert_called_once_with(model_id="example/argument", revision="0123456789abcdef",
                                                    model_path="/models/model", device="cpu")

    def test_help_does_not_start_server(self) -> None:
        with patch("uvicorn.run") as run, redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as raised:
            main(["--help"])
        self.assertEqual(raised.exception.code, 0)
        run.assert_not_called()

    def test_invalid_ports_do_not_start_server(self) -> None:
        with patch("uvicorn.run") as run, redirect_stderr(io.StringIO()):
            for port in ("0", "65536", "-1", "text"):
                with self.subTest(port=port), self.assertRaises(SystemExit) as raised:
                    main(["--port", port])
                self.assertEqual(raised.exception.code, 2)
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
