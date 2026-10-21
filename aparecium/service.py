"""Serve reconstruction of MPNet token matrices over HTTP."""

from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
from dataclasses import asdict
from importlib import metadata
import logging
import os
from typing import Annotated, Any, Callable, Sequence, TYPE_CHECKING

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.exception_handlers import http_exception_handler
    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import JSONResponse
    from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator
    from starlette.concurrency import run_in_threadpool
    from starlette.exceptions import HTTPException as StarletteHTTPException
    from starlette.types import ASGIApp, Receive, Scope, Send
except ImportError as exc:
    raise ImportError('Install the HTTP dependencies with: pip install "aparecium[server]"') from exc

from .api import (
    DEFAULT_MODEL_ID, MAX_BATCH_SIZE, MAX_MATRIX_ROWS, STATE_WIDTH,
    _validate_matrices, _validate_matrix,
)

if TYPE_CHECKING:
    from .api import Aparecium


logger = logging.getLogger(__name__)
MAX_REQUEST_BYTES = 16 * 1024 * 1024
INPUT_CONTRACT = {
    "representation": "MPNet final hidden-state matrix",
    "columns": STATE_WIDTH,
    "minimum_rows": 3,
    "maximum_rows": MAX_MATRIX_ROWS,
    "maximum_batch_size": MAX_BATCH_SIZE,
    "attention_mask": "binary right-padding prefix with at least 3 visible rows",
}


def _binary_prefix(values: list[float | bool]) -> list[float | bool]:
    visible = 0
    padded = False
    for value in values:
        if value not in (0, 1):
            raise ValueError("attention masks must contain only zero or one")
        if value == 0:
            padded = True
        elif padded:
            raise ValueError("attention masks must use right padding")
        else:
            visible += 1
    if visible < 3:
        raise ValueError("attention masks must include at least 3 visible rows")
    return values


FiniteNumber = Annotated[float, Field(strict=True, allow_inf_nan=False)]
StrictBoolean = Annotated[bool, Field(strict=True)]
MatrixRow = Annotated[list[FiniteNumber], Field(min_length=STATE_WIDTH, max_length=STATE_WIDTH)]
Matrix = Annotated[list[MatrixRow], Field(min_length=3, max_length=MAX_MATRIX_ROWS)]
AttentionMask = Annotated[
    list[FiniteNumber | StrictBoolean],
    Field(min_length=3, max_length=MAX_MATRIX_ROWS),
    AfterValidator(_binary_prefix),
]


class InvertRequest(BaseModel):
    """Supply one token matrix or an ordered batch with aligned masks."""

    model_config = ConfigDict(extra="forbid", strict=True)

    matrix: Matrix | None = None
    matrices: Annotated[list[Matrix], Field(min_length=1, max_length=MAX_BATCH_SIZE)] | None = None
    attention_mask: AttentionMask | None = None
    attention_masks: Annotated[list[AttentionMask], Field(min_length=1, max_length=MAX_BATCH_SIZE)] | None = None

    @model_validator(mode="after")
    def aligned_inputs(self) -> InvertRequest:
        supplied = {"matrix", "matrices"} & self.model_fields_set
        if len(supplied) != 1 or (self.matrix is None and self.matrices is None):
            raise ValueError("provide exactly one of matrix or matrices")
        if self.matrix is not None:
            if "attention_masks" in self.model_fields_set:
                raise ValueError("attention_masks requires matrices")
            if self.attention_mask is not None and len(self.attention_mask) != len(self.matrix):
                raise ValueError("attention_mask must align with matrix rows")
            _validate_matrix(self.matrix, self.attention_mask)
        else:
            if "attention_mask" in self.model_fields_set:
                raise ValueError("attention_mask requires matrix")
            if self.attention_masks is not None:
                if len(self.attention_masks) != len(self.matrices):
                    raise ValueError("attention_masks must align with matrices")
                if any(len(mask) != len(matrix) for matrix, mask in zip(self.matrices, self.attention_masks)):
                    raise ValueError("each attention mask must align with its matrix rows")
            _validate_matrices(self.matrices, self.attention_masks)
        return self


class _BodyLimitMiddleware:
    """Bound buffered request bytes before JSON parsing or model dispatch."""

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        for name, value in scope.get("headers", []):
            if name.lower() == b"content-length":
                try:
                    declared_size = int(value)
                except ValueError:
                    declared_size = 0
                if declared_size > self.max_bytes:
                    await JSONResponse(status_code=413, content={"detail": "Request body is too large"})(scope, receive, send)
                    return
        chunks: list[bytes] = []
        size = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body = message.get("body", b"")
            size += len(body)
            if size > self.max_bytes:
                await JSONResponse(status_code=413, content={"detail": "Request body is too large"})(scope, receive, send)
                return
            chunks.append(body)
            if not message.get("more_body", False):
                break
        body = b"".join(chunks)
        delivered = False

        async def bounded_receive() -> dict[str, Any]:
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, bounded_receive, send)


def _environment_boolean(name: str) -> bool:
    value = os.environ.get(name, "false").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


def _environment_settings() -> dict[str, Any]:
    try:
        token_batch_size = int(os.environ.get("APARECIUM_TOKEN_BATCH_SIZE", "256"))
    except ValueError:
        raise ValueError("APARECIUM_TOKEN_BATCH_SIZE must be an integer between 1 and 256") from None
    if not 1 <= token_batch_size <= 256:
        raise ValueError("APARECIUM_TOKEN_BATCH_SIZE must be an integer between 1 and 256")
    device = os.environ.get("APARECIUM_DEVICE") or None
    if device not in {None, "cpu", "mps"}:
        raise ValueError("APARECIUM_DEVICE must be cpu or mps")
    return {
        "model_id": os.environ.get("APARECIUM_MODEL_ID") or DEFAULT_MODEL_ID,
        "revision": os.environ.get("APARECIUM_MODEL_REVISION") or None,
        "model_path": os.environ.get("APARECIUM_MODEL_PATH") or None,
        "device": device,
        "local_files_only": _environment_boolean("APARECIUM_LOCAL_FILES_ONLY"),
        "token_batch_size": token_batch_size,
    }


def _build_model(settings: dict[str, Any]) -> Aparecium:
    from .api import Aparecium

    return Aparecium(**settings)


def create_app(model_factory: Callable[[], Aparecium] | None = None) -> FastAPI:
    """Create an application with one model initialized during its lifespan."""
    factory = model_factory or (lambda: _build_model(_environment_settings()))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            app.state.model = await run_in_threadpool(factory)
        except Exception:
            logger.error("Model initialization failed")
        try:
            yield
        finally:
            app.state.model = None

    app = FastAPI(
        title="Aparecium",
        description="Reconstruct text from MPNet final hidden-state token matrices.",
        version=metadata.version("aparecium"),
        lifespan=lifespan,
    )
    app.add_middleware(_BodyLimitMiddleware, max_bytes=MAX_REQUEST_BYTES)
    app.state.model = None

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": "Invalid matrix request"})

    @app.exception_handler(StarletteHTTPException)
    async def request_error(request: Request, exc: StarletteHTTPException):
        if exc.status_code == 400 and request.url.path == "/invert":
            return JSONResponse(status_code=422, content={"detail": "Invalid matrix request"})
        return await http_exception_handler(request, exc)

    @app.get("/health")
    def health() -> JSONResponse:
        ready = app.state.model is not None
        return JSONResponse(status_code=200 if ready else 503,
                            content={"ready": ready, "input_contract": INPUT_CONTRACT})

    @app.post("/invert")
    def invert(payload: InvertRequest) -> JSONResponse:
        model = app.state.model
        if model is None:
            raise HTTPException(status_code=503, detail="Model is not ready")
        try:
            if payload.matrix is not None:
                content = asdict(model.invert_matrix(payload.matrix, attention_mask=payload.attention_mask))
            else:
                results = model.invert_matrices(payload.matrices, attention_masks=payload.attention_masks)
                content = {"results": [asdict(result) for result in results]}
            return JSONResponse(content=content)
        except Exception:
            logger.error("Matrix reconstruction failed")
            raise HTTPException(status_code=503, detail="Matrix reconstruction is unavailable") from None

    return app


def _positive_integer(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be an integer between 1 and 256") from None
    if not 1 <= number <= 256:
        raise argparse.ArgumentTypeError("must be an integer between 1 and 256")
    return number


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve Aparecium token-matrix reconstruction over HTTP.")
    parser.add_argument("--host", default="127.0.0.1", help="listen address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="listen port (default: 8000)")
    parser.add_argument("--model-id", help="Hugging Face model repository")
    parser.add_argument("--revision", help="model repository revision")
    parser.add_argument("--model-path", help="local model directory")
    parser.add_argument("--device", choices=("cpu", "mps"), help="execution device (default: cpu)")
    parser.add_argument("--local-files-only", action="store_true", default=None,
                        help="use only locally available model files")
    parser.add_argument("--token-batch-size", type=_positive_integer,
                        help="token rows decoded per batch, 1 to 256 (default: 256)")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    try:
        settings = _environment_settings()
    except ValueError as exc:
        parser.error(str(exc))
    for name in ("model_id", "revision", "model_path", "device", "local_files_only", "token_batch_size"):
        value = getattr(args, name)
        if value is not None:
            settings[name] = value
    import uvicorn

    app = create_app(model_factory=lambda: _build_model(settings))
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
