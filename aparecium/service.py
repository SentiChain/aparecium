"""Serve sentence-embedding reconstruction over HTTP."""

from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
from dataclasses import asdict
from importlib import metadata
import logging
import os
from typing import Annotated, Callable, Sequence, TYPE_CHECKING

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import JSONResponse
    from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator
    from starlette.concurrency import run_in_threadpool
except ImportError as exc:
    raise ImportError('Install the HTTP dependencies with: pip install "aparecium[server]"') from exc

if TYPE_CHECKING:
    from .api import Aparecium


logger = logging.getLogger(__name__)


def _nonzero(values: list[float]) -> list[float]:
    if not any(values):
        raise ValueError("embedding must not be a zero vector")
    return values


FiniteNumber = Annotated[float, Field(strict=True, allow_inf_nan=False)]
Embedding = Annotated[
    list[FiniteNumber], Field(min_length=768, max_length=768), AfterValidator(_nonzero)
]


class InvertRequest(BaseModel):
    """Supply either one embedding or a batch of up to 32 embeddings."""

    model_config = ConfigDict(extra="forbid")

    embedding: Embedding | None = None
    embeddings: Annotated[list[Embedding], Field(min_length=1, max_length=32)] | None = None
    num_candidates: Annotated[int, Field(strict=True, ge=1, le=32)] = 5
    max_length: Annotated[int, Field(strict=True, ge=1, le=64)] = 64
    seed: Annotated[int, Field(strict=True, ge=0, le=2**63 - 1)] = 0

    @model_validator(mode="after")
    def exactly_one_input(self) -> InvertRequest:
        supplied = {"embedding", "embeddings"} & self.model_fields_set
        if len(supplied) != 1 or (self.embedding is None and self.embeddings is None):
            raise ValueError("provide exactly one of embedding or embeddings")
        return self


def _environment_settings() -> dict[str, str | None]:
    return {
        "model_id": os.environ.get("APARECIUM_MODEL_ID", "SentiChain/aparecium"),
        "revision": os.environ.get("APARECIUM_MODEL_REVISION") or None,
        "model_path": os.environ.get("APARECIUM_MODEL_PATH") or None,
        "device": os.environ.get("APARECIUM_DEVICE") or None,
    }


def _build_model(settings: dict[str, str | None]) -> Aparecium:
    from .api import Aparecium

    return Aparecium(**settings)


def create_app(model_factory: Callable[[], Aparecium] | None = None) -> FastAPI:
    """Create an application that loads one model during startup."""
    factory = model_factory or (lambda: _build_model(_environment_settings()))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.model = await run_in_threadpool(factory)
        try:
            yield
        finally:
            app.state.model = None

    app = FastAPI(
        title="Aparecium",
        description="Reconstruct English text from stock MPNet sentence embeddings.",
        version=metadata.version("aparecium"),
        lifespan=lifespan,
    )
    app.state.model = None

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = [{"loc": error["loc"], "msg": error["msg"], "type": error["type"]}
                  for error in exc.errors()]
        return JSONResponse(status_code=422, content={"detail": errors})

    @app.get("/health")
    def health() -> dict[str, str]:
        if app.state.model is None:
            raise HTTPException(status_code=503, detail="Model is not ready")
        return {"status": "ready"}

    @app.post("/invert")
    def invert(payload: InvertRequest) -> JSONResponse:
        model = app.state.model
        if model is None:
            raise HTTPException(status_code=503, detail="Model is not ready")
        options = {
            "num_candidates": payload.num_candidates,
            "max_length": payload.max_length,
            "seed": payload.seed,
        }
        try:
            if payload.embedding is not None:
                content = asdict(model.invert_embedding(payload.embedding, **options))
            else:
                results = model.invert_embeddings(payload.embeddings, **options)
                content = {"results": [asdict(result) for result in results]}
            return JSONResponse(content=content)
        except Exception:
            logger.exception("Text reconstruction failed")
            raise HTTPException(status_code=503, detail="Text reconstruction is unavailable") from None

    return app


def main(argv: Sequence[str] | None = None) -> int:
    settings = _environment_settings()
    parser = argparse.ArgumentParser(description="Serve Aparecium sentence-embedding reconstruction over HTTP.")
    parser.add_argument("--host", default="127.0.0.1", help="listen address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="listen port (default: 8000)")
    parser.add_argument("--model-id", default=settings["model_id"], help="Hugging Face model repository")
    parser.add_argument("--revision", default=settings["revision"], help="model repository revision (default: main)")
    parser.add_argument("--model-path", default=settings["model_path"], help="local model directory")
    parser.add_argument("--device", default=settings["device"], help="cpu, cuda, cuda:<index>, or mps")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    import uvicorn

    settings = {"model_id": args.model_id, "revision": args.revision,
                "model_path": args.model_path, "device": args.device}
    app = create_app(model_factory=lambda: _build_model(settings))
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
