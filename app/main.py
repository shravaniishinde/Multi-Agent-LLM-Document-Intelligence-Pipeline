"""FastAPI application.

    uvicorn app.main:app --host 127.0.0.1 --port 8000

The pipeline is built once at startup from environment settings (offline by
default). Tests pass a pre-built `DocumentService` to `create_app` instead.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import __version__
from app.api.routes import ResultStore, router
from app.config import Settings
from app.observability import configure_logging, log_event
from app.services import DocumentService, build_pipeline


def create_app(service: DocumentService | None = None,
               settings: Settings | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal service
        if service is None:
            cfg = settings or Settings.from_env()
            configure_logging(cfg)
            service = build_pipeline(cfg)
        app.state.service = service
        app.state.results = ResultStore()
        log_event("app_started", components=service.components,
                  max_retries=service.settings.max_retries)
        yield

    app = FastAPI(
        title="Document Intelligence Pipeline",
        version=__version__,
        description="Four-agent LangGraph pipeline: OCR -> validation (+RAG) -> "
                    "structured output -> supervisory review.",
        lifespan=lifespan,
    )
    app.include_router(router)
    return app


app = create_app()
