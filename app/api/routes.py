"""HTTP endpoints. Each handler validates input, calls the service, maps the result."""

from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

from app import __version__
from app.api.models import (
    BatchRequest,
    BatchResponse,
    DocumentResponse,
    HealthResponse,
    ProcessRequest,
)
from app.observability import MetricsSnapshot
from app.schemas.document import DocumentInput
from app.schemas.review import ProcessingResult
from app.services import BatchService, DocumentService

router = APIRouter()


class ResultStore:
    """Most recent result per document id, bounded (oldest evicted). In-memory only."""

    def __init__(self, max_items: int = 1000):
        self.max_items = max_items
        self._items: OrderedDict[str, ProcessingResult] = OrderedDict()
        self._lock = threading.Lock()

    def put(self, result: ProcessingResult) -> None:
        with self._lock:
            self._items.pop(result.document_id, None)
            self._items[result.document_id] = result
            while len(self._items) > self.max_items:
                self._items.popitem(last=False)

    def get(self, document_id: str) -> ProcessingResult | None:
        with self._lock:
            return self._items.get(document_id)


def _service(request: Request) -> DocumentService:
    return request.app.state.service


def _to_input(req: ProcessRequest, service: DocumentService) -> DocumentInput:
    """Resolve `file_path` inside DOCUMENTS_DIR; refuse anything that escapes it."""
    if req.file_path is None:
        return DocumentInput(document_id=req.document_id)
    root = Path(service.settings.documents_dir).resolve()
    path = (root / req.file_path).resolve()
    if not path.is_relative_to(root):
        raise HTTPException(status_code=400, detail="file_path must be inside DOCUMENTS_DIR")
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"file not found: {req.file_path}")
    return DocumentInput(document_id=req.document_id, file_path=str(path))


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    service = _service(request)
    s = service.settings
    offline = s.llm_provider == "fake" and s.ocr_provider == "mock" \
        and service.components["vector_store"] != "PineconeVectorStore"
    return HealthResponse(status="ok", version=__version__, offline_mode=offline,
                          components=service.components, max_retries=s.max_retries)


@router.post("/documents/process", response_model=DocumentResponse)
async def process_document(body: ProcessRequest, request: Request) -> DocumentResponse:
    service = _service(request)
    result = await run_in_threadpool(service.process, _to_input(body, service))
    request.app.state.results.put(result)
    return DocumentResponse.from_result(result)


@router.post("/documents/batch", response_model=BatchResponse)
async def process_batch(body: BatchRequest, request: Request) -> BatchResponse:
    service = _service(request)
    inputs = [_to_input(d, service) for d in body.documents]
    summary = await run_in_threadpool(
        BatchService(service, body.max_concurrency).process_many, inputs)
    for result in summary.results:
        request.app.state.results.put(result)
    return BatchResponse.from_summary(summary)


@router.get("/documents/{document_id}", response_model=DocumentResponse)
def get_document(document_id: str, request: Request) -> DocumentResponse:
    result = request.app.state.results.get(document_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"no result for document {document_id!r}")
    return DocumentResponse.from_result(result)


@router.get("/metrics", response_model=MetricsSnapshot)
def metrics(request: Request) -> MetricsSnapshot:
    return _service(request).metrics.snapshot()
