"""Batch processing with bounded concurrency.

Each document goes through `DocumentService.process` on a worker thread
(`max_concurrency` workers at most). `process` already turns pipeline exceptions
into `failed` results; the extra guard here also covers anything raised outside
the graph, so one bad document can never abort the batch.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pydantic import BaseModel, Field

from app.schemas.document import DocumentInput
from app.schemas.review import ProcessingResult, ReviewDecision
from app.services.document_service import DocumentService, failed_result

SUPPORTED_SUFFIXES = {".txt", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".pdf"}


class BatchSummary(BaseModel):
    total: int
    successful: int = Field(description="auto-approved")
    escalated: int = Field(description="completed, routed to human review")
    failed: int = Field(description="pipeline raised for this document")
    total_retries: int
    validation_failures: int
    average_processing_ms: float
    wall_clock_ms: float
    by_decision: dict[str, int]
    results: list[ProcessingResult] = Field(default_factory=list)


def summarize(results: list[ProcessingResult], wall_clock_ms: float = 0.0) -> BatchSummary:
    n = len(results)
    failed = sum(r.status == "failed" for r in results)
    approved = sum(r.status == "completed" and r.decision == ReviewDecision.APPROVE
                   for r in results)
    return BatchSummary(
        total=n,
        successful=approved,
        escalated=n - failed - approved,
        failed=failed,
        total_retries=sum(r.retry_count for r in results),
        validation_failures=sum(r.validation_failures for r in results),
        average_processing_ms=round(sum(r.processing_ms for r in results) / n, 2) if n else 0.0,
        wall_clock_ms=round(wall_clock_ms, 2),
        by_decision={d.value: sum(r.decision == d for r in results)
                     for d in (ReviewDecision.APPROVE, ReviewDecision.ESCALATE)},
        results=results,
    )


class BatchService:
    def __init__(self, service: DocumentService, max_concurrency: int | None = None):
        self.service = service
        self.max_concurrency = max_concurrency or service.settings.batch_max_concurrency

    def _process_one(self, document: DocumentInput) -> ProcessingResult:
        try:
            return self.service.process(document)
        except Exception as exc:  # noqa: BLE001
            result = failed_result(document, exc)
            self.service.metrics.record(result)
            return result

    def process_many(self, documents: list[DocumentInput]) -> BatchSummary:
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=self.max_concurrency) as pool:
            results = list(pool.map(self._process_one, documents))  # keeps input order
        return summarize(results, (time.perf_counter() - started) * 1000)


def discover_documents(input_dir: Path) -> list[DocumentInput]:
    """Find documents in a directory.

    A directory with a `manifest.json` (like `data/sample_documents`) is a
    mock-OCR sample set: one document per manifest id, scans resolved by the
    mock adapter. Otherwise every supported file becomes a document whose id is
    the file name without extension.
    """
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"input directory not found: {input_dir}")
    manifest = input_dir / "manifest.json"
    if manifest.exists():
        return [DocumentInput(document_id=doc_id)
                for doc_id in sorted(json.loads(manifest.read_text(encoding="utf-8")))]
    return [DocumentInput(document_id=p.stem, file_path=str(p))
            for p in sorted(input_dir.iterdir())
            if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES]
