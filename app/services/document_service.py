"""Single-document entry point: builds the graph once, runs documents through it.

`process` never raises for a bad document: an exception inside the pipeline is
logged and returned as a `failed` result (decision ESCALATE), and every result
is timed and recorded in the metrics store.
"""

from __future__ import annotations

import logging
import time
import uuid

from app.adapters.llm import LLMAdapter, build_llm
from app.adapters.ocr import OCRAdapter, build_ocr
from app.config import Settings
from app.graph.state import PipelineState, initial_state
from app.graph.workflow import build_workflow, recursion_limit
from app.observability import MetricsStore, log_event
from app.rag.knowledge_base import build_retriever
from app.rag.retriever import Retriever
from app.schemas.document import DocumentInput
from app.schemas.review import ProcessingResult, ReviewDecision, SupervisorReview


class DocumentService:
    def __init__(self, ocr: OCRAdapter, llm: LLMAdapter, retriever: Retriever | None,
                 settings: Settings, metrics: MetricsStore | None = None):
        self.settings = settings
        self.components = {
            "ocr": ocr.name,
            "llm": llm.name,
            "vector_store": type(retriever.store).__name__ if retriever else "none",
        }
        self.graph = build_workflow(ocr, llm, retriever, settings)
        self.config = {"recursion_limit": recursion_limit(settings)}
        self.metrics = metrics or MetricsStore()

    def run(self, document: DocumentInput, correlation_id: str = "") -> PipelineState:
        """Run the graph and return the raw final state (raises on pipeline errors)."""
        return self.graph.invoke(initial_state(document, correlation_id), config=self.config)

    def process(self, document: DocumentInput) -> ProcessingResult:
        correlation_id = uuid.uuid4().hex[:12]
        ids = {"document_id": document.document_id, "correlation_id": correlation_id}
        log_event("document_received", ids, file_path=document.file_path)
        started = time.perf_counter()
        try:
            result = to_result(self.run(document, correlation_id))
        except Exception as exc:  # noqa: BLE001 - one bad document must not crash callers
            result = failed_result(document, exc, correlation_id)
            log_event("document_failed", ids, logging.ERROR,
                      error=f"{type(exc).__name__}: {exc}")
        result.processing_ms = round((time.perf_counter() - started) * 1000, 2)
        if result.status == "completed":
            log_event("document_completed", ids, decision=result.decision.value,
                      retry_count=result.retry_count, processing_ms=result.processing_ms)
        self.metrics.record(result)
        return result


def to_result(state: PipelineState) -> ProcessingResult:
    review = state.get("review") or SupervisorReview(
        decision=ReviewDecision.ESCALATE, reasons=["pipeline ended without a review"],
        confidence=0.0, summary="Escalated: no supervisor review")
    return ProcessingResult(
        document_id=state["document_id"],
        correlation_id=state.get("correlation_id", ""),
        decision=review.decision,
        review=review,
        record=state.get("record"),
        extracted_fields=state.get("extracted_fields"),
        validation=state.get("validation"),
        retry_count=state.get("retry_count", 0),
        structuring_attempts=state.get("structuring_attempts", 0),
        validation_failures=state.get("validation_failures", 0),
        structuring_failures=state.get("structuring_failures", 0),
        events=state.get("events", []),
        errors=state.get("errors", []),
    )


def failed_result(document: DocumentInput, exc: BaseException,
                  correlation_id: str = "") -> ProcessingResult:
    message = f"pipeline error: {type(exc).__name__}: {exc}"
    review = SupervisorReview(decision=ReviewDecision.ESCALATE, reasons=[message],
                              confidence=0.0, summary="Escalated: processing error")
    return ProcessingResult(document_id=document.document_id, status="failed",
                            correlation_id=correlation_id, decision=review.decision,
                            review=review, errors=[message])


def build_pipeline(settings: Settings | None = None,
                   metrics: MetricsStore | None = None) -> DocumentService:
    """Wire adapters from settings (offline by default)."""
    settings = settings or Settings.from_env()
    return DocumentService(build_ocr(settings), build_llm(settings),
                           build_retriever(settings), settings, metrics)
