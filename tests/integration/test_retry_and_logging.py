"""Retry bounds, escalation, and the structured log trail."""

import json
import logging

import pytest

import app.graph.workflow as workflow
from app.adapters.llm import FakeLLMAdapter
from app.config import Settings
from app.observability import JsonFormatter, logger
from app.schemas.document import DocumentInput
from app.schemas.review import ReviewDecision
from app.services import DocumentService


class ListHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines: list[dict] = []

    def emit(self, record):
        self.lines.append(json.loads(JsonFormatter().format(record)))


@pytest.fixture
def logs():
    handler = ListHandler()
    logger.addHandler(handler)
    previous = logger.level
    logger.setLevel(logging.INFO)
    yield handler.lines
    logger.removeHandler(handler)
    logger.setLevel(previous)


def make_service(mock_ocr, retriever, llm=None, **overrides):
    return DocumentService(mock_ocr, llm or FakeLLMAdapter(), retriever, Settings(**overrides))


def ocr_passes(result):
    return sum(e.startswith("OCR") for e in result.events)


def test_valid_document_does_not_retry(mock_ocr, retriever, logs):
    result = make_service(mock_ocr, retriever).process(DocumentInput(document_id="inv_001"))
    assert result.retry_count == 0 and ocr_passes(result) == 1
    assert not [l for l in logs if l["event"] == "retry_started"]


def test_invalid_document_retries_and_count_increments(mock_ocr, retriever, logs):
    result = make_service(mock_ocr, retriever, max_retries=3).process(
        DocumentInput(document_id="inv_005"))
    retries = [l["retry_count"] for l in logs if l["event"] == "retry_started"]
    failures = [l["retry_count"] for l in logs if l["event"] == "validation_failed"]
    assert retries == [1, 2, 3]
    assert failures == [0, 1, 2, 3]
    assert result.retry_count == 3 and result.validation_failures == 4


@pytest.mark.parametrize("max_retries", [0, 1, 2, 5])
def test_max_retry_count_is_respected(mock_ocr, retriever, max_retries):
    result = make_service(mock_ocr, retriever, max_retries=max_retries).process(
        DocumentInput(document_id="inv_005"))
    assert result.retry_count == max_retries
    assert ocr_passes(result) == max_retries + 1


def test_exhausted_retries_reach_supervisor_and_escalate(mock_ocr, retriever):
    result = make_service(mock_ocr, retriever).process(DocumentInput(document_id="inv_005"))
    assert result.events[-1].startswith("Supervisor: ESCALATE")
    assert result.review.reasons[0] == "validation failed after 2 retries"
    assert result.decision == ReviewDecision.ESCALATE


def test_max_retries_comes_from_environment(mock_ocr, retriever):
    settings = Settings.from_env({"MAX_RETRIES": "1"})
    result = DocumentService(mock_ocr, FakeLLMAdapter(), retriever, settings).process(
        DocumentInput(document_id="inv_005"))
    assert result.retry_count == 1


def test_supervisor_that_always_says_retry_cannot_loop(mock_ocr, retriever):
    always_retry = lambda payload: {"decision": "RETRY", "reasons": ["again"],  # noqa: E731
                                    "confidence": 0.5, "summary": "again"}
    from app.schemas.review import SupervisorReview

    llm = FakeLLMAdapter(handlers={SupervisorReview: always_retry})
    result = make_service(mock_ocr, retriever, llm=llm).process(
        DocumentInput(document_id="inv_001"))
    assert result.retry_count == 2
    assert result.decision == ReviewDecision.ESCALATE
    assert "retry budget exhausted" in result.review.reasons


def test_llm_that_never_returns_valid_output_cannot_loop(mock_ocr, retriever):
    llm = FakeLLMAdapter(scripted=["{not json"] * 100)
    result = make_service(mock_ocr, retriever, llm=llm, max_structuring_attempts=3).process(
        DocumentInput(document_id="inv_001"))
    assert result.structuring_failures == 3
    assert result.decision == ReviewDecision.ESCALATE


def test_recursion_limit_backstops_a_broken_router(mock_ocr, retriever, monkeypatch, logs):
    """Even if a routing bug ignored the counters, LangGraph's recursion limit stops it."""
    monkeypatch.setattr(workflow, "route_after_supervisor", lambda state, settings: "retry")
    result = make_service(mock_ocr, retriever).process(DocumentInput(document_id="inv_001"))
    assert result.status == "failed"
    assert "GraphRecursionError" in result.errors[0]
    assert any(l["event"] == "document_failed" for l in logs)


def test_log_trail_for_a_retried_document(mock_ocr, retriever, logs):
    result = make_service(mock_ocr, retriever).process(DocumentInput(document_id="inv_004"))
    trail = [l for l in logs if l.get("correlation_id") == result.correlation_id]
    assert [l["event"] for l in trail] == [
        "document_received",
        "ocr_completed", "validation_started", "rag_retrieval_completed", "validation_failed",
        "retry_started",
        "ocr_completed", "validation_started", "rag_retrieval_completed", "validation_passed",
        "structuring_completed", "supervisor_decision", "document_completed",
    ]
    assert all(l["document_id"] == "inv_004" for l in trail)
    failed = next(l for l in trail if l["event"] == "validation_failed")
    assert failed["retry_count"] == 0 and failed["retryable"] is True
    assert any("INVALID_DATE" in e for e in failed["errors"])
    assert trail[-1]["decision"] == "APPROVE" and trail[-1]["processing_ms"] > 0
