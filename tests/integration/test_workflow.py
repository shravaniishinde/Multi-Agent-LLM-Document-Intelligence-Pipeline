"""End-to-end LangGraph runs over the fixture documents (fully offline)."""

import json

import pytest

from app.adapters.llm import FakeLLMAdapter
from app.config import DATA_DIR, Settings
from app.schemas.document import DocumentInput
from app.schemas.review import ReviewDecision
from app.services import BatchService, DocumentService

MANIFEST = json.loads((DATA_DIR / "sample_documents" / "manifest.json").read_text())


@pytest.fixture
def service(mock_ocr, retriever, settings):
    return DocumentService(mock_ocr, FakeLLMAdapter(), retriever, settings)


def node_sequence(result):
    """Compress events into the node path the graph took."""
    prefixes = {"OCR": "ocr", "Validation": "validate", "Structur": "structure",
                "Supervisor": "supervisor", "Retry": "retry"}
    return [v for e in result.events for k, v in prefixes.items() if e.startswith(k)]


@pytest.mark.parametrize("doc_id", sorted(MANIFEST))
def test_fixture_documents_route_as_labelled(service, doc_id):
    result = service.process(DocumentInput(document_id=doc_id))
    assert result.decision.value == MANIFEST[doc_id]["expected_decision"]
    assert result.retry_count == MANIFEST[doc_id]["expected_retries"]


def test_successful_end_to_end_record(service):
    result = service.process(DocumentInput(document_id="inv_001"))
    assert node_sequence(result) == ["ocr", "validate", "structure", "supervisor"]
    assert result.record.customer_id == "CUST-1001"
    assert str(result.record.total_amount) == "1242.00"
    assert result.record.currency == "USD"
    assert result.review.reasons


def test_validation_retry_then_success(service):
    result = service.process(DocumentInput(document_id="inv_004"))
    assert node_sequence(result) == ["ocr", "validate", "retry", "ocr", "validate",
                                     "structure", "supervisor"]
    assert result.decision == ReviewDecision.APPROVE


def test_retries_are_bounded_then_escalated(service, settings):
    result = service.process(DocumentInput(document_id="inv_005"))
    assert result.retry_count == settings.max_retries
    assert node_sequence(result).count("ocr") == settings.max_retries + 1
    assert result.decision == ReviewDecision.ESCALATE
    assert result.record is None


def test_zero_retry_budget_escalates_immediately(mock_ocr, retriever):
    settings = Settings(max_retries=0)
    result = DocumentService(mock_ocr, FakeLLMAdapter(), retriever, settings).process(
        DocumentInput(document_id="inv_004"))
    assert result.retry_count == 0
    assert result.decision == ReviewDecision.ESCALATE


def test_missing_field_on_clean_scan_skips_retry(service):
    result = service.process(DocumentInput(document_id="inv_003"))
    assert node_sequence(result) == ["ocr", "validate", "supervisor"]
    assert result.decision == ReviewDecision.ESCALATE


def test_supervisor_retry_loop(service):
    result = service.process(DocumentInput(document_id="inv_008"))
    assert node_sequence(result) == ["ocr", "validate", "structure", "supervisor", "retry",
                                     "ocr", "validate", "structure", "supervisor"]
    assert result.decision == ReviewDecision.APPROVE


def test_malformed_llm_output_is_retried_then_accepted(mock_ocr, retriever, settings):
    llm = FakeLLMAdapter(scripted=["```json\n{oops"])
    result = DocumentService(mock_ocr, llm, retriever, settings).process(
        DocumentInput(document_id="inv_001"))
    assert result.structuring_attempts == 2
    assert result.decision == ReviewDecision.APPROVE
    assert any("structuring attempt 1" in e for e in result.errors)


def test_persistently_malformed_llm_output_escalates(mock_ocr, retriever, settings):
    llm = FakeLLMAdapter(scripted=["{bad"] * settings.max_structuring_attempts)
    result = DocumentService(mock_ocr, llm, retriever, settings).process(
        DocumentInput(document_id="inv_001"))
    assert result.record is None
    assert result.structuring_attempts == settings.max_structuring_attempts
    assert result.decision == ReviewDecision.ESCALATE
    assert "structured output failed" in result.review.reasons[0]


def test_unknown_document_escalates_without_crashing(service):
    result = service.process(DocumentInput(document_id="not_in_manifest"))
    assert result.decision == ReviewDecision.ESCALATE
    assert any(e.startswith("ocr:") for e in result.errors)


def test_batch_processing_summary(service):
    docs = [DocumentInput(document_id=d) for d in sorted(MANIFEST)]
    summary = BatchService(service, max_concurrency=3).process_many(docs)
    assert summary.total == len(MANIFEST)
    assert [r.document_id for r in summary.results] == sorted(MANIFEST)
    expected = sum(m["expected_decision"] == "APPROVE" for m in MANIFEST.values())
    assert summary.successful == summary.by_decision["APPROVE"] == expected
    assert summary.escalated == len(MANIFEST) - expected and summary.failed == 0
    assert summary.total_retries == sum(m["expected_retries"] for m in MANIFEST.values())


def test_batch_isolates_failures(service, monkeypatch):
    docs = [DocumentInput(document_id=d) for d in ("inv_001", "inv_002", "inv_003")]
    original_run = service.run

    def flaky_run(document, correlation_id=""):
        if document.document_id == "inv_002":
            raise RuntimeError("disk on fire")
        return original_run(document, correlation_id)

    monkeypatch.setattr(service, "run", flaky_run)
    summary = BatchService(service).process_many(docs)
    assert [r.status for r in summary.results] == ["completed", "failed", "completed"]
    assert summary.results[0].decision == ReviewDecision.APPROVE
    assert summary.results[1].decision == ReviewDecision.ESCALATE
    assert "disk on fire" in summary.results[1].review.reasons[0]
    assert (summary.successful, summary.escalated, summary.failed) == (1, 1, 1)
