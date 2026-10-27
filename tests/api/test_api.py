import pytest
from fastapi.testclient import TestClient

from app.adapters.llm import FakeLLMAdapter
from app.main import create_app
from app.services import DocumentService


@pytest.fixture
def service(mock_ocr, retriever, settings):
    return DocumentService(mock_ocr, FakeLLMAdapter(), retriever, settings)


@pytest.fixture
def client(service):
    with TestClient(create_app(service=service)) as c:  # runs the lifespan
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["offline_mode"] is True
    assert body["components"] == {"ocr": "mock", "llm": "fake",
                                  "vector_store": "InMemoryVectorStore"}
    assert body["max_retries"] == 2


def test_process_approved_document(client):
    r = client.post("/documents/process", json={"document_id": "inv_001"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "completed" and body["decision"] == "APPROVE"
    assert body["document_type"] == "invoice"
    assert body["extracted_fields"]["customer_id"] == "CUST-1001"
    assert body["validation"]["valid"] is True and body["confidence"] == 0.96
    assert body["record"]["total_amount"] == "1242.00"
    assert body["retry_count"] == 0 and body["errors"] == [] and body["correlation_id"]


def test_process_escalated_document_exposes_errors(client):
    body = client.post("/documents/process", json={"document_id": "inv_005"}).json()
    assert body["decision"] == "ESCALATE" and body["retry_count"] == 2
    assert body["validation"]["valid"] is False
    assert {e["code"] for e in body["validation"]["errors"]} >= {"INVALID_DATE", "INVALID_AMOUNT"}
    assert any("INVALID_CURRENCY" in e for e in body["errors"])
    assert body["record"] is None


def test_process_warning_document(client):
    body = client.post("/documents/process", json={"document_id": "inv_006"}).json()
    assert body["decision"] == "ESCALATE"
    assert any("APPROVAL_LIMIT_EXCEEDED" in w for w in body["warnings"])


def test_process_file_inside_documents_dir(client):
    body = client.post("/documents/process",
                       json={"document_id": "upload_1", "file_path": "inv_001.txt"}).json()
    assert body["decision"] == "APPROVE" and body["document_id"] == "upload_1"


@pytest.mark.parametrize("payload,status", [
    ({}, 422),                                                  # missing document_id
    ({"document_id": ""}, 422),
    ({"document_id": "../etc/passwd"}, 422),                    # id pattern
    ({"document_id": "x", "file_path": "../../README.md"}, 400),  # escapes DOCUMENTS_DIR
    ({"document_id": "x", "file_path": "missing.txt"}, 404),
])
def test_invalid_process_requests(client, payload, status):
    assert client.post("/documents/process", json=payload).status_code == status


def test_pipeline_error_is_reported_not_raised(client, service, monkeypatch):
    def broken(document, correlation_id=""):
        raise RuntimeError("OCR engine crashed")

    monkeypatch.setattr(service, "run", broken)
    body = client.post("/documents/process", json={"document_id": "inv_001"}).json()
    assert body["status"] == "failed" and body["decision"] == "ESCALATE"
    assert "OCR engine crashed" in body["errors"][0]


def test_batch(client):
    r = client.post("/documents/batch", json={
        "documents": [{"document_id": d} for d in ("inv_001", "inv_003", "inv_004", "nope")],
        "max_concurrency": 2})
    assert r.status_code == 200
    body = r.json()
    assert (body["total"], body["successful"], body["escalated"], body["failed"]) == (4, 2, 2, 0)
    # inv_004 re-scans once; "nope" has no scan, OCR failure is retried until the budget is used.
    assert body["total_retries"] == 3
    assert [d["document_id"] for d in body["results"]] == ["inv_001", "inv_003", "inv_004", "nope"]


@pytest.mark.parametrize("payload", [
    {"documents": []},
    {"documents": [{"document_id": "inv_001"}], "max_concurrency": 0},
    {"documents": [{"document_id": "inv_001"}] * 101},
])
def test_invalid_batch_requests(client, payload):
    assert client.post("/documents/batch", json=payload).status_code == 422


def test_document_lookup(client):
    assert client.get("/documents/inv_004").status_code == 404
    client.post("/documents/process", json={"document_id": "inv_004"})
    body = client.get("/documents/inv_004").json()
    assert body["decision"] == "APPROVE" and body["retry_count"] == 1


def test_metrics(client):
    empty = client.get("/metrics").json()
    assert empty["documents_processed"] == 0 and empty["average_latency_ms"] == 0
    client.post("/documents/process", json={"document_id": "inv_001"})
    client.post("/documents/batch", json={"documents": [{"document_id": "inv_004"},
                                                        {"document_id": "inv_005"}]})
    m = client.get("/metrics").json()
    assert m["documents_processed"] == 3
    assert (m["successful"], m["escalated"], m["failed"]) == (2, 1, 0)
    assert m["retries"] == 3                       # inv_004: 1, inv_005: 2
    assert m["validation_failures"] == 4           # inv_004: 1, inv_005: 3
    assert m["average_latency_ms"] > 0
