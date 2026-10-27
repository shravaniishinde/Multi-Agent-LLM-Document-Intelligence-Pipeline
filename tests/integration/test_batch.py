import json
import sys
import threading
import time

import pytest

from app.adapters.llm import FakeLLMAdapter
from app.config import DATA_DIR
from app.schemas.document import DocumentInput
from app.services import BatchService, DocumentService, discover_documents


@pytest.fixture
def service(mock_ocr, retriever, settings):
    return DocumentService(mock_ocr, FakeLLMAdapter(), retriever, settings)


def test_multiple_successful_documents(service):
    docs = [DocumentInput(document_id=d) for d in ("inv_001", "inv_002", "inv_004", "inv_008")]
    summary = BatchService(service).process_many(docs)
    assert (summary.total, summary.successful, summary.escalated, summary.failed) == (4, 4, 0, 0)
    assert summary.total_retries == 2          # inv_004 and inv_008 each re-scan once
    assert summary.validation_failures == 1    # inv_004's first pass
    assert summary.average_processing_ms > 0


def test_mixed_outcomes_over_sample_set(service):
    summary = BatchService(service).process_many(discover_documents(DATA_DIR / "sample_documents"))
    assert (summary.total, summary.successful, summary.escalated, summary.failed) == (8, 4, 4, 0)
    assert summary.total_retries == 4
    assert service.metrics.snapshot().documents_processed == 8


def test_concurrency_is_bounded(service, monkeypatch):
    active, peak, lock = 0, 0, threading.Lock()
    original = service.process

    def tracking_process(document):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.02)
        try:
            return original(document)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(service, "process", tracking_process)
    docs = [DocumentInput(document_id="inv_001") for _ in range(8)]
    BatchService(service, max_concurrency=2).process_many(docs)
    assert peak == 2


def test_exception_outside_graph_is_contained(service, monkeypatch):
    original = service.process

    def explode_on_second(document):
        if document.document_id == "inv_002":
            raise MemoryError("boom")
        return original(document)

    monkeypatch.setattr(service, "process", explode_on_second)
    docs = [DocumentInput(document_id=d) for d in ("inv_001", "inv_002", "inv_003")]
    summary = BatchService(service).process_many(docs)
    assert [r.status for r in summary.results] == ["completed", "failed", "completed"]
    assert summary.failed == 1 and summary.total == 3


def test_discover_documents_without_manifest(tmp_path):
    for name in ("b.txt", "a.png", "notes.md", "c.PDF"):
        (tmp_path / name).write_text("x")
    docs = discover_documents(tmp_path)
    assert [d.document_id for d in docs] == ["a", "b", "c"]
    assert docs[0].file_path.endswith("a.png")


def test_discover_documents_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        discover_documents(tmp_path / "nope")


def test_run_batch_script_writes_results(tmp_path, monkeypatch, capsys):
    sys.path.insert(0, str(DATA_DIR.parent / "scripts"))
    import run_batch

    out = tmp_path / "results.json"
    monkeypatch.setattr(sys, "argv", ["run_batch.py", "--input", str(DATA_DIR / "sample_documents"),
                                      "--output", str(out), "--concurrency", "3"])
    run_batch.main()
    report = json.loads(out.read_text())
    assert report["summary"]["total"] == 8 and report["summary"]["failed"] == 0
    assert len(report["results"]) == 8
    assert "Total documents" in capsys.readouterr().out
