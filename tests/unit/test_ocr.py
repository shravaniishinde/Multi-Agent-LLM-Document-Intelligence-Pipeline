import pytest

from app.adapters.ocr import MockOCRAdapter, OCRError, TesseractOCRAdapter, build_ocr
from app.agents.ocr_agent import OCRAgent, extract_fields, normalize_text
from app.schemas.document import DocumentInput


def test_mock_ocr_returns_attempt_specific_scan(mock_ocr):
    first = mock_ocr.extract(DocumentInput(document_id="inv_004"), attempt=0)
    second = mock_ocr.extract(DocumentInput(document_id="inv_004"), attempt=1)
    assert first.confidence == 0.58 and "2O26" in first.text
    assert second.confidence == 0.91 and "2026-05-07" in second.text


def test_mock_ocr_reuses_last_scan_past_the_end(mock_ocr):
    result = mock_ocr.extract(DocumentInput(document_id="inv_005"), attempt=5)
    assert result.attempt == 5 and result.metadata["source"] == "inv_005.txt"


def test_mock_ocr_reads_plain_text_file(tmp_path):
    path = tmp_path / "scan.txt"
    path.write_text("INVOICE\nTotal: 10.00")
    result = MockOCRAdapter(tmp_path).extract(DocumentInput(document_id="x", file_path=str(path)))
    assert result.confidence == 1.0 and "Total" in result.text


def test_mock_ocr_unknown_document_raises(mock_ocr):
    with pytest.raises(OCRError):
        mock_ocr.extract(DocumentInput(document_id="does-not-exist"))


def test_normalize_text_collapses_noise():
    raw = "INVOICE\x0c\n\n   Total:\t\t 1,000.00   \n \nＤate: 2026-01-01"
    assert normalize_text(raw) == "INVOICE\nTotal: 1,000.00\nDate: 2026-01-01"


def test_extract_fields_handles_label_synonyms_and_currency_symbol():
    text = "RECEIPT\nReceipt No: RCT-1\nBilled To: Globex GmbH\nClient ID: CUST-1002\nTotal: € 9.00"
    fields = extract_fields(text)
    assert fields.document_type == "receipt"
    assert fields.reference_number == "RCT-1"
    assert fields.customer_name == "Globex GmbH"
    assert fields.customer_id == "CUST-1002"
    assert fields.currency == "EUR"


def test_extract_fields_first_label_wins_and_skips_empty_values():
    fields = extract_fields("INVOICE\nTotal:\nTotal: 5.00\nTotal: 7.00")
    assert fields.total_amount == "5.00"


def test_ocr_agent_reports_failure_without_crashing(mock_ocr):
    update = OCRAgent(mock_ocr)({"document": DocumentInput(document_id="missing"), "retry_count": 0})
    assert update["ocr_result"] is None
    assert update["extracted_fields"].present() == {}
    assert update["errors"] and update["events"][0].startswith("OCR failed")


def test_build_ocr_defaults_to_mock(settings):
    assert build_ocr(settings).name == "mock"


def test_tesseract_adapter_text_passthrough(tmp_path):
    """Only runs when the optional pytesseract/Pillow packages are installed."""
    pytest.importorskip("pytesseract")
    pytest.importorskip("PIL")
    path = tmp_path / "scan.txt"
    path.write_text("INVOICE")
    result = TesseractOCRAdapter().extract(DocumentInput(document_id="t", file_path=str(path)))
    assert result.engine == "text-passthrough"


def test_ocr_agent_uses_retry_count_as_attempt(mock_ocr):
    update = OCRAgent(mock_ocr)({"document": DocumentInput(document_id="inv_004"), "retry_count": 1})
    assert update["ocr_result"].attempt == 1
    assert update["extracted_fields"].document_date == "2026-05-07"
