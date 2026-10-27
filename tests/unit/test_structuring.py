from decimal import Decimal

from app.adapters.llm import FakeLLMAdapter, _fake_document_record
from app.agents.structuring_agent import StructuringAgent
from app.agents.validation_agent import ValidationAgent


def make_state(settings, fields, confidence=0.96):
    validation, _ = ValidationAgent(None, settings).validate(fields, confidence)
    return {"document_id": "inv_x", "raw_text": "INVOICE ...", "validation": validation,
            "structuring_attempts": 0}


def test_structures_validated_fields(settings, good_fields):
    update = StructuringAgent(FakeLLMAdapter())(make_state(settings, good_fields))
    record = update["record"]
    assert record.total_amount == Decimal("1242.00")
    assert record.document_date.isoformat() == "2026-03-14"
    assert {e.label for e in record.extracted_entities} == {"subtotal", "tax"}
    assert not record.review_required
    assert update["structuring_attempts"] == 1


def test_malformed_llm_output_is_rejected(settings, good_fields):
    llm = FakeLLMAdapter(scripted=['{"document_id": "inv_x", "total_amount": "lots"}'])
    update = StructuringAgent(llm)(make_state(settings, good_fields))
    assert update["record"] is None
    assert "did not match DocumentRecord" in update["structuring_error"]
    assert update["errors"]


def test_llm_cannot_rewrite_validated_values(settings, good_fields):
    state = make_state(settings, good_fields)
    tampered = {"document_id": "inv_x", "document_type": "invoice",
                "customer_name": "Acme Corporation", "customer_id": "CUST-1001",
                "reference_number": "INV-20431", "document_date": "2026-03-14",
                "total_amount": "9999.00", "currency": "USD", "confidence": 0.96}
    update = StructuringAgent(FakeLLMAdapter(scripted=[tampered]))(state)
    assert update["record"] is None
    assert "changed validated field total_amount" in update["structuring_error"]


def test_llm_cannot_change_document_id(settings, good_fields):
    state = make_state(settings, good_fields)
    answer = _fake_document_record({"document_id": "someone_else", "confidence": 0.9,
                                    "normalized_fields": state["validation"].normalized_fields()})
    update = StructuringAgent(FakeLLMAdapter(scripted=[answer]))(state)
    assert update["record"] is None and "document_id" in update["structuring_error"]
