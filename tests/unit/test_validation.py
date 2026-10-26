from datetime import date, timedelta
from decimal import Decimal

import pytest

from app.agents.validation_agent import ValidationAgent, parse_amount, parse_date
from app.schemas.validation import IssueCode


def codes(result):
    return {(e.field, e.code) for e in result.errors}


@pytest.fixture
def agent(retriever, settings):
    return ValidationAgent(retriever, settings)


@pytest.fixture
def offline_agent(settings):
    """No retriever: only the deterministic checks."""
    return ValidationAgent(None, settings)


@pytest.mark.parametrize("raw,expected", [
    ("2026-03-14", date(2026, 3, 14)),
    ("14/03/2026", date(2026, 3, 14)),
    ("March 14, 2026", date(2026, 3, 14)),
    ("14 Mar 2026", date(2026, 3, 14)),
    ("2O26-03-14", None),
    ("2026-13-45", None),
])
def test_parse_date(raw, expected):
    assert parse_date(raw) == expected


@pytest.mark.parametrize("raw,expected", [
    ("1,242.00", Decimal("1242.00")),
    ("USD 1,242.00", Decimal("1242.00")),
    ("€ 3,480", Decimal("3480.00")),
    ("12", Decimal("12.00")),
    ("4,3l0.00", None),
    ("1,24,2.00", None),
    ("", None),
])
def test_parse_amount(raw, expected):
    assert parse_amount(raw) == expected


def test_good_fields_pass(agent, good_fields):
    result, context = agent.validate(good_fields, ocr_confidence=0.96)
    assert result.valid and not result.errors and not result.warnings
    assert result.confidence == 0.96
    assert result.normalized_fields()["total_amount"] == "1242.00"
    assert any(c.metadata.get("customer_id") == "CUST-1001" for c in context)


def test_missing_field_retryable_only_on_low_confidence(offline_agent, good_fields):
    fields = good_fields.model_copy(update={"customer_id": None})
    clean, _ = offline_agent.validate(fields, ocr_confidence=0.95)
    noisy, _ = offline_agent.validate(fields, ocr_confidence=0.5)
    assert codes(clean) == {("customer_id", IssueCode.MISSING_FIELD)}
    assert not clean.retryable
    assert noisy.retryable


def test_format_errors_are_retryable(offline_agent, good_fields):
    fields = good_fields.model_copy(update={
        "document_date": "2O26-03-14", "total_amount": "1,24O.00", "currency": "U5D",
        "customer_id": "CUST-10O1", "reference_number": "IVN-1"})
    result, _ = offline_agent.validate(fields, ocr_confidence=0.95)
    assert codes(result) == {
        ("document_date", IssueCode.INVALID_DATE), ("total_amount", IssueCode.INVALID_AMOUNT),
        ("currency", IssueCode.INVALID_CURRENCY), ("customer_id", IssueCode.INVALID_FORMAT),
        ("reference_number", IssueCode.INVALID_FORMAT)}
    assert all(e.retryable for e in result.errors)
    assert result.confidence == pytest.approx(0.95 / 6, abs=0.001)


def test_future_date_and_non_positive_amount(offline_agent, good_fields):
    future = (date.today() + timedelta(days=10)).isoformat()
    fields = good_fields.model_copy(update={"document_date": future, "total_amount": "0.00",
                                            "subtotal": None, "tax": None})
    result, _ = offline_agent.validate(fields, ocr_confidence=0.95)
    assert codes(result) == {("document_date", IssueCode.FUTURE_DATE),
                             ("total_amount", IssueCode.NON_POSITIVE_AMOUNT)}


def test_inconsistent_totals(offline_agent, good_fields):
    fields = good_fields.model_copy(update={"total_amount": "1,300.00"})
    result, _ = offline_agent.validate(fields, ocr_confidence=0.95)
    assert codes(result) == {("total_amount", IssueCode.AMOUNT_MISMATCH)}
    assert "total_amount" not in result.normalized_fields()


def test_rag_currency_policy_violation(agent, good_fields):
    fields = good_fields.model_copy(update={"currency": "EUR"})
    result, _ = agent.validate(fields, ocr_confidence=0.95)
    assert codes(result) == {("currency", IssueCode.CURRENCY_POLICY)}
    assert not result.retryable


def test_rag_name_does_not_match_customer_id(agent, good_fields):
    fields = good_fields.model_copy(update={"customer_name": "Globex GmbH"})
    result, _ = agent.validate(fields, ocr_confidence=0.95)
    assert ("customer_name", IssueCode.CUSTOMER_MISMATCH) in codes(result)


def test_rag_known_name_under_wrong_id(agent, good_fields):
    fields = good_fields.model_copy(update={"customer_id": "CUST-9999"})
    result, context = agent.validate(fields, ocr_confidence=0.95)
    assert codes(result) == {("customer_id", IssueCode.CUSTOMER_MISMATCH)}
    assert "CUST-1001" in result.errors[0].message


def test_rag_unknown_customer_is_a_warning(agent, good_fields):
    fields = good_fields.model_copy(update={"customer_id": "CUST-9999",
                                            "customer_name": "Brand New Client"})
    result, _ = agent.validate(fields, ocr_confidence=0.95)
    assert result.valid
    assert [w.code for w in result.warnings] == [IssueCode.UNKNOWN_CUSTOMER]


def test_rag_approval_limit_warning(agent, good_fields):
    fields = good_fields.model_copy(update={"subtotal": None, "tax": None,
                                            "total_amount": "30,000.00"})
    result, _ = agent.validate(fields, ocr_confidence=0.95)
    assert result.valid
    assert [w.code for w in result.warnings] == [IssueCode.APPROVAL_LIMIT_EXCEEDED]
