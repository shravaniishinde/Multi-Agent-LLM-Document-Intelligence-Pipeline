from datetime import date, timedelta
from decimal import Decimal

import pytest
from pydantic import ValidationError

from app.schemas.document import DocumentRecord
from app.schemas.validation import IssueCode, ValidationIssue, ValidationResult


def record(**overrides) -> dict:
    data = dict(document_id="d1", document_type="invoice", customer_name="Acme Corporation",
                customer_id="CUST-1001", reference_number="INV-20431",
                document_date="2026-03-14", total_amount="1242.00", currency="usd",
                confidence=0.9)
    data.update(overrides)
    return data


def test_valid_record_is_coerced():
    r = DocumentRecord.model_validate(record())
    assert r.currency == "USD"
    assert r.total_amount == Decimal("1242.00")
    assert r.document_date == date(2026, 3, 14)


@pytest.mark.parametrize("override", [
    {"customer_id": "1001"},
    {"reference_number": "20431"},
    {"currency": "XYZ"},
    {"total_amount": "-5"},
    {"total_amount": "1.234"},
    {"confidence": 1.5},
    {"document_type": "letter"},
    {"document_date": (date.today() + timedelta(days=3)).isoformat()},
])
def test_invalid_record_values_are_rejected(override):
    with pytest.raises(ValidationError):
        DocumentRecord.model_validate(record(**override))


def test_validation_errors_require_review_flag():
    with pytest.raises(ValidationError, match="review_required"):
        DocumentRecord.model_validate(record(validation_errors=["x"], review_required=False))
    assert DocumentRecord.model_validate(record(validation_errors=["x"], review_required=True))


def test_validation_result_retryable_only_with_retryable_error():
    issue = lambda r: ValidationIssue(field="f", code=IssueCode.INVALID_DATE,  # noqa: E731
                                      message="m", retryable=r)
    assert ValidationResult(valid=False, errors=[issue(True)], confidence=0.5).retryable
    assert not ValidationResult(valid=False, errors=[issue(False)], confidence=0.5).retryable
    assert not ValidationResult(valid=True, confidence=0.5).retryable
