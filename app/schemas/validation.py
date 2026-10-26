"""Typed validation results."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class IssueCode(str, Enum):
    MISSING_FIELD = "MISSING_FIELD"
    INVALID_DATE = "INVALID_DATE"
    FUTURE_DATE = "FUTURE_DATE"
    INVALID_AMOUNT = "INVALID_AMOUNT"
    NON_POSITIVE_AMOUNT = "NON_POSITIVE_AMOUNT"
    AMOUNT_MISMATCH = "AMOUNT_MISMATCH"
    INVALID_CURRENCY = "INVALID_CURRENCY"
    INVALID_FORMAT = "INVALID_FORMAT"
    # Contextual checks backed by RAG reference data.
    CUSTOMER_MISMATCH = "CUSTOMER_MISMATCH"
    CURRENCY_POLICY = "CURRENCY_POLICY"
    UNKNOWN_CUSTOMER = "UNKNOWN_CUSTOMER"
    APPROVAL_LIMIT_EXCEEDED = "APPROVAL_LIMIT_EXCEEDED"
    OLD_DOCUMENT = "OLD_DOCUMENT"


class ValidationIssue(BaseModel):
    field: str
    code: IssueCode
    message: str
    # True when re-running OCR could plausibly fix it (noise, not bad data).
    retryable: bool = False

    def __str__(self) -> str:
        return f"{self.field}: {self.code.value} - {self.message}"


class FieldResult(BaseModel):
    field: str
    value: str | None
    valid: bool
    normalized: str | None = None
    message: str | None = None


class ValidationResult(BaseModel):
    valid: bool
    errors: list[ValidationIssue] = Field(default_factory=list)
    warnings: list[ValidationIssue] = Field(default_factory=list)
    field_results: list[FieldResult] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)

    @property
    def retryable(self) -> bool:
        return bool(self.errors) and any(e.retryable for e in self.errors)

    def normalized_fields(self) -> dict[str, str]:
        return {
            r.field: r.normalized
            for r in self.field_results
            if r.valid and r.normalized is not None
        }
