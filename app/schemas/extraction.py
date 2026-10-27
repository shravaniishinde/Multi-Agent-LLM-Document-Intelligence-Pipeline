"""OCR output and the raw (unvalidated) fields pulled from it."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class OCRResult(BaseModel):
    text: str
    confidence: float = Field(ge=0, le=1)
    engine: str
    page_count: int = Field(default=1, ge=1)
    attempt: int = Field(default=0, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExtractedFields(BaseModel):
    """Candidate key/value pairs found in OCR text.

    Every field is a raw string on purpose: this is what the scan *says*, before
    any parsing. The validation agent decides whether each value is usable.
    """

    document_type: str | None = None
    reference_number: str | None = None
    customer_name: str | None = None
    customer_id: str | None = None
    document_date: str | None = None
    subtotal: str | None = None
    tax: str | None = None
    total_amount: str | None = None
    currency: str | None = None
    address: str | None = None

    def present(self) -> dict[str, str]:
        return {k: v for k, v in self.model_dump().items() if v not in (None, "")}
