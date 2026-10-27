"""Document input and the final structured record.

`DocumentRecord` is the contract every downstream consumer relies on. Whatever an
LLM returns must pass these validators before it is accepted, which is the main
reason to use structured output instead of free-form JSON.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator

SUPPORTED_CURRENCIES = frozenset({"USD", "EUR", "GBP", "INR", "JPY", "CAD", "AUD"})

CUSTOMER_ID_PATTERN = r"^CUST-\d{4}$"
REFERENCE_NUMBER_PATTERN = r"^(INV|RCT|PO)-\d{4,}$"


class DocumentType(str, Enum):
    INVOICE = "invoice"
    RECEIPT = "receipt"
    PURCHASE_ORDER = "purchase_order"
    UNKNOWN = "unknown"


class DocumentInput(BaseModel):
    """A document submitted to the pipeline (a scan on disk, or a fixture id)."""

    document_id: str = Field(min_length=1)
    file_path: str | None = None


class ExtractedEntity(BaseModel):
    label: str
    value: str


class DocumentRecord(BaseModel):
    document_id: str = Field(min_length=1)
    document_type: DocumentType
    customer_name: str = Field(min_length=2)
    customer_id: str = Field(pattern=CUSTOMER_ID_PATTERN)
    reference_number: str = Field(pattern=REFERENCE_NUMBER_PATTERN)
    document_date: date
    total_amount: Decimal = Field(gt=0, decimal_places=2)
    currency: str
    address: str | None = None
    extracted_entities: list[ExtractedEntity] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    validation_errors: list[str] = Field(default_factory=list)
    review_required: bool = False

    @field_validator("currency")
    @classmethod
    def _currency_supported(cls, value: str) -> str:
        value = value.upper()
        if value not in SUPPORTED_CURRENCIES:
            raise ValueError(f"unsupported currency {value!r}")
        return value

    @field_validator("document_date")
    @classmethod
    def _date_not_in_future(cls, value: date) -> date:
        if value > date.today():
            raise ValueError("document_date is in the future")
        return value

    @model_validator(mode="after")
    def _errors_imply_review(self) -> "DocumentRecord":
        if self.validation_errors and not self.review_required:
            raise ValueError("records with validation_errors must set review_required")
        return self
