"""Supervisor decision and the pipeline's final output."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.document import DocumentRecord
from app.schemas.extraction import ExtractedFields
from app.schemas.validation import ValidationResult


class ReviewDecision(str, Enum):
    APPROVE = "APPROVE"
    RETRY = "RETRY"
    ESCALATE = "ESCALATE"


# Higher = more conservative. Used so an LLM can never soften a rule decision.
DECISION_SEVERITY = {
    ReviewDecision.APPROVE: 0,
    ReviewDecision.RETRY: 1,
    ReviewDecision.ESCALATE: 2,
}


class SupervisorReview(BaseModel):
    decision: ReviewDecision
    reasons: list[str] = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    summary: str


class ProcessingResult(BaseModel):
    document_id: str
    # "completed": the pipeline reached a decision; "failed": it raised (decision is ESCALATE).
    status: Literal["completed", "failed"] = "completed"
    correlation_id: str = ""
    decision: ReviewDecision
    review: SupervisorReview
    record: DocumentRecord | None = None
    extracted_fields: ExtractedFields | None = None
    validation: ValidationResult | None = None
    retry_count: int = 0
    structuring_attempts: int = 0
    validation_failures: int = 0
    structuring_failures: int = 0
    processing_ms: float = 0.0
    events: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
