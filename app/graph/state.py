"""Typed LangGraph state shared by all agents.

Nodes return partial updates. `events` and `errors` use an additive reducer so
every node appends to an audit trail instead of overwriting it.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from app.rag.retriever import RetrievedChunk
from app.schemas.document import DocumentInput, DocumentRecord
from app.schemas.extraction import ExtractedFields, OCRResult
from app.schemas.review import ReviewDecision, SupervisorReview
from app.schemas.validation import ValidationResult


class PipelineState(TypedDict, total=False):
    document_id: str
    document: DocumentInput
    raw_text: str
    ocr_result: OCRResult | None
    extracted_fields: ExtractedFields | None
    validation: ValidationResult | None
    retrieved_context: list[RetrievedChunk]
    record: DocumentRecord | None
    structuring_attempts: int
    structuring_error: str | None
    retry_count: int
    review: SupervisorReview | None
    decision: ReviewDecision | None
    errors: Annotated[list[str], operator.add]
    events: Annotated[list[str], operator.add]
    metadata: dict[str, Any]


def initial_state(document: DocumentInput) -> PipelineState:
    return PipelineState(
        document_id=document.document_id,
        document=document,
        retry_count=0,
        structuring_attempts=0,
        retrieved_context=[],
        errors=[],
        events=[],
        metadata={},
    )
