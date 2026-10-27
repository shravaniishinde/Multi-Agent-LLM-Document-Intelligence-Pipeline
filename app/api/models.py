"""Request/response models for the HTTP API (separate from the pipeline schemas)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.document import DocumentRecord
from app.schemas.review import ProcessingResult, ReviewDecision
from app.services.batch_service import BatchSummary

DOCUMENT_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$"


class ProcessRequest(BaseModel):
    document_id: str = Field(pattern=DOCUMENT_ID_PATTERN, examples=["inv_004"])
    file_path: str | None = Field(
        default=None,
        description="Path relative to the server's DOCUMENTS_DIR. Omit to resolve the "
                    "document by id (mock OCR sample set).")


class BatchRequest(BaseModel):
    documents: list[ProcessRequest] = Field(min_length=1, max_length=100)
    max_concurrency: int | None = Field(default=None, ge=1, le=32)


class IssueOut(BaseModel):
    field: str
    code: str
    message: str
    retryable: bool


class ValidationOut(BaseModel):
    valid: bool
    confidence: float
    errors: list[IssueOut]
    warnings: list[IssueOut]


class DocumentResponse(BaseModel):
    document_id: str
    correlation_id: str
    status: Literal["completed", "failed"]
    decision: ReviewDecision
    document_type: str | None
    extracted_fields: dict[str, str]
    validation: ValidationOut | None
    confidence: float
    retry_count: int
    review_reasons: list[str]
    record: DocumentRecord | None
    errors: list[str]
    warnings: list[str]
    events: list[str]
    processing_ms: float

    @classmethod
    def from_result(cls, r: ProcessingResult) -> "DocumentResponse":
        v = r.validation
        issue = lambda i: IssueOut(field=i.field, code=i.code.value,  # noqa: E731
                                   message=i.message, retryable=i.retryable)
        fields = r.extracted_fields.present() if r.extracted_fields else {}
        if r.record is not None:
            doc_type, confidence = r.record.document_type.value, r.record.confidence
        else:
            doc_type, confidence = fields.get("document_type"), v.confidence if v else 0.0
        return cls(
            document_id=r.document_id,
            correlation_id=r.correlation_id,
            status=r.status,
            decision=r.decision,
            document_type=doc_type,
            extracted_fields=fields,
            validation=ValidationOut(valid=v.valid, confidence=v.confidence,
                                     errors=[issue(e) for e in v.errors],
                                     warnings=[issue(w) for w in v.warnings]) if v else None,
            confidence=confidence,
            retry_count=r.retry_count,
            review_reasons=r.review.reasons,
            record=r.record,
            errors=r.errors + ([str(e) for e in v.errors] if v else []),
            warnings=[str(w) for w in v.warnings] if v else [],
            events=r.events,
            processing_ms=r.processing_ms,
        )


class BatchResponse(BaseModel):
    total: int
    successful: int
    escalated: int
    failed: int
    total_retries: int
    validation_failures: int
    average_processing_ms: float
    wall_clock_ms: float
    results: list[DocumentResponse]

    @classmethod
    def from_summary(cls, s: BatchSummary) -> "BatchResponse":
        data = s.model_dump(exclude={"results", "by_decision"})
        return cls(**data, results=[DocumentResponse.from_result(r) for r in s.results])


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str
    offline_mode: bool
    components: dict[str, str]
    max_retries: int
