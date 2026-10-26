"""Agent 3 - Schema Structuring.

Asks the LLM for a `DocumentRecord` via structured output. The adapter rejects
anything that fails Pydantic validation; this agent additionally rejects records
where the LLM changed a value the validation agent already verified, so the
model can format and enrich but cannot silently rewrite facts.
"""

from __future__ import annotations

from decimal import Decimal

from app.adapters.llm import LLMAdapter, StructuredOutputError
from app.graph.state import PipelineState
from app.schemas.document import DocumentRecord

INSTRUCTIONS = """You convert OCR text from a business document into a DocumentRecord.
Use the validated `normalized_fields` as the source of truth; do not change them.
Use `ocr_text` only to fill optional fields (address, extracted_entities).
Copy `validation_warnings` into validation_errors and set review_required accordingly.
Set confidence to the provided `confidence` value."""

LOCKED_FIELDS = ["customer_id", "reference_number", "currency", "document_date", "total_amount"]


def _check_locked_fields(record: DocumentRecord, normalized: dict[str, str]) -> None:
    for field in LOCKED_FIELDS:
        expected = normalized.get(field)
        if expected is None:
            continue
        actual = getattr(record, field)
        if field == "total_amount":
            same = actual == Decimal(expected)
        elif field == "document_date":
            same = actual.isoformat() == expected
        else:
            same = str(actual) == expected
        if not same:
            raise StructuredOutputError(
                f"LLM changed validated field {field}: {expected!r} -> {actual!r}")


class StructuringAgent:
    def __init__(self, llm: LLMAdapter):
        self.llm = llm

    def __call__(self, state: PipelineState) -> dict:
        attempts = state.get("structuring_attempts", 0) + 1
        validation = state["validation"]
        normalized = validation.normalized_fields()
        payload = {
            "document_id": state["document_id"],
            "ocr_text": state.get("raw_text", ""),
            "normalized_fields": normalized,
            "validation_warnings": [str(w) for w in validation.warnings],
            "confidence": validation.confidence,
        }
        try:
            record = self.llm.invoke_structured(DocumentRecord, INSTRUCTIONS, payload)
            if record.document_id != state["document_id"]:
                raise StructuredOutputError("LLM changed document_id")
            _check_locked_fields(record, normalized)
        except StructuredOutputError as exc:
            message = str(exc).splitlines()[0][:200]
            return {
                "record": None,
                "structuring_attempts": attempts,
                "structuring_error": message,
                "errors": [f"structuring attempt {attempts}: {message}"],
                "events": [f"Structured output rejected (attempt {attempts}): {message}"],
            }
        return {
            "record": record,
            "structuring_attempts": attempts,
            "structuring_error": None,
            "events": [f"Structured record created (attempt {attempts})"],
        }
