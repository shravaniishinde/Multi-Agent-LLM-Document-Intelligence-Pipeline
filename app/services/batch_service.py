"""Batch processing on top of LangGraph's native `batch` (bounded concurrency).

A document that raises is recorded as ESCALATE with the exception, so one bad
file never aborts the batch.
"""

from __future__ import annotations

from collections import Counter

from pydantic import BaseModel, Field

from app.graph.state import initial_state
from app.schemas.document import DocumentInput
from app.schemas.review import ProcessingResult, ReviewDecision, SupervisorReview
from app.services.document_service import DocumentService, to_result


class BatchSummary(BaseModel):
    total: int
    by_decision: dict[str, int]
    total_retries: int
    results: list[ProcessingResult] = Field(default_factory=list)


class BatchService:
    def __init__(self, service: DocumentService, max_concurrency: int = 4):
        self.service = service
        self.max_concurrency = max_concurrency

    def process_many(self, documents: list[DocumentInput]) -> BatchSummary:
        states = [initial_state(d) for d in documents]
        config = {**self.service.config, "max_concurrency": self.max_concurrency}
        outputs = self.service.graph.batch(states, config=config, return_exceptions=True)

        results: list[ProcessingResult] = []
        for document, output in zip(documents, outputs):
            if isinstance(output, Exception):
                review = SupervisorReview(
                    decision=ReviewDecision.ESCALATE, confidence=0.0,
                    reasons=[f"pipeline error: {type(output).__name__}: {output}"],
                    summary="Escalated: processing error")
                results.append(ProcessingResult(document_id=document.document_id,
                                                decision=review.decision, review=review,
                                                errors=[str(output)]))
            else:
                results.append(to_result(output))

        counts = Counter(r.decision.value for r in results)
        return BatchSummary(
            total=len(results),
            by_decision={d.value: counts.get(d.value, 0) for d in ReviewDecision},
            total_retries=sum(r.retry_count for r in results),
            results=results,
        )
