"""Single-document entry point: builds the graph once, runs documents through it."""

from __future__ import annotations

from app.adapters.llm import LLMAdapter, build_llm
from app.adapters.ocr import OCRAdapter, build_ocr
from app.config import Settings
from app.graph.state import PipelineState, initial_state
from app.graph.workflow import build_workflow, recursion_limit
from app.rag.knowledge_base import build_retriever
from app.rag.retriever import Retriever
from app.schemas.document import DocumentInput
from app.schemas.review import ProcessingResult, ReviewDecision, SupervisorReview


class DocumentService:
    def __init__(self, ocr: OCRAdapter, llm: LLMAdapter, retriever: Retriever | None,
                 settings: Settings):
        self.settings = settings
        self.graph = build_workflow(ocr, llm, retriever, settings)
        self.config = {"recursion_limit": recursion_limit(settings)}

    def run(self, document: DocumentInput) -> PipelineState:
        """Run the graph and return the raw final state."""
        return self.graph.invoke(initial_state(document), config=self.config)

    def process(self, document: DocumentInput) -> ProcessingResult:
        return to_result(self.run(document))


def to_result(state: PipelineState) -> ProcessingResult:
    review = state.get("review") or SupervisorReview(
        decision=ReviewDecision.ESCALATE, reasons=["pipeline ended without a review"],
        confidence=0.0, summary="Escalated: no supervisor review")
    return ProcessingResult(
        document_id=state["document_id"],
        decision=review.decision,
        review=review,
        record=state.get("record"),
        extracted_fields=state.get("extracted_fields"),
        validation=state.get("validation"),
        retry_count=state.get("retry_count", 0),
        structuring_attempts=state.get("structuring_attempts", 0),
        events=state.get("events", []),
        errors=state.get("errors", []),
    )


def build_pipeline(settings: Settings | None = None) -> DocumentService:
    """Wire adapters from settings (offline by default)."""
    settings = settings or Settings.from_env()
    return DocumentService(build_ocr(settings), build_llm(settings),
                           build_retriever(settings), settings)
