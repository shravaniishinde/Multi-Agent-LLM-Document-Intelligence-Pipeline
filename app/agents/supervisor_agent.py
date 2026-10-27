"""Agent 4 - Supervisory Review.

Two layers:
1. A deterministic policy produces APPROVE / RETRY / ESCALATE with reasons.
   This is the auditable part and is fully unit-tested.
2. The LLM reviews the full evidence (OCR text, validation, record, RAG context)
   and returns a structured `SupervisorReview`. Its decision is merged with the
   policy decision by taking the *more conservative* of the two, so a model can
   escalate something the rules missed but can never approve something the rules
   escalated. If the LLM output is invalid, the policy review is used as-is.
"""

from __future__ import annotations

from app.adapters.llm import LLMAdapter, StructuredOutputError
from app.config import Settings
from app.graph.state import PipelineState
from app.observability import log_event
from app.schemas.review import DECISION_SEVERITY, ReviewDecision, SupervisorReview
from app.schemas.validation import IssueCode

# Warnings that are acceptable on their own but need a human signature.
ESCALATING_WARNINGS = {IssueCode.APPROVAL_LIMIT_EXCEEDED, IssueCode.UNKNOWN_CUSTOMER}

INSTRUCTIONS = """You are the supervisory reviewer in a document-processing pipeline.
Review the evidence and decide APPROVE, RETRY (re-run OCR) or ESCALATE (human review).
`rule_decision` is the compliance policy result; you may be stricter, never looser.
Give concrete reasons that cite fields, validation issues or reference context."""


def policy_review(state: PipelineState, settings: Settings) -> SupervisorReview:
    validation = state.get("validation")
    record = state.get("record")
    retry_count = state.get("retry_count", 0)
    retries_left = retry_count < settings.max_retries

    if validation is None or not validation.valid:
        errors = [str(e) for e in validation.errors] if validation else ["no validation result"]
        reasons = [f"validation failed after {retry_count} retr{'y' if retry_count == 1 else 'ies'}"]
        return SupervisorReview(decision=ReviewDecision.ESCALATE, reasons=reasons + errors,
                                confidence=validation.confidence if validation else 0.0,
                                summary="Escalated: document did not pass validation")

    if record is None:
        attempts = state.get("structuring_attempts", 0)
        return SupervisorReview(
            decision=ReviewDecision.ESCALATE,
            reasons=[f"structured output failed after {attempts} attempt(s)",
                     state.get("structuring_error") or "unknown structuring error"],
            confidence=validation.confidence,
            summary="Escalated: could not produce a schema-valid record")

    flagged = [w for w in validation.warnings if w.code in ESCALATING_WARNINGS]
    if flagged:
        return SupervisorReview(decision=ReviewDecision.ESCALATE,
                                reasons=[str(w) for w in flagged],
                                confidence=record.confidence,
                                summary="Escalated: reference policy requires human sign-off")

    threshold = settings.approval_confidence_threshold
    if record.confidence < threshold:
        reason = f"record confidence {record.confidence:.2f} below {threshold:.2f}"
        if retries_left:
            return SupervisorReview(decision=ReviewDecision.RETRY, reasons=[reason],
                                    confidence=record.confidence,
                                    summary="Retry: re-scan to raise OCR confidence")
        return SupervisorReview(decision=ReviewDecision.ESCALATE,
                                reasons=[reason, f"retry budget ({settings.max_retries}) used"],
                                confidence=record.confidence,
                                summary="Escalated: confidence still low after retries")

    context_ids = [c.id for c in state.get("retrieved_context", [])]
    reasons = ["all required fields present and valid",
               f"record confidence {record.confidence:.2f} >= {threshold:.2f}"]
    if context_ids:
        reasons.append(f"consistent with reference context: {', '.join(context_ids)}")
    return SupervisorReview(decision=ReviewDecision.APPROVE, reasons=reasons,
                            confidence=record.confidence, summary="Approved")


class SupervisorAgent:
    def __init__(self, llm: LLMAdapter, settings: Settings):
        self.llm = llm
        self.settings = settings

    def __call__(self, state: PipelineState) -> dict:
        rule = policy_review(state, self.settings)
        validation, record = state.get("validation"), state.get("record")
        payload = {
            "document_id": state["document_id"],
            "rule_decision": rule.model_dump(mode="json"),
            "ocr_text": state.get("raw_text", "")[:2000],
            "validation": validation.model_dump(mode="json") if validation else None,
            "record": record.model_dump(mode="json") if record else None,
            "retrieved_context": [c.text for c in state.get("retrieved_context", [])],
            "retry_count": state.get("retry_count", 0),
            "max_retries": self.settings.max_retries,
        }
        events: list[str] = []
        try:
            review = self.llm.invoke_structured(SupervisorReview, INSTRUCTIONS, payload)
        except StructuredOutputError as exc:
            log_event("supervisor_llm_invalid", state, error=str(exc).splitlines()[0][:200])
            review = rule
            events.append(f"Supervisor LLM output invalid, using policy review ({exc})")

        if DECISION_SEVERITY[review.decision] < DECISION_SEVERITY[rule.decision]:
            review = rule  # the LLM may not be more lenient than policy
        if (review.decision == ReviewDecision.RETRY
                and state.get("retry_count", 0) >= self.settings.max_retries):
            review = review.model_copy(update={
                "decision": ReviewDecision.ESCALATE,
                "reasons": review.reasons + ["retry budget exhausted"]})

        log_event("supervisor_decision", state, decision=review.decision.value,
                  policy_decision=rule.decision.value, reasons=review.reasons,
                  retry_count=state.get("retry_count", 0))
        events.append(f"Supervisor: {review.decision.value} - {'; '.join(review.reasons)}")
        return {"review": review, "decision": review.decision, "events": events}
