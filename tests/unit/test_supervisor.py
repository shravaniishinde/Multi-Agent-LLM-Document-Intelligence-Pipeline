import pytest

from app.adapters.llm import FakeLLMAdapter
from app.agents.supervisor_agent import SupervisorAgent, policy_review
from app.agents.validation_agent import ValidationAgent
from app.schemas.document import DocumentRecord
from app.schemas.review import ReviewDecision


def build_state(settings, fields, confidence=0.96, retriever=None, retry_count=0,
                with_record=True):
    validation, context = ValidationAgent(retriever, settings).validate(fields, confidence)
    record = None
    if with_record and validation.valid:
        record = DocumentRecord(
            document_id="d", document_type="invoice", customer_name=fields.customer_name,
            customer_id=fields.customer_id, reference_number=fields.reference_number,
            document_date="2026-03-14", total_amount="1242.00", currency="USD",
            confidence=validation.confidence,
            validation_errors=[str(w) for w in validation.warnings],
            review_required=bool(validation.warnings))
    return {"document_id": "d", "validation": validation, "record": record,
            "retrieved_context": context, "retry_count": retry_count,
            "structuring_attempts": 1 if record else 0}


def test_approves_clean_document(settings, good_fields, retriever):
    review = policy_review(build_state(settings, good_fields, retriever=retriever), settings)
    assert review.decision == ReviewDecision.APPROVE
    assert any("reference context" in r for r in review.reasons)


def test_escalates_failed_validation(settings, good_fields):
    fields = good_fields.model_copy(update={"customer_id": None})
    review = policy_review(build_state(settings, fields), settings)
    assert review.decision == ReviewDecision.ESCALATE
    assert any("MISSING_FIELD" in r for r in review.reasons)


def test_escalates_failed_structuring(settings, good_fields):
    state = build_state(settings, good_fields, with_record=False)
    state.update(structuring_attempts=2, structuring_error="bad json")
    review = policy_review(state, settings)
    assert review.decision == ReviewDecision.ESCALATE
    assert review.reasons == ["structured output failed after 2 attempt(s)", "bad json"]


def test_escalates_policy_warning(settings, good_fields, retriever):
    fields = good_fields.model_copy(update={"subtotal": None, "tax": None,
                                            "total_amount": "30,000.00"})
    review = policy_review(build_state(settings, fields, retriever=retriever), settings)
    assert review.decision == ReviewDecision.ESCALATE
    assert "APPROVAL_LIMIT_EXCEEDED" in review.reasons[0]


@pytest.mark.parametrize("retry_count,expected", [(0, ReviewDecision.RETRY),
                                                  (2, ReviewDecision.ESCALATE)])
def test_low_confidence_retries_until_budget_used(settings, good_fields, retry_count, expected):
    state = build_state(settings, good_fields, confidence=0.6, retry_count=retry_count)
    assert policy_review(state, settings).decision == expected


def test_llm_cannot_be_more_lenient_than_policy(settings, good_fields):
    fields = good_fields.model_copy(update={"customer_id": None})
    lenient = {"decision": "APPROVE", "reasons": ["looks fine to me"], "confidence": 0.99,
               "summary": "approve"}
    update = SupervisorAgent(FakeLLMAdapter(scripted=[lenient]), settings)(
        build_state(settings, fields))
    assert update["decision"] == ReviewDecision.ESCALATE


def test_llm_can_be_stricter_than_policy(settings, good_fields):
    strict = {"decision": "ESCALATE", "reasons": ["vendor bank details changed"],
              "confidence": 0.8, "summary": "suspicious"}
    update = SupervisorAgent(FakeLLMAdapter(scripted=[strict]), settings)(
        build_state(settings, good_fields))
    assert update["decision"] == ReviewDecision.ESCALATE
    assert update["review"].reasons == ["vendor bank details changed"]


def test_invalid_llm_review_falls_back_to_policy(settings, good_fields):
    update = SupervisorAgent(FakeLLMAdapter(scripted=["not json"]), settings)(
        build_state(settings, good_fields))
    assert update["decision"] == ReviewDecision.APPROVE
    assert any("using policy review" in e for e in update["events"])


def test_llm_retry_with_no_budget_becomes_escalate(settings, good_fields):
    retry = {"decision": "RETRY", "reasons": ["blurry"], "confidence": 0.5, "summary": "rescan"}
    update = SupervisorAgent(FakeLLMAdapter(scripted=[retry]), settings)(
        build_state(settings, good_fields, retry_count=settings.max_retries))
    assert update["decision"] == ReviewDecision.ESCALATE
    assert "retry budget exhausted" in update["review"].reasons
