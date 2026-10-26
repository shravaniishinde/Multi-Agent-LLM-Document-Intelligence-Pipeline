import pytest

from app.adapters.llm import FakeLLMAdapter, LangChainLLMAdapter, StructuredOutputError, build_llm
from app.config import Settings
from app.schemas.review import ReviewDecision, SupervisorReview

GOOD_REVIEW = {"decision": "APPROVE", "reasons": ["ok"], "confidence": 0.9, "summary": "fine"}


def test_accepts_dict_json_string_and_model():
    for raw in [GOOD_REVIEW, '{"decision": "APPROVE", "reasons": ["ok"], "confidence": 0.9, '
                             '"summary": "fine"}', SupervisorReview(**GOOD_REVIEW)]:
        review = FakeLLMAdapter(scripted=[raw]).invoke_structured(SupervisorReview, "", {})
        assert review.decision == ReviewDecision.APPROVE


@pytest.mark.parametrize("raw", [
    "Sure! Here is the JSON you asked for: {decision: APPROVE}",  # not JSON
    {"decision": "MAYBE", "reasons": ["x"], "confidence": 0.5, "summary": "s"},  # bad enum
    {"decision": "APPROVE", "reasons": [], "confidence": 0.5, "summary": "s"},  # empty reasons
    {"decision": "APPROVE", "reasons": ["x"], "confidence": 7, "summary": "s"},  # out of range
    ["not", "an", "object"],
])
def test_malformed_output_raises(raw):
    with pytest.raises(StructuredOutputError):
        FakeLLMAdapter(scripted=[raw]).invoke_structured(SupervisorReview, "", {})


def test_scripted_responses_then_handlers():
    llm = FakeLLMAdapter(scripted=["garbage"])
    with pytest.raises(StructuredOutputError):
        llm.invoke_structured(SupervisorReview, "", {})
    payload = {"rule_decision": {"decision": "ESCALATE", "reasons": ["r"], "confidence": 0.1}}
    assert llm.invoke_structured(SupervisorReview, "", payload).decision == ReviewDecision.ESCALATE
    assert [name for name, _ in llm.calls] == ["SupervisorReview", "SupervisorReview"]


class _StubStructured:
    def __init__(self, result):
        self.result = result

    def invoke(self, messages):
        self.messages = messages
        return self.result


class _StubChatModel:
    """Mimics `BaseChatModel.with_structured_output(..., include_raw=True)`."""

    def __init__(self, result):
        self.runnable = _StubStructured(result)

    def with_structured_output(self, schema, include_raw=False):
        assert include_raw
        return self.runnable


def test_langchain_adapter_validates_parsed_output():
    model = _StubChatModel({"raw": None, "parsed": SupervisorReview(**GOOD_REVIEW),
                            "parsing_error": None})
    review = LangChainLLMAdapter(model).invoke_structured(SupervisorReview, "sys", {"a": 1})
    assert review.summary == "fine"
    assert model.runnable.messages[0].content == "sys"


def test_langchain_adapter_surfaces_parsing_error():
    model = _StubChatModel({"raw": None, "parsed": None, "parsing_error": ValueError("bad")})
    with pytest.raises(StructuredOutputError):
        LangChainLLMAdapter(model).invoke_structured(SupervisorReview, "sys", {})


def test_real_provider_requires_model_name():
    with pytest.raises(ValueError, match="LLM_MODEL"):
        build_llm(Settings(llm_provider="openai"))
