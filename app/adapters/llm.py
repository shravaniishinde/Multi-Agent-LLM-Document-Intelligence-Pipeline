"""LLM adapters with Pydantic-enforced structured output.

Every adapter goes through `invoke_structured`, which validates whatever the
model produced against the requested Pydantic schema. Malformed JSON or values
that break the schema raise `StructuredOutputError`; nothing unvalidated leaves
this module.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Callable, TypeVar

from pydantic import BaseModel, ValidationError

from app.config import Settings
from app.schemas.document import DocumentRecord
from app.schemas.review import SupervisorReview

T = TypeVar("T", bound=BaseModel)


class StructuredOutputError(RuntimeError):
    pass


class LLMAdapter(ABC):
    name: str

    @abstractmethod
    def _invoke(self, schema: type[BaseModel], instructions: str, payload: dict) -> Any:
        """Return the model's raw answer: a schema instance, dict or JSON string."""

    def invoke_structured(self, schema: type[T], instructions: str, payload: dict) -> T:
        raw = self._invoke(schema, instructions, payload)
        try:
            if isinstance(raw, BaseModel):
                raw = raw.model_dump()
            if isinstance(raw, str):
                raw = json.loads(raw)
            return schema.model_validate(raw)
        except (json.JSONDecodeError, ValidationError, TypeError) as exc:
            raise StructuredOutputError(
                f"{self.name} output did not match {schema.__name__}: {exc}"
            ) from exc


class LangChainLLMAdapter(LLMAdapter):
    """Wraps any LangChain chat model that supports `with_structured_output`."""

    def __init__(self, chat_model: Any, name: str = "langchain"):
        self.chat_model = chat_model
        self.name = name

    def _invoke(self, schema: type[BaseModel], instructions: str, payload: dict) -> Any:
        from langchain_core.messages import HumanMessage, SystemMessage

        runnable = self.chat_model.with_structured_output(schema, include_raw=True)
        result = runnable.invoke([
            SystemMessage(content=instructions),
            HumanMessage(content=json.dumps(payload, indent=2, default=str)),
        ])
        if result.get("parsing_error") or result.get("parsed") is None:
            raise StructuredOutputError(f"{self.name} parsing error: {result.get('parsing_error')}")
        return result["parsed"]


# --- Offline fake -------------------------------------------------------------

def _fake_document_record(payload: dict) -> dict:
    """What a well-behaved LLM would return: normalized fields mapped to the schema."""
    fields = payload["normalized_fields"]
    entities = [
        {"label": label, "value": fields[key]}
        for label, key in [("subtotal", "subtotal"), ("tax", "tax")]
        if fields.get(key)
    ]
    return {
        "document_id": payload["document_id"],
        "document_type": fields.get("document_type", "unknown"),
        "customer_name": fields.get("customer_name"),
        "customer_id": fields.get("customer_id"),
        "reference_number": fields.get("reference_number"),
        "document_date": fields.get("document_date"),
        "total_amount": fields.get("total_amount"),
        "currency": fields.get("currency"),
        "address": fields.get("address"),
        "extracted_entities": entities,
        "confidence": payload["confidence"],
        "validation_errors": payload.get("validation_warnings", []),
        "review_required": bool(payload.get("validation_warnings")),
    }


def _fake_supervisor_review(payload: dict) -> dict:
    """Echo the deterministic policy decision, phrased as a summary."""
    rule = payload["rule_decision"]
    return {
        "decision": rule["decision"],
        "reasons": rule["reasons"],
        "confidence": rule["confidence"],
        "summary": f"{rule['decision']}: " + "; ".join(rule["reasons"]),
    }


DEFAULT_FAKE_HANDLERS: dict[type[BaseModel], Callable[[dict], Any]] = {
    DocumentRecord: _fake_document_record,
    SupervisorReview: _fake_supervisor_review,
}


class FakeLLMAdapter(LLMAdapter):
    """Deterministic stand-in for a chat model.

    `scripted` responses are returned first (in order) before falling back to the
    schema handlers, which lets tests inject malformed output on purpose.
    """

    name = "fake"

    def __init__(self, scripted: list[Any] | None = None,
                 handlers: dict[type[BaseModel], Callable[[dict], Any]] | None = None):
        self.scripted = deque(scripted or [])
        self.handlers = {**DEFAULT_FAKE_HANDLERS, **(handlers or {})}
        self.calls: list[tuple[str, dict]] = []

    def _invoke(self, schema: type[BaseModel], instructions: str, payload: dict) -> Any:
        self.calls.append((schema.__name__, payload))
        if self.scripted:
            return self.scripted.popleft()
        handler = self.handlers.get(schema)
        if handler is None:
            raise StructuredOutputError(f"fake LLM has no handler for {schema.__name__}")
        return handler(payload)


def build_llm(settings: Settings) -> LLMAdapter:
    if settings.llm_provider == "fake":
        return FakeLLMAdapter()
    if not settings.llm_model:
        raise ValueError(f"LLM_MODEL must be set when LLM_PROVIDER={settings.llm_provider}")
    if settings.llm_provider == "openai":
        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(model=settings.llm_model, temperature=0)
    elif settings.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        model = ChatAnthropic(model=settings.llm_model, temperature=0)
    else:
        from langchain_groq import ChatGroq

        model = ChatGroq(model=settings.llm_model, temperature=0)
    return LangChainLLMAdapter(model, name=f"{settings.llm_provider}:{settings.llm_model}")
