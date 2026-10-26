"""LangGraph wiring for the four-agent pipeline.

    START -> ocr -> validate
    validate   --valid-->                         structure
    validate   --invalid, retryable, budget left-> retry -> ocr
    validate   --invalid otherwise-->             supervisor
    structure  --record ok-->                     supervisor
    structure  --schema error, attempts left-->   structure
    structure  --schema error, attempts used-->   supervisor
    supervisor --RETRY, budget left-->            retry -> ocr
    supervisor --APPROVE / ESCALATE-->            END
"""

from __future__ import annotations

from functools import partial

from langgraph.graph import END, START, StateGraph

from app.adapters.llm import LLMAdapter
from app.adapters.ocr import OCRAdapter
from app.agents import OCRAgent, StructuringAgent, SupervisorAgent, ValidationAgent
from app.config import Settings
from app.graph.routing import (
    route_after_structuring,
    route_after_supervisor,
    route_after_validation,
)
from app.graph.state import PipelineState
from app.rag.retriever import Retriever


def make_retry_node(settings: Settings):
    def prepare_retry(state: PipelineState) -> dict:
        retry = state.get("retry_count", 0) + 1
        return {
            "retry_count": retry,
            "validation": None,
            "record": None,
            "structuring_attempts": 0,
            "structuring_error": None,
            "review": None,
            "decision": None,
            "events": [f"Retry {retry}/{settings.max_retries}: re-running OCR with "
                       f"enhanced preprocessing"],
        }
    return prepare_retry


def build_workflow(ocr: OCRAdapter, llm: LLMAdapter, retriever: Retriever | None,
                   settings: Settings):
    graph = StateGraph(PipelineState)
    graph.add_node("ocr", OCRAgent(ocr))
    graph.add_node("validate", ValidationAgent(retriever, settings))
    graph.add_node("structure", StructuringAgent(llm))
    graph.add_node("supervisor", SupervisorAgent(llm, settings))
    graph.add_node("retry", make_retry_node(settings))

    graph.add_edge(START, "ocr")
    graph.add_edge("ocr", "validate")
    graph.add_conditional_edges(
        "validate", partial(route_after_validation, settings=settings),
        {"structure": "structure", "retry": "retry", "supervisor": "supervisor"})
    graph.add_conditional_edges(
        "structure", partial(route_after_structuring, settings=settings),
        {"structure": "structure", "supervisor": "supervisor"})
    graph.add_conditional_edges(
        "supervisor", partial(route_after_supervisor, settings=settings),
        {"retry": "retry", "end": END})
    graph.add_edge("retry", "ocr")
    return graph.compile()


def recursion_limit(settings: Settings) -> int:
    """Upper bound on graph steps; a hard backstop on top of the counters."""
    per_pass = 4 + settings.max_structuring_attempts  # ocr, validate, structure*, supervisor, retry
    return per_pass * (settings.max_retries + 1) + 5
