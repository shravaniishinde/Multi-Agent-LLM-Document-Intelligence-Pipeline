"""Conditional-edge functions. Pure, so every branch is unit-testable.

Every loop in the graph is bounded by a counter in state:
- validation/supervisor -> retry -> OCR is bounded by `retry_count < max_retries`
- structure -> structure is bounded by `structuring_attempts < max_structuring_attempts`
"""

from __future__ import annotations

from typing import Literal

from app.config import Settings
from app.graph.state import PipelineState
from app.schemas.review import ReviewDecision


def route_after_validation(state: PipelineState, settings: Settings
                           ) -> Literal["structure", "retry", "supervisor"]:
    validation = state.get("validation")
    if validation is not None and validation.valid:
        return "structure"
    if (validation is not None and validation.retryable
            and state.get("retry_count", 0) < settings.max_retries):
        return "retry"
    return "supervisor"


def route_after_structuring(state: PipelineState, settings: Settings
                            ) -> Literal["structure", "supervisor"]:
    if state.get("record") is not None:
        return "supervisor"
    if state.get("structuring_attempts", 0) < settings.max_structuring_attempts:
        return "structure"
    return "supervisor"


def route_after_supervisor(state: PipelineState, settings: Settings
                           ) -> Literal["retry", "end"]:
    if (state.get("decision") == ReviewDecision.RETRY
            and state.get("retry_count", 0) < settings.max_retries):
        return "retry"
    return "end"
