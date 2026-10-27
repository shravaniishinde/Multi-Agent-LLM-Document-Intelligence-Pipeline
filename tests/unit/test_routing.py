from app.graph.routing import (
    route_after_structuring,
    route_after_supervisor,
    route_after_validation,
)
from app.schemas.review import ReviewDecision
from app.schemas.validation import IssueCode, ValidationIssue, ValidationResult


def validation(valid=True, retryable=False):
    errors = [] if valid else [ValidationIssue(field="f", code=IssueCode.INVALID_DATE,
                                               message="m", retryable=retryable)]
    return ValidationResult(valid=valid, errors=errors, confidence=0.5)


def test_valid_goes_to_structuring(settings):
    assert route_after_validation({"validation": validation()}, settings) == "structure"


def test_retryable_failure_retries_while_budget_left(settings):
    state = {"validation": validation(False, True), "retry_count": settings.max_retries - 1}
    assert route_after_validation(state, settings) == "retry"


def test_retryable_failure_goes_to_supervisor_when_budget_used(settings):
    state = {"validation": validation(False, True), "retry_count": settings.max_retries}
    assert route_after_validation(state, settings) == "supervisor"


def test_non_retryable_failure_goes_straight_to_supervisor(settings):
    state = {"validation": validation(False, False), "retry_count": 0}
    assert route_after_validation(state, settings) == "supervisor"


def test_structuring_retries_then_gives_up(settings):
    assert route_after_structuring({"record": None, "structuring_attempts": 1}, settings) == "structure"
    state = {"record": None, "structuring_attempts": settings.max_structuring_attempts}
    assert route_after_structuring(state, settings) == "supervisor"
    assert route_after_structuring({"record": object(), "structuring_attempts": 1},
                                   settings) == "supervisor"


def test_supervisor_routing(settings):
    assert route_after_supervisor({"decision": ReviewDecision.RETRY, "retry_count": 0},
                                  settings) == "retry"
    assert route_after_supervisor({"decision": ReviewDecision.RETRY,
                                   "retry_count": settings.max_retries}, settings) == "end"
    assert route_after_supervisor({"decision": ReviewDecision.APPROVE}, settings) == "end"
    assert route_after_supervisor({"decision": ReviewDecision.ESCALATE}, settings) == "end"
