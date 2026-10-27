"""Structured JSON logging and in-memory metrics.

Every pipeline log line is one JSON object with an `event` name plus the
document's `document_id` and per-run `correlation_id`, so one document's journey
can be grepped out of interleaved batch logs.

The metrics store is in-process and resets on restart. That is deliberate for a
single-instance portfolio service; a production deployment would export the
same counters to Prometheus/OpenTelemetry or another centralized backend.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

if TYPE_CHECKING:
    from app.config import Settings
    from app.schemas.review import ProcessingResult

LOGGER_NAME = "docintel"
logger = logging.getLogger(LOGGER_NAME)

_SENSITIVE = ("key", "secret", "token", "password", "authorization")


def _redact(fields: dict[str, Any]) -> dict[str, Any]:
    return {k: ("***" if any(s in k.lower() for s in _SENSITIVE) else v)
            for k, v in fields.items()}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
            **_redact(getattr(record, "fields", {})),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        fields = " ".join(f"{k}={v}" for k, v in _redact(getattr(record, "fields", {})).items())
        return f"{record.levelname:<7} {record.getMessage():<24} {fields}"


def configure_logging(settings: "Settings") -> None:
    """Attach one stderr handler to the `docintel` logger (idempotent)."""
    for handler in list(logger.handlers):
        if getattr(handler, "_docintel", False):
            logger.removeHandler(handler)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter() if settings.log_format == "json" else TextFormatter())
    handler._docintel = True  # type: ignore[attr-defined]
    logger.addHandler(handler)
    logger.setLevel(settings.log_level)
    logger.propagate = False


def log_event(event: str, state: dict | None = None, level: int = logging.INFO,
              **fields: Any) -> None:
    if state is not None:
        fields = {"document_id": state.get("document_id"),
                  "correlation_id": state.get("correlation_id"), **fields}
    logger.log(level, event, extra={"fields": fields})


class MetricsSnapshot(BaseModel):
    documents_processed: int
    successful: int
    escalated: int
    failed: int
    retries: int
    validation_failures: int
    structuring_failures: int
    average_latency_ms: float
    max_latency_ms: float


class MetricsStore:
    """Thread-safe counters. `successful` = auto-approved, `escalated` = sent to a
    human, `failed` = the pipeline itself raised for that document."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with getattr(self, "_lock", threading.Lock()):
            self._counts = dict(documents_processed=0, successful=0, escalated=0, failed=0,
                                retries=0, validation_failures=0, structuring_failures=0)
            self._latency_total = 0.0
            self._latency_max = 0.0

    def record(self, result: "ProcessingResult") -> None:
        from app.schemas.review import ReviewDecision

        with self._lock:
            c = self._counts
            c["documents_processed"] += 1
            if result.status == "failed":
                c["failed"] += 1
            elif result.decision == ReviewDecision.APPROVE:
                c["successful"] += 1
            else:
                c["escalated"] += 1
            c["retries"] += result.retry_count
            c["validation_failures"] += result.validation_failures
            c["structuring_failures"] += result.structuring_failures
            self._latency_total += result.processing_ms
            self._latency_max = max(self._latency_max, result.processing_ms)

    def snapshot(self) -> MetricsSnapshot:
        with self._lock:
            n = self._counts["documents_processed"]
            return MetricsSnapshot(
                **self._counts,
                average_latency_ms=round(self._latency_total / n, 2) if n else 0.0,
                max_latency_ms=round(self._latency_max, 2),
            )
