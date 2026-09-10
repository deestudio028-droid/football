"""Monitoring event schema (Gate 10 emission contract).

One immutable record type, one validator. Every field required by the
authorized emission contract is present and checked; the validator is
strict so a malformed event fails at construction rather than silently
entering the store.

`model_version` is READ from `models.config.MODEL_VERSION` and never
written back. It is therefore "v1.0" for every event this package emits.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping

from models.config import MODEL_VERSION

from .config import SCHEMA_VERSION, Severity, Status, severity_for

#: Fields every persisted event must carry.
REQUIRED_EVENT_FIELDS: tuple[str, ...] = (
    "schema_version",
    "event_id",
    "timestamp",
    "model_version",
    "signal_id",
    "signal_name",
    "status",
    "severity",
    "metrics",
    "baseline",
    "threshold",
    "sample_count",
    "context",
    "message",
)


class SchemaViolation(ValueError):
    """Raised when an event does not satisfy the emission contract."""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class MonitoringEvent:
    """A single monitoring observation.

    Immutable by design: an emitted observation is a record of what was
    measured and must not be edited after the fact.
    """

    signal_id: str
    signal_name: str
    status: Status
    metrics: Mapping[str, Any]
    baseline: Any
    threshold: Any
    sample_count: int
    message: str
    context: Mapping[str, Any] = field(default_factory=dict)
    severity: Severity | None = None
    timestamp: str = field(default_factory=_utc_now_iso)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    schema_version: str = SCHEMA_VERSION
    model_version: str = MODEL_VERSION

    def __post_init__(self) -> None:
        # Severity is derived from status unless explicitly supplied, and
        # is never an independent policy decision.
        if self.severity is None:
            object.__setattr__(self, "severity", severity_for(self.status))
        object.__setattr__(self, "status", Status(self.status))
        object.__setattr__(self, "severity", Severity(self.severity))
        validate_event(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        raw = asdict(self)
        raw["status"] = Status(self.status).value
        raw["severity"] = Severity(self.severity).value
        raw["metrics"] = dict(self.metrics)
        raw["context"] = dict(self.context)
        return raw

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, default=str)


def validate_event(payload: Mapping[str, Any]) -> None:
    """Strict schema check. Raises `SchemaViolation` on any problem."""
    missing = [f for f in REQUIRED_EVENT_FIELDS if f not in payload]
    if missing:
        raise SchemaViolation(f"event is missing required field(s): {missing}")

    if payload["schema_version"] != SCHEMA_VERSION:
        raise SchemaViolation(
            f"unexpected schema_version {payload['schema_version']!r}; expected {SCHEMA_VERSION!r}"
        )

    try:
        Status(payload["status"])
    except ValueError as exc:
        raise SchemaViolation(f"unknown status {payload['status']!r}") from exc

    try:
        Severity(payload["severity"])
    except ValueError as exc:
        raise SchemaViolation(f"unknown severity {payload['severity']!r}") from exc

    if not isinstance(payload["signal_id"], str) or not payload["signal_id"]:
        raise SchemaViolation("signal_id must be a non-empty string")
    if not isinstance(payload["signal_name"], str) or not payload["signal_name"]:
        raise SchemaViolation("signal_name must be a non-empty string")
    if not isinstance(payload["message"], str) or not payload["message"]:
        raise SchemaViolation("message must be a non-empty string")
    if not isinstance(payload["metrics"], Mapping):
        raise SchemaViolation("metrics must be a mapping")
    if not isinstance(payload["context"], Mapping):
        raise SchemaViolation("context must be a mapping")
    if not isinstance(payload["sample_count"], int) or isinstance(payload["sample_count"], bool):
        raise SchemaViolation("sample_count must be an int")
    if payload["sample_count"] < 0:
        raise SchemaViolation("sample_count must be non-negative")
    if payload["model_version"] != MODEL_VERSION:
        raise SchemaViolation(
            f"model_version {payload['model_version']!r} does not match the project's "
            f"{MODEL_VERSION!r}; monitoring never rewrites the model version"
        )
    # Timestamp must be parseable ISO-8601.
    try:
        datetime.fromisoformat(str(payload["timestamp"]))
    except ValueError as exc:
        raise SchemaViolation(f"timestamp {payload['timestamp']!r} is not ISO-8601") from exc


__all__ = ["MonitoringEvent", "SchemaViolation", "validate_event", "REQUIRED_EVENT_FIELDS"]
