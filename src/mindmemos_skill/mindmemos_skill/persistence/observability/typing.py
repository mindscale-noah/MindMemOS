"""Validated run, span, and log records for one observability backend."""

from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, JsonValue


class AuditRun(BaseModel):
    """Latest durable state of an algorithm run."""

    model_config = ConfigDict(extra="forbid")
    """Reject unknown database fields."""

    project_id: str = Field(min_length=1)
    """Repository scope owning this run."""

    run_id: str = Field(min_length=1)
    """Stable run identity shared by recovery attempts."""

    algorithm: str = Field(min_length=1)
    """Configured algorithm or training pipeline name."""

    base_version_id: str = Field(min_length=1)
    """Input Skill version captured before execution."""

    published_version_id: str | None = None
    """Committed output version, absent before publication."""

    status: Literal["running", "completed", "failed"]
    """Most recent execution outcome."""

    audit_status: Literal["complete", "incomplete"]
    """Whether every observed export in this process succeeded."""

    capture_level: Literal["info", "debug"]
    """Agent payload level selected when this execution began."""

    trace_id: str = Field(min_length=32, max_length=32)
    """Trace identity of the latest physical execution attempt."""

    started_at: AwareDatetime
    """Start time of the latest execution attempt."""

    finished_at: AwareDatetime | None = None
    """End time of the latest execution attempt, if terminal."""

    error_type: str | None = None
    """Exception class for a failed run without potentially sensitive detail."""


class AuditSpan(BaseModel):
    """One completed OpenTelemetry span and its tree edges."""

    model_config = ConfigDict(extra="forbid")
    """Reject unknown database fields."""

    trace_id: str = Field(min_length=32, max_length=32)
    """OpenTelemetry trace identity."""

    span_id: str = Field(min_length=16, max_length=16)
    """OpenTelemetry span identity within the trace."""

    parent_span_id: str | None = None
    """Parent span identity, absent at the trace root."""

    project_id: str = Field(min_length=1)
    """Repository scope copied from the active run."""

    run_id: str = Field(min_length=1)
    """Stable algorithm run identity."""

    name: str = Field(min_length=1)
    """Operation name displayed in a trace tree."""

    kind: str
    """OpenTelemetry span kind."""

    status: str
    """OpenTelemetry span status."""

    started_at: AwareDatetime
    """Operation start timestamp."""

    finished_at: AwareDatetime
    """Operation end timestamp."""

    attributes: dict[str, JsonValue] = Field(default_factory=dict)
    """Bounded operation attributes without debug input or output bodies."""

    events: list[dict[str, JsonValue]] = Field(default_factory=list)
    """Timed span events such as candidate decisions."""

    links: list[dict[str, JsonValue]] = Field(default_factory=list)
    """Causal links to other traces, including recovered attempts."""

    resource: dict[str, JsonValue] = Field(default_factory=dict)
    """OpenTelemetry resource attributes such as service.name."""


class AuditLog(BaseModel):
    """One trace-correlated structured audit event."""

    model_config = ConfigDict(extra="forbid")
    """Reject unknown database fields."""

    event_id: str = Field(min_length=1)
    """Idempotent event identity."""

    project_id: str = Field(min_length=1)
    """Repository scope owning the event."""

    run_id: str = Field(min_length=1)
    """Stable algorithm run identity."""

    trace_id: str = Field(min_length=32, max_length=32)
    """Trace containing the event."""

    span_id: str = Field(min_length=16, max_length=16)
    """Span containing the event."""

    occurred_at: AwareDatetime
    """Time the event was emitted."""

    level: str = Field(min_length=1)
    """OpenTelemetry severity text for this event."""

    name: str = Field(min_length=1)
    """Stable event type."""

    body: dict[str, JsonValue] = Field(default_factory=dict)
    """Structured and size-limited event content."""
