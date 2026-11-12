"""Transport-independent task execution records."""

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, computed_field

from ...logging.context import capture_trace_context, current_request_id

GRAPH_BUILD_TASK = "skill.graph_build"
JUDGE_TASK = "skill.judge"
EVOLVE_TASK = "skill.evolve"


class TaskStatus(StrEnum):
    """Lifecycle state of one submitted task."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskSpec(BaseModel):
    """Immutable work requested by an authorized caller."""

    model_config = ConfigDict(extra="forbid")
    """Reject unknown task inputs."""

    task_id: str = Field(min_length=1)
    """Stable identity used for delivery and status lookup."""

    project_id: str = Field(min_length=1)
    """Project that owns the task."""

    kind: str = Field(min_length=1)
    """Explicit handler name from a trusted application catalog."""

    payload: dict[str, Any]
    """Validated JSON-compatible business request, including explicit configuration."""

    trace_context: dict[str, str] = Field(default_factory=capture_trace_context)
    """Saved W3C traceparent and optional tracestate from the submission span."""

    request_id: str | None = Field(default_factory=current_request_id)
    """Original submission request identity, retained across worker recovery."""


class TaskSnapshot(BaseModel):
    """Current task state returned through either execution backend."""

    model_config = ConfigDict(extra="forbid")
    """Reject undeclared task state."""

    task_id: str
    """Stable task identity."""

    project_id: str
    """Owning project identity."""

    kind: str
    """Trusted handler name."""

    status: TaskStatus
    """Current lifecycle state."""

    result: dict[str, Any] | None = None
    """Successful JSON-compatible result."""

    error_type: str | None = None
    """Failed exception type without sensitive exception text."""

    attempt: int = 0
    """Number of physical execution attempts begun."""

    accepted_at: AwareDatetime | None = None
    """First task acceptance time; never reset when reclaiming an attempt."""

    started_at: AwareDatetime | None = None
    """Start of the current physical attempt, or None while queued."""

    finished_at: AwareDatetime | None = None
    """Terminal result commit time, or None before successful persistence."""

    @computed_field
    @property
    def waiting_ms(self) -> float | None:
        """Return time before this attempt, including prior interrupted attempts."""
        if self.accepted_at is None:
            return None
        return max(
            0, ((self.started_at or self.finished_at or datetime.now(UTC)) - self.accepted_at).total_seconds() * 1000
        )

    @computed_field
    @property
    def execution_ms(self) -> float | None:
        """Return elapsed execution time for this attempt, including result commit."""
        if self.started_at is None:
            return None
        return max(0, ((self.finished_at or datetime.now(UTC)) - self.started_at).total_seconds() * 1000)

    @computed_field
    @property
    def end_to_end_ms(self) -> float | None:
        """Return elapsed time since first acceptance, including recovery waits."""
        if self.accepted_at is None:
            return None
        return max(0, ((self.finished_at or datetime.now(UTC)) - self.accepted_at).total_seconds() * 1000)


TaskHandler = Callable[[TaskSpec], Awaitable[dict[str, Any]]]
"""Request-only task handler returning JSON-compatible fields defined by its pipeline."""
