"""Transport-independent task and progress records."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


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


class ProgressEvent(BaseModel):
    """One algorithm-selected progress boundary."""

    model_config = ConfigDict(extra="forbid")
    """Reject undeclared progress fields."""

    stage: str = Field(min_length=1, max_length=100)
    """Stable stage identifier selected by the algorithm."""

    message: str = Field(min_length=1, max_length=500)
    """Short, non-sensitive user-facing progress summary."""

    sequence: int | None = Field(default=None, ge=1)
    """Task-scoped event order assigned after the report is stored."""

    completed: int | None = Field(default=None, ge=0)
    """Completed items in this stage, if a denominator exists."""

    total: int | None = Field(default=None, ge=1)
    """Total items in this stage, if meaningful."""

    occurred_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    """UTC time when the algorithm reported progress."""

    @model_validator(mode="after")
    def check_count(self) -> "ProgressEvent":
        """Require both counts together and keep the fraction valid."""
        if (self.completed is None) != (self.total is None):
            raise ValueError("completed and total must be supplied together")
        if self.total is not None and self.completed > self.total:
            raise ValueError("completed cannot exceed total")
        return self


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

    progress: ProgressEvent | None = None
    """Most recently committed algorithm progress."""

    result: dict[str, Any] | None = None
    """Successful JSON-compatible result."""

    error_type: str | None = None
    """Failed exception type without sensitive exception text."""

    attempt: int = 0
    """Number of physical execution attempts begun."""
