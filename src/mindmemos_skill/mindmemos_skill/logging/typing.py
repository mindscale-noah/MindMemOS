"""Task log records shared by capture, persistence and query APIs."""

from typing import Any

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field


class TaskLogRecord(BaseModel):
    """Immutable Python log emitted during one task attempt."""

    model_config = ConfigDict(extra="forbid")
    """Reject fields outside the durable log contract."""

    sequence: int | None = Field(default=None, ge=1)
    """Database-assigned pagination cursor; None before the record is committed."""

    log_id: str
    """Globally unique log identity."""

    project_id: str
    """Authorized project owning the task."""

    task_id: str
    """Durable task emitting the log."""

    attempt: int = Field(ge=1)
    """Physical task attempt that emitted the log."""

    occurred_at: AwareDatetime
    """Original Python log emission time."""

    payload: dict[str, Any]
    """Sanitized name, level, message, body, traceback and correlation IDs."""


class TaskLog(TaskLogRecord):
    """Stored task log with a stable pagination sequence."""

    sequence: int = Field(ge=1)
    """Increasing committed log cursor; gaps are allowed in durable storage."""
