"""Durable task records stored in the existing Skill database."""

from typing import Any

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from ....infra.task_manager.typing import ProgressEvent, TaskSnapshot, TaskSpec, TaskStatus


class TaskRecord(BaseModel):
    """One durable request and the current execution attempt."""

    model_config = ConfigDict(extra="forbid")
    """Reject unknown stored fields."""

    task_id: str = Field(min_length=1)
    """Globally unique task identity."""

    project_id: str = Field(min_length=1)
    """Project owning the task."""

    kind: str = Field(min_length=1)
    """Trusted application handler name."""

    payload: dict[str, Any]
    """Immutable request with explicit configuration needed after process restart."""

    status: TaskStatus
    """Current task lifecycle state."""

    progress: dict[str, Any] | None = None
    """Latest validated progress event for fast reads."""

    progress_seq: int = Field(default=0, ge=0)
    """Last allocated progress sequence."""

    result: dict[str, Any] | None = None
    """Final successful result when available."""

    error_type: str | None = None
    """Failed exception type without sensitive exception text."""

    attempt: int = Field(default=0, ge=0)
    """Number of physical attempts started."""

    lease_token: str | None = None
    """Random fencing token of the current worker."""

    lease_expires_at: AwareDatetime | None = None
    """Time after which another worker may reclaim the task."""

    next_notify_at: AwareDatetime
    """Earliest time for another fire-and-forget Kafka notification."""

    created_at: AwareDatetime
    """Initial database commit time."""

    updated_at: AwareDatetime
    """Time of the latest state change."""

    def spec(self) -> TaskSpec:
        """Return the immutable work sent to the handler."""
        return TaskSpec(task_id=self.task_id, project_id=self.project_id, kind=self.kind, payload=self.payload)

    def snapshot(self) -> TaskSnapshot:
        """Return a transport-independent task view."""
        return TaskSnapshot(
            task_id=self.task_id,
            project_id=self.project_id,
            kind=self.kind,
            status=self.status,
            progress=(
                ProgressEvent.model_validate(self.progress).model_copy(update={"sequence": self.progress_seq})
                if self.progress is not None
                else None
            ),
            result=self.result,
            error_type=self.error_type,
            attempt=self.attempt,
        )


class TaskProgressRecord(BaseModel):
    """One ordered progress boundary committed by a leased worker."""

    model_config = ConfigDict(extra="forbid")
    """Reject unknown stored fields."""

    project_id: str
    """Owning project identity."""

    task_id: str
    """Owning task identity."""

    sequence: int = Field(ge=1)
    """One-based order within the task."""

    attempt: int = Field(ge=1)
    """Physical attempt that reported the event."""

    event: dict[str, Any]
    """Validated stage, message, count and timestamp."""

    occurred_at: AwareDatetime
    """UTC time of the progress report."""
