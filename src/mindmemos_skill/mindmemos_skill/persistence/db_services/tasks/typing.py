"""Durable task records stored in the existing Skill database."""

from typing import Any

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from ....infra.task_manager.typing import TaskSnapshot, TaskSpec, TaskStatus


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

    trace_context: dict[str, str] = Field(default_factory=dict)
    """Submission W3C headers used for publication and restored worker spans."""

    request_id: str | None = None
    """Original HTTP or Python submission identity."""

    started_at: AwareDatetime | None = None
    """Start of the current physical attempt."""

    finished_at: AwareDatetime | None = None
    """Final result commit time; absent while recovery remains possible."""

    status: TaskStatus
    """Current task lifecycle state."""

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
        return TaskSpec(
            task_id=self.task_id,
            project_id=self.project_id,
            kind=self.kind,
            payload=self.payload,
            trace_context=self.trace_context,
            request_id=self.request_id,
        )

    def snapshot(self) -> TaskSnapshot:
        """Return a transport-independent task view."""
        return TaskSnapshot(
            task_id=self.task_id,
            project_id=self.project_id,
            kind=self.kind,
            status=self.status,
            result=self.result,
            error_type=self.error_type,
            attempt=self.attempt,
            accepted_at=self.created_at,
            started_at=self.started_at,
            finished_at=self.finished_at,
        )
