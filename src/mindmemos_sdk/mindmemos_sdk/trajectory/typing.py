"""Public event and receipt types for unified trajectory reporting."""

from __future__ import annotations

import json
from typing import Literal, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

from ..memory.models import AddMode, DialogueMessage
from ..skills.typing import NonemptyIdentity


class TrajectoryMessage(DialogueMessage):
    """One ordered dialogue or tool message captured by a Harness."""

    model_config = ConfigDict(extra="forbid")
    """Accept only fields intentionally included in the delivery contract."""

    role: NonemptyIdentity
    """Role of the message author, such as user, assistant, or tool."""

    tool_calls: list[dict[str, JsonValue]] | None = None
    """Structured call objects in emission order, including IDs when available."""

    tool_call_id: NonemptyIdentity | None = None
    """Call ID that this tool result answers, when the Harness supplies one."""

    is_error: bool | None = None
    """Whether this tool result represents a tool failure, when known."""

    @field_validator("tool_calls")
    @classmethod
    def validate_tool_calls(cls, calls: list[dict[str, JsonValue]] | None) -> list[dict[str, JsonValue]] | None:
        """Reject non-finite JSON numbers before sending either request."""
        if calls is not None:
            json.dumps(calls, allow_nan=False)
        return calls


class TrajectoryEvent(BaseModel):
    """One captured execution to deliver to Memory and Skill."""

    model_config = ConfigDict(extra="forbid")
    """Reject unrecognized event fields."""

    event_id: NonemptyIdentity
    """Stable caller-assigned identity used to correlate delivery receipts."""

    source: NonemptyIdentity
    """Harness or application that captured the execution."""

    messages: list[TrajectoryMessage] = Field(min_length=1)
    """Ordered, caller-filtered messages. Message content is text."""

    skill_version_ids: list[NonemptyIdentity] = Field(default_factory=list)
    """Exact loaded versions. Empty means no exact version was identified."""

    user_id: NonemptyIdentity | None = None
    """Business user, falling back to the client default."""

    agent_id: NonemptyIdentity | None = None
    """Executing agent, falling back to the client default. Required for Skill delivery."""

    app_id: NonemptyIdentity | None = None
    """Application scope for Memory, falling back to the client default."""

    session_id: NonemptyIdentity | None = None
    """Harness session, falling back to the client default."""

    turn_id: NonemptyIdentity | None = None
    """Turn within the Harness session, if available."""

    started_at: AwareDatetime | None = None
    """Execution start time with a timezone, if available."""

    finished_at: AwareDatetime | None = None
    """Execution end time with a timezone, if available."""

    status: NonemptyIdentity = "completed"
    """Execution outcome as observed by the caller."""

    memory_mode: AddMode = "async"
    """Whether Memory extraction is enqueued or completes before returning."""

    @model_validator(mode="after")
    def validate_timing(self) -> Self:
        """Require the finish time to follow the start time."""
        if self.started_at and self.finished_at and self.finished_at < self.started_at:
            raise ValueError("finished_at must not precede started_at")
        return self


class TrajectoryDelivery(BaseModel):
    """Outcome of one destination write."""

    status: Literal["queued", "completed", "stored", "failed", "unknown"]
    """Observed result. Unknown means the server may already have accepted the write."""

    code: str | None = None
    """Service outcome code or SDK failure reason."""

    request_id: str | None = None
    """Server request ID for support and correlation, when available."""

    http_status: int | None = None
    """HTTP status from a rejected or indeterminate request, when available."""

    trajectory_id: str | None = None
    """Stored Skill trajectory ID returned by the Skill service."""


class TrajectoryReportResult(BaseModel):
    """Independent delivery receipts for one execution."""

    event_id: str
    """Caller-assigned event identity echoed in the result."""

    memory: TrajectoryDelivery | None = None
    """Memory receipt, or None when the caller excluded Memory."""

    skill: TrajectoryDelivery | None = None
    """Receipt for the shared Skill trajectory, or None when Skill was excluded."""

    @property
    def has_failures(self) -> bool:
        """Return whether any attempted delivery failed or is indeterminate."""
        return any(
            delivery is not None and delivery.status in {"failed", "unknown"} for delivery in (self.memory, self.skill)
        )
