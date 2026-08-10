"""Input contract for one physical Agent execution attempt."""

from __future__ import annotations

from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from .env import Environment
from .skill import Skill
from .task import Task
from .trajectory import Rollout


class AgentExecutionRequest(BaseModel):
    """Everything an Agent needs to produce exactly one trajectory row.

    Attempt identity and environment are explicit fields instead of control
    values hidden in metadata. This makes the request structurally match the
    eventual ``TrajectoryRecord``.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    trajectory_id: str = Field(default_factory=lambda: uuid4().hex, min_length=1)
    task: Task
    rollout: Rollout
    environment: Environment = Field(default_factory=Environment)
    skills: list[Skill] = Field(default_factory=list)
    options: dict[str, JsonValue] = Field(default_factory=dict)
    """Per-attempt Agent configuration overrides validated by the implementation."""

    metadata: dict[str, JsonValue] = Field(default_factory=dict)


__all__ = ["AgentExecutionRequest"]
