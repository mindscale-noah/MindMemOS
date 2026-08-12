"""Contracts for dynamic retry and direct virtual-Skill changes."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ....typing import Skill
from ..skill_grpo_with_replay_buffer.contracts import RolloutOutcome
from .models import TrajectoryKeyPoints


class TaskSkillChange(BaseModel):
    """Exactly one create, update, or noop decision for one task."""

    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(min_length=1)
    operation: Literal["create", "update", "noop"]
    skill_id: str | None = Field(default=None, pattern=r"^[a-z0-9][a-z0-9-]*$")
    name: str | None = None
    description: str | None = None
    content: str | None = None
    diagnosis: str = Field(min_length=1)
    evidence_trajectory_ids: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_operation_fields(self) -> TaskSkillChange:
        def present(value: str | None) -> bool:
            return isinstance(value, str) and bool(value.strip())

        if self.operation == "create":
            if not all(present(value) for value in (self.skill_id, self.name, self.description, self.content)):
                raise ValueError("create requires skill_id, name, description, and content")
        elif self.operation == "update":
            if not present(self.skill_id):
                raise ValueError("update requires skill_id")
            supplied = (self.name, self.description, self.content)
            if not any(present(value) for value in supplied):
                raise ValueError("update requires at least one of name, description, or content")
            if any(value is not None and not present(value) for value in supplied):
                raise ValueError("supplied update fields must not be blank")
        elif any(value is not None for value in (self.skill_id, self.name, self.description, self.content)):
            raise ValueError("noop must not include Skill fields")
        return self


class VirtualSkillMerge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation: Literal["create", "update"]
    skill_id: str = Field(min_length=1)
    name: str | None = None
    description: str | None = None
    source_task_ids: list[str] = Field(min_length=1)
    original_content: str | None = None
    revised_content: str | None = None
    change_summary: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_operation_fields(self) -> VirtualSkillMerge:
        if self.operation == "create":
            if not self.name or not self.description or not self.revised_content:
                raise ValueError("create merge requires name, description, and revised_content")
            if self.original_content is not None:
                raise ValueError("create merge must not have original_content")
        else:
            if self.original_content is None:
                raise ValueError("update merge requires original_content")
            if not any(value is not None for value in (self.name, self.description, self.revised_content)):
                raise ValueError("update merge requires at least one changed field")
        return self


class TaskVirtualSkillRefinementResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_run_id: str
    run_id: str
    before_skill: Skill
    after_skill: Skill
    retry_rollouts: list[RolloutOutcome] = Field(default_factory=list)
    retry_summaries: list[TrajectoryKeyPoints] = Field(default_factory=list)
    failed_summary_trajectory_ids: list[str] = Field(default_factory=list)
    changes: list[TaskSkillChange]
    merges: list[VirtualSkillMerge]


__all__ = [
    "TaskSkillChange",
    "TaskVirtualSkillRefinementResult",
    "VirtualSkillMerge",
]
