"""Typed contracts for trajectory-grounded virtual-Skill decomposition."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ....typing import EvolveInput, EvolveOutput, SkillCandidate
from ..skill_grpo_with_replay_buffer.contracts import RolloutOutcome
from .config import TaskVirtualSkillRunConfig


class TrajectoryKeyPoints(BaseModel):
    """One independent, evidence-only summary of a physical trajectory."""

    model_config = ConfigDict(extra="forbid")

    trajectory_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    task_goal: str = Field(min_length=1)
    task_family: str = Field(min_length=1)
    key_actions: list[str] = Field(min_length=1)
    turning_points: list[str] = Field(default_factory=list)
    skill_usage: list[str] = Field(default_factory=list)
    outcome: str = Field(min_length=1)
    score: float | None = None


class VirtualSkillDefinition(BaseModel):
    """One subtask boundary grounded in trajectories and exact source excerpts."""

    model_config = ConfigDict(extra="forbid")

    skill_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    supporting_trajectory_ids: list[str] = Field(min_length=1)
    source_excerpts: list[str] = Field(min_length=1)


class TaskVirtualSkillPlan(BaseModel):
    """Flat subtask plan; generated guidance is forbidden by construction."""

    model_config = ConfigDict(extra="forbid")

    skill_name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    virtual_skills: list[VirtualSkillDefinition] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unique_skills(self) -> TaskVirtualSkillPlan:
        ids = [item.skill_id for item in self.virtual_skills]
        if len(ids) != len(set(ids)):
            raise ValueError("virtual skill_id values must be unique")
        return self


class TaskVirtualSkillInput(EvolveInput):
    """One initial Skill and task pool for exactly one rollout batch."""

    config: TaskVirtualSkillRunConfig


class VirtualSkillArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    component_id: str
    name: str
    description: str
    supporting_trajectory_ids: list[str]
    source_excerpts: list[str]
    relative_path: str
    markdown: str


class TaskVirtualSkillResult(EvolveOutput):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    candidate: SkillCandidate | None = None
    plan: TaskVirtualSkillPlan
    artifacts: list[VirtualSkillArtifact]
    rollouts: list[RolloutOutcome]
    trajectory_summaries: list[TrajectoryKeyPoints]
    sampled_trajectory_ids: list[str]
    failed_summary_trajectory_ids: list[str] = Field(default_factory=list)
    raw_decomposition_response: str


__all__ = [
    "TaskVirtualSkillInput",
    "TaskVirtualSkillPlan",
    "TaskVirtualSkillResult",
    "TrajectoryKeyPoints",
    "VirtualSkillArtifact",
    "VirtualSkillDefinition",
]
