"""Data contracts for local Skill analysis and optimization."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

from ..contracts import SkillBundle, SkillRuntimeSpec
from .skill import Skill
from .task import Task
from .trajectory import Trajectory


class SkillFinding(BaseModel):
    """One actionable observation produced while analyzing a Skill."""

    model_config = ConfigDict(extra="forbid")

    category: str
    message: str
    severity: Literal["info", "warning", "error"] = "info"
    evidence: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SkillAnalysisRequest(BaseModel):
    """Inputs needed to analyze one Skill without SDK or cloud state."""

    model_config = ConfigDict(extra="forbid")

    skill: Skill
    trajectories: list[Trajectory] = Field(default_factory=list)
    options: dict[str, Any] = Field(default_factory=dict)
    request_id: str | None = None


class SkillAnalysisResult(BaseModel):
    """Transport-neutral result of one Skill analysis."""

    model_config = ConfigDict(extra="forbid")

    summary: str
    findings: list[SkillFinding] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Trace2SkillInput(BaseModel):
    """One base Skill plus offline traces, collection tasks, or both."""

    model_config = ConfigDict(extra="forbid")

    base_skill: Skill
    trajectories: list[Trajectory] = Field(default_factory=list)
    tasks: list[Task] = Field(default_factory=list)
    run_id: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_source(self) -> Trace2SkillInput:
        if not self.trajectories and not self.tasks:
            raise ValueError("provide at least one non-empty trajectory source: trajectories or tasks")
        return self


class EvolveInput(BaseModel):
    """Shared input boundary for algorithms that run a complete Skill evolution loop."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    run_id: str = Field(min_length=1)
    base_skill: Skill
    train_tasks: list[Task] = Field(min_length=1)
    validation_tasks: list[Task] = Field(default_factory=list)
    test_tasks: list[Task] = Field(default_factory=list)


class EvolveOutput(BaseModel):
    """Shared output boundary returned by complete Skill evolution algorithms."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    run_id: str
    final_skill: Skill
    changed: bool
    trajectories: list[Trajectory] = Field(default_factory=list)
    finished_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class SkillCandidate(BaseModel):
    """Unpersisted Skill contents without version identity or lifecycle state."""

    model_config = ConfigDict(extra="forbid")

    blob: dict[str, str]
    resources: dict[str, str] = Field(default_factory=dict)
    runtime_type: str = "static"
    runtime_schema_version: int = Field(default=1, ge=1)
    runtime_metadata: dict[str, JsonValue] = Field(default_factory=dict)
    commit_message: str | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("blob")
    @classmethod
    def normalize_blob(cls, value: dict[str, str]) -> dict[str, str]:
        bundle = SkillBundle.from_files(value)
        return {item.path: item.content for item in bundle.files}

    @model_validator(mode="after")
    def validate_files(self) -> SkillCandidate:
        SkillRuntimeSpec(
            runtime_type=self.runtime_type,
            runtime_schema_version=self.runtime_schema_version,
            runtime_metadata=self.runtime_metadata,
        )
        if set(self.blob) != {"SKILL.md"}:
            raise ValueError("Skill candidate blob must contain exactly one SKILL.md file")
        invalid_paths = [path for path in (*self.blob, *self.resources) if not path]
        if invalid_paths:
            raise ValueError("Skill candidate file paths must not be empty")
        if self.blob.keys() & self.resources.keys():
            raise ValueError("Skill candidate blob and resources may not contain the same path")
        return self


ReportT = TypeVar("ReportT")


class Trace2SkillOutput(BaseModel, Generic[ReportT]):
    """Algorithm output: an optional content candidate and a typed audit report."""

    model_config = ConfigDict(extra="forbid")

    candidate: SkillCandidate | None = None
    trajectories: list[Trajectory] = Field(default_factory=list)
    report: ReportT

    @property
    def changed(self) -> bool:
        return self.candidate is not None


__all__ = [
    "EvolveInput",
    "EvolveOutput",
    "SkillAnalysisRequest",
    "SkillAnalysisResult",
    "SkillFinding",
    "SkillCandidate",
    "Trace2SkillInput",
    "Trace2SkillOutput",
]
