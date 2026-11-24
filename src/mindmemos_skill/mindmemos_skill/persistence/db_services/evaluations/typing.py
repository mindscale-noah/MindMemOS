"""Independent stored results from Skill and trajectory judges."""

from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, FiniteFloat, JsonValue

from ....common.time_utils import utcnow


class EvaluationRecord(BaseModel):
    """One immutable analysis input identity and its latest result state."""

    model_config = ConfigDict(extra="forbid")
    """Reject fields not explicitly owned by evaluation storage."""

    evaluation_id: str = Field(min_length=1)
    """Globally unique evaluation result identity."""

    project_id: str = Field(min_length=1)
    """Project scope injected by the persistence repository."""

    evaluation_key: str = Field(min_length=1)
    """Stable digest of every result-affecting input."""

    judge_type: Literal["static_skill", "trajectory"]
    """Analysis family that produced the result."""

    skill_id: str | None = None
    """Optional owning Skill family identity."""

    skill_version_id: str | None = None
    """Optional exact Skill version identity."""

    trajectory_id: str | None = None
    """Optional exact trajectory identity."""

    algorithm: str = Field(min_length=1)
    """Registered analyzer implementation name."""

    algorithm_version: str = Field(min_length=1)
    """Analyzer behavior version."""

    prompt_version: str = Field(min_length=1)
    """Prompt family revision used by the analyzer."""

    status: Literal["completed", "failed"]
    """Only completed rows are reusable as cache hits."""

    summary: str | None = None
    """Successful human-readable assessment."""

    score_mean: FiniteFloat | None = None
    """Mean static score when the analyzer supplies one."""

    scores: dict[str, JsonValue] = Field(default_factory=dict)
    """Named static score dimensions."""

    success: bool | None = None
    """Trajectory success verdict when available."""

    confidence: str | None = None
    """Trajectory verdict confidence when available."""

    skill_usage_rate: FiniteFloat | None = None
    """Trajectory Skill usage ratio when available."""

    metrics: dict[str, JsonValue] = Field(default_factory=dict)
    """Complete transport-neutral analyzer metrics."""

    findings: list[dict[str, JsonValue]] = Field(default_factory=list)
    """Persisted analyzer findings."""

    artifacts: dict[str, str] = Field(default_factory=dict)
    """Optional analyzer-generated text artifacts."""

    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    """Sanitized provenance and analyzer metadata."""

    error: str | None = None
    """Failure description without model credentials."""

    created_at: AwareDatetime = Field(default_factory=utcnow)
    """Initial result creation time."""

    updated_at: AwareDatetime = Field(default_factory=utcnow)
    """Latest result state update time."""


__all__ = ["EvaluationRecord"]
