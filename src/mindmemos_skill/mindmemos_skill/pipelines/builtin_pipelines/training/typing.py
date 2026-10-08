"""Stable per-task results emitted by the training pipeline's final evaluation."""

from pydantic import BaseModel, ConfigDict, Field


class HeldOutTaskResult(BaseModel):
    """Identify one completed held-out rollout without copying private inputs."""

    model_config = ConfigDict(extra="forbid")
    """Reject unknown result fields."""

    task_id: str
    """Dataset task ID supplied in the training request."""

    score: float
    """Score returned by the environment's reward implementation."""

    success: bool
    """Success flag returned by the environment's reward implementation."""

    duration_s: float | None = None
    """Elapsed rollout seconds, or None when the environment did not report timing."""

    usage: dict[str, int] = Field(default_factory=dict)
    """Provider token counters when available; an empty mapping means unreported."""

    result_path: str
    """Path to the completed rollout record relative to the evaluation directory."""


class HeldOutResultIndex(BaseModel):
    """List all completed task results for one final validation or test split."""

    model_config = ConfigDict(extra="forbid")
    """Reject unknown index fields."""

    tasks: list[HeldOutTaskResult] = Field(default_factory=list)
    """Completed tasks in the same order as the request split."""
