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


class HeldOutSummary(BaseModel):
    """Aggregate rewards for one final evaluation split."""

    count: int
    """Number of completed tasks."""

    success_rate: float | None
    """Fraction of successful tasks, or None for an empty split."""

    mean_score: float | None
    """Mean reward score, or None for an empty split."""


class HeldOutReport(HeldOutSummary):
    """Evaluation scores, execution costs, and trajectory location."""

    errors: int = 0
    """Failed executions; evaluation raises before reporting an execution error."""

    wall_seconds: float
    """Elapsed seconds for this evaluation, including reused result processing."""

    task_seconds: float | None
    """Sum of task durations, or None if any duration is unreported."""

    input_tokens: int | None
    """Total input tokens, or None if any task has unreported usage."""

    output_tokens: int | None
    """Total output tokens, or None if any task has unreported usage."""

    total_tokens: int | None
    """Total tokens, or None if any task has unreported usage."""

    trajectory_file: str
    """Path to the evaluation trajectory JSONL file."""


class HeldOutEvaluation(BaseModel):
    """Return evaluation artifacts without reading them back from disk."""

    summary: HeldOutSummary
    """Reward aggregates also saved in summary.json."""

    report: HeldOutReport
    """Scores and execution costs also saved in report.json."""

    tasks: list[HeldOutTaskResult]
    """Ordered task records also saved in results.json."""
