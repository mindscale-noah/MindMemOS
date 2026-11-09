"""Inputs and outcomes shared by Skill evaluation functions."""

from pathlib import Path
from typing import Literal

from mindmemos_skill.envs.builtin_envs.agentskillos.typing import SkillID
from mindmemos_skill.envs.config import EnvironmentConfig
from mindmemos_skill.envs.data import DataSample
from mindmemos_skill.envs.typing import RolloutResult, Task
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, FiniteFloat, JsonValue, SerializeAsAny

BenchmarkName = Literal["livemath", "alfworld", "spreadsheetbench", "searchqa", "officeqa"]
ArmName = Literal["no_skill", "initial_skill", "trained_skill"]
BENCHMARKS: tuple[BenchmarkName, ...] = ("livemath", "alfworld", "spreadsheetbench", "searchqa", "officeqa")


class AgentSkillOSRetrievalTask(BaseModel):
    """Read the public query and private labels from an original AgentSkillOS task."""

    task_id: SkillID
    """Task identity used to associate the ranking with the original task."""

    prompt: str = Field(min_length=1)
    """Public instruction sent to the Skill Graph."""

    skills: list[SkillID] = Field(min_length=1)
    """Official reference Skill IDs used as binary retrieval labels."""


class RetrievalScoreReport(BaseModel):
    """Record metrics computed from the submitted AgentSkillOS rankings."""

    label_source: Literal["official_task_skills"] = "official_task_skills"
    """Source of the relevance labels used to score the submission."""

    metrics: dict[str, float]
    """Macro-average scores keyed by metric name."""

    per_task: dict[str, dict[str, float]]
    """Per-task scores keyed by task ID, then metric name."""


class SkillGraphAccuracyReport(BaseModel):
    """Report native AgentSkillOS success separately from execution failures."""

    model: str
    """Execution model used to generate task artifacts."""

    execution_top_k: int
    """Maximum number of Skills injected for each task."""

    accuracy: float | None
    """Official pass rate over successfully scored tasks, or None if there are none."""

    scoring_coverage: float
    """Fraction of the planned tasks with an official score."""

    errors: dict[str, str]
    """Execution and scorer failures keyed by task ID."""


class DatasetInputs(BaseModel):
    """Keep ordered fixed splits and their execution environment together."""

    splits: dict[str, list[DataSample]]
    """Train, val, and test samples in the committed manifest order."""

    environment: EnvironmentConfig
    """Validated environment with target model and resolved runtime inputs."""

    initial_skill: Path
    """Committed initial Skill directory for this benchmark."""

    split_file: Path
    """Committed manifest defining the complete fixed membership."""

    source: Path
    """Original downloaded benchmark directory, never modified by preparation."""


class CaseResult(BaseModel):
    """Represent a scored task or an explicit infrastructure error."""

    model_config = ConfigDict(extra="forbid")
    """Reject malformed saved task records."""

    task_id: str
    """Stable ID from the fixed test split."""

    score: FiniteFloat | None = None
    """Environment score; None when execution or scoring failed."""

    success: bool | None = None
    """Scorer pass flag; None for infrastructure failures."""

    metrics: dict[str, JsonValue] = Field(default_factory=dict)
    """Additional environment reward fields, retaining their original meanings."""

    error: str | None = None
    """Execution or scoring error; never substituted with a normal zero score."""

    duration_s: float | None = Field(default=None, ge=0)
    """Elapsed task time in seconds, including failed attempts when measured."""

    usage: dict[str, int | float] | None = None
    """Reported token counts and optional numeric cost in USD; None when unavailable."""

    started_at: AwareDatetime | None = None
    """Recorded rollout start time; None when execution did not publish timing."""

    finished_at: AwareDatetime | None = None
    """Recorded rollout completion time, used to recover the pipeline test's observed wall span."""

    result_path: Path
    """Original rollout result or error record used to build this row."""


class ArmResult(BaseModel):
    """Save one independently executed test repeat."""

    arm: ArmName
    """Empty, initial, or exact trained Skill condition."""

    repeat: int = Field(ge=1)
    """One-based independent test repetition."""

    task_ids: list[str]
    """Planned test task IDs and order, including unfinished tasks."""

    cases: list[CaseResult] = Field(default_factory=list)
    """Completed task outcomes and explicit errors."""

    wall_seconds: float | None = Field(default=None, ge=0)
    """Wall time for this repeat; None when unavailable from pipeline artifacts."""

    seed: int | None = None
    """Actual simulator or target sampling seed; None if unspecified."""


class BenchmarkReport(BaseModel):
    """Record one benchmark independently of other score scales."""

    benchmark: BenchmarkName
    """Dataset and scorer selected for this benchmark."""

    arms: list[ArmResult] = Field(default_factory=list)
    """Results for each requested arm and repeat."""

    error: str | None = None
    """Benchmark-level failure, including runtime or training initialization."""

    initial_version_id: str | None = None
    """Registered initial immutable version; None if no service was needed."""

    trained_version_id: str | None = None
    """Exact frozen publication used for every trained repeat."""

    training_seconds: float | None = None
    """Training pipeline wall time including its validation and first test."""

    optimizer_usage: dict[str, int] | None = None
    """Optimizer-only usage when available; rollout usage is not a substitute."""


class SuiteReport(BaseModel):
    """Save the requested scope and per-benchmark outcomes of one fresh run."""

    output_dir: Path
    """Isolated run directory containing configuration and raw result references."""

    smoke: bool
    """Whether fixed split prefixes and minimal forest settings were selected."""

    benchmarks: list[BenchmarkReport]
    """All requested benchmarks, including failures and unfinished conditions."""


class EnvTestSummary(BaseModel):
    """Report the required counts, elapsed time, usage, and outcome rate for one test."""

    environment: str
    """Registered environment used for this run."""

    mode: Literal["no_skill", "static_skill", "treeskill"]
    """Exactly one execution mode; static Skills are injected as a single collection."""

    planned: int
    """Number of selected tasks after split and limit selection."""

    attempted: int
    """Number of tasks that ran, including execution failures."""

    completed: int
    """Tasks that returned a normal scored result, including zero rewards."""

    not_started: int = 0
    """Selected tasks never attempted because the service stopped earlier."""

    errors: int
    """Tasks that failed to execute or score, such as API timeouts."""

    successful: int
    """Completed tasks whose native reward.success is true."""

    success_rate: float | None
    """Successful divided by completed; None when no task completed normally."""

    wall_seconds: float | None
    """Elapsed evaluation wall time, or None when an interrupted phase has no timing record."""

    input_tokens: int | None
    """Total input tokens, or None if any task has unreported usage."""

    output_tokens: int | None
    """Total output tokens, or None if any task has unreported usage."""

    total_tokens: int | None
    """Total tokens, or None if any task has unreported usage."""

    reported_usage: dict[str, int]
    """Known input_tokens, output_tokens, and total_tokens; partial when usage_missing is positive."""

    usage_missing: int
    """Task count with unavailable usage; failed requests may still have incurred tokens."""

    trajectory_file: Path
    """JSONL audit file with task outcomes, including errors and explicitly unstarted tasks."""


class TaskTrajectory(BaseModel):
    """Persist one attempted task and its available native trajectory as a JSONL record."""

    position: int
    """Zero-based position in the selected task list; JSONL rows are written on completion."""

    task: SerializeAsAny[Task]
    """Public task input, preserving environment-specific fields."""

    outcome: CaseResult
    """Task ID, timing, token usage, native success flag, and any execution error."""

    result: SerializeAsAny[RolloutResult] | None
    """Native output, trajectory, and reward; None if execution failed before returning a result."""
