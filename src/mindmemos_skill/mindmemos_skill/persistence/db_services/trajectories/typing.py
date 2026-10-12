"""Validated records owned by this business data service."""

from datetime import date
from typing import Literal

from pydantic import AwareDatetime, Field, FiniteFloat, JsonValue, model_validator

from ....common.time_utils import utcnow
from ...typing import RecordModel, TrajectoryCounts


class TrajectoryDayCounts(TrajectoryCounts):
    """Stored trajectory counts grouped by creation day."""

    day: date
    """Trajectory creation day in the statistics timezone."""


class TrajectoryStatistics(RecordModel):
    """Bounded aggregation for one version without trajectory payloads."""

    skill_version_id: str
    """Exact queried Skill version."""

    timezone: str = "Asia/Shanghai"
    """Calendar timezone used by the fixed 30-day window."""

    totals: TrajectoryCounts
    """All-time counts for the selected version, independent of chart selection."""

    days: list[TrajectoryDayCounts] = Field(min_length=30, max_length=30)
    """Thirty consecutive calendar days, including zero-count days."""

    last_evolved_at: AwareDatetime | None = None
    """Latest recorded evolution start time; None when no matching audit run exists."""


class TrajectoryConsumption(RecordModel):
    """First successful evolution use of a stored trajectory."""

    project_id: str = Field(min_length=1)
    """Project owning the trajectory and evolution run."""

    trajectory_id: str = Field(min_length=1)
    """Stored trajectory processed by the evolver."""

    skill_version_id: str = Field(min_length=1)
    """Exact input Skill version associated with the trajectory."""

    run_id: str = Field(min_length=1)
    """First evolution run that successfully processed this trajectory."""

    consumed_at: AwareDatetime = Field(default_factory=utcnow)
    """Time the successful use was recorded."""


class TrajectoryRecord(RecordModel):
    """Complete insertable record for trajectories."""

    project_id: str = Field(min_length=1)
    """Project that owns this physical trajectory attempt."""

    trajectory_id: str = Field(min_length=1)
    """Local identity of one physical rollout attempt."""

    trajectory_hash: str = Field(min_length=1)
    """Hash of the ordered local events or reported messages."""

    source: Literal["local", "reported"] = "local"
    """Whether this attempt was collected locally or reported by a Harness."""

    task_id: str = Field(min_length=1)
    """Task identity from local data or generated for a reported attempt."""

    rollout_id: str = Field(min_length=1)
    """Logical rollout identity shared by retries."""

    attempt_no: int = Field(default=0, ge=0)
    """Zero-based physical attempt number."""

    rollout_type: str = "inference"
    """Purpose of the rollout, such as inference or training."""

    task_instruction: str
    """Task instruction, or empty when the report contained no user message."""

    task_system_prompt: str | None = None
    """Optional task-level system prompt."""

    task_metadata: dict[str, JsonValue] = Field(default_factory=dict)
    """Dataset identity, task configuration, and extra task fields."""

    env_ref: str = Field(min_length=1)
    """Environment implementation or registered name."""

    env_metadata: dict[str, JsonValue] = Field(default_factory=dict)
    """Environment configuration and reproducibility information."""

    running_dir: str | None = None
    """Optional local execution directory."""

    injected_skills: list[JsonValue] = Field(default_factory=list)
    """Ordered Skill configurations injected into the run."""

    skill_bindings: list[JsonValue] = Field(default_factory=list)
    """Local Skill/version references and usage information."""

    agent_type: str = "unknown"
    """Agent implementation identifier."""

    agent_profile: dict[str, JsonValue] = Field(default_factory=dict)
    """Agent configuration snapshot without credentials."""

    status: str = "running"
    """Execution outcome recorded by the rollout owner."""

    trajectory: list[JsonValue] = Field(default_factory=list)
    """Ordered interaction records, including tool calls."""

    reward_score: FiniteFloat | None = None
    """Optional finite environment-defined score."""

    reward_detail: str | None = None
    """Optional explanation of the reward."""

    reward_metadata: dict[str, JsonValue] = Field(default_factory=dict)
    """Extra evaluation metrics, including success when available."""

    started_at: AwareDatetime | None = Field(default_factory=utcnow)
    """Execution start time; None when the Harness did not report it."""

    finished_at: AwareDatetime | None = None
    """Execution completion time; absent while running."""

    n_turn: int = Field(default=0, ge=0)
    """Number of agent interaction turns."""

    error_info: str | None = None
    """Optional execution failure details."""

    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    """Additional local execution annotations."""

    created_at: AwareDatetime = Field(default_factory=utcnow)
    """Time this trajectory record was stored."""

    skill_id: str | None = None
    """Exact Skill family associated with a reported attempt, if any."""

    skill_version_id: str | None = None
    """Exact loaded Skill version associated with a reported attempt, if any."""

    user_id: str | None = None
    """Business user identity supplied with a reported attempt, if any."""

    agent_id: str | None = None
    """Agent identity supplied with a reported attempt, if any."""

    session_id: str | None = None
    """Optional logical session associated with a reported attempt."""

    received_at: AwareDatetime | None = None
    """Server receipt time for a reported attempt; absent for local records."""

    @model_validator(mode="after")
    def validate_timing(self):
        """Preserve source-specific fields and reject reversed timestamps."""
        if self.started_at is not None and self.finished_at is not None and self.finished_at < self.started_at:
            raise ValueError("finished_at must not precede started_at")
        if self.source == "local" and (not self.task_instruction or self.started_at is None):
            raise ValueError("Local trajectories require a task instruction and start time")
        if self.source == "reported" and not all((self.skill_id, self.skill_version_id, self.received_at)):
            raise ValueError("Reported trajectories require a Skill version and receipt time")
        return self


class TrajectorySummary(RecordModel):
    """Projected fields returned by a reported trajectory page."""

    trajectory_id: str = Field(min_length=1)
    """Physical attempt identity and receipt-order tie breaker."""

    task_id: str = Field(min_length=1)
    """Task identity associated with the reported attempt."""

    skill_id: str = Field(min_length=1)
    """Skill family associated with the report."""

    skill_version_id: str = Field(min_length=1)
    """Exact Skill version loaded during execution."""

    user_id: str | None = Field(default=None, min_length=1)
    """Business user who reported the attempt, if supplied."""

    agent_id: str | None = Field(default=None, min_length=1)
    """Agent that reported the attempt, if supplied."""

    status: str
    """Caller-declared execution outcome."""

    started_at: AwareDatetime | None
    """Execution start time, if supplied."""

    finished_at: AwareDatetime | None
    """Execution finish time, if supplied."""

    received_at: AwareDatetime
    """Server receipt time used by the page cursor."""
