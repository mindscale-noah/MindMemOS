"""Validated records owned by this business data service."""

from pydantic import AwareDatetime, Field, FiniteFloat, JsonValue, model_validator

from ....common.time_utils import utcnow
from ...typing import RecordModel


class TrajectoryRecord(RecordModel):
    """Complete insertable record for trajectories."""

    project_id: str = Field(min_length=1)
    """Project that owns this physical rollout attempt."""

    trajectory_id: str = Field(min_length=1)
    """Local identity of one physical rollout attempt."""

    trajectory_hash: str = Field(min_length=1)
    """Hash of the local trajectory representation."""

    task_id: str = Field(min_length=1)
    """Task identity within its dataset."""

    rollout_id: str = Field(min_length=1)
    """Logical rollout identity shared by retries."""

    attempt_no: int = Field(default=0, ge=0)
    """Zero-based physical attempt number."""

    rollout_type: str = "inference"
    """Purpose of the rollout, such as inference or training."""

    task_instruction: str = Field(min_length=1)
    """Instruction presented for this attempt."""

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

    started_at: AwareDatetime = Field(default_factory=utcnow)
    """Execution start time."""

    finished_at: AwareDatetime | None = None
    """Execution completion time; absent while running."""

    n_turn: int = Field(default=0, ge=0)
    """Number of agent interaction turns."""

    error_info: str | None = None
    """Optional execution failure details."""

    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    """Additional local execution annotations."""

    created_at: AwareDatetime = Field(default_factory=utcnow)
    """Time this local record was stored."""

    @model_validator(mode="after")
    def validate_timing(self):
        """Reject reversed execution timestamps."""
        if self.finished_at is not None and self.finished_at < self.started_at:
            raise ValueError("finished_at must not precede started_at")
        return self
