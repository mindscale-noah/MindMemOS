"""Persisted inputs and outcomes of one skill evaluation rollout."""

from typing import Self

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    JsonValue,
    SerializeAsAny,
    model_validator,
)

from ..skill.typing import SkillSetIdentity, SkillSnapshot


class Task(BaseModel):
    """Describe the initial task; subclasses may add task-specific fields."""

    id: str = Field(min_length=1, pattern=r"\S")
    """Stable identifier unique within the dataset, used by split manifests."""

    instruction: str = Field(min_length=1)
    """Initial instruction given to the agent."""


class RolloutRequest(BaseModel):
    """Identify the inputs whose completed result can be reused."""

    model_config = ConfigDict(extra="forbid")
    """Reject unknown fields when loading stored requests."""

    task: dict[str, JsonValue]
    """Task class identity and serialized fields used for resume checking."""

    reward_data: dict[str, JsonValue]
    """Private scoring input class and fields, including reference versions or hashes."""

    skills: list[SkillSnapshot]
    """Ordered candidate Skill snapshots; empty for a baseline run."""

    skill_set: SkillSetIdentity | None = None
    """Candidate identities and selection settings; None for legacy static records."""

    selected_skill_ids: list[str] | None = None
    """Ordered assembled member IDs; None before assembly or in legacy records."""

    assembled_skills: list[SkillSnapshot] | None = None
    """Actual task-time snapshots, or None until assembly has completed."""

    assembled_hashes: list[str] | None = None
    """Full content-and-resource hashes of the assembled snapshots, in the same order."""

    environment: str
    """Fully qualified concrete environment class name."""

    config: dict[str, JsonValue]
    """Execution and scoring settings, including dependency versions or identities."""

    @model_validator(mode="after")
    def validate_assembly(self) -> Self:
        """Reject incomplete or corrupt task-time content records."""
        if self.skill_set is not None and len(self.skill_set.skill_ids) != len(self.skills):
            raise ValueError("Collection IDs must match the source Skill count")
        expected_count = len(self.skills)
        if self.selected_skill_ids is not None:
            ids = self.selected_skill_ids
            if (
                self.skill_set is None
                or len(set(ids)) != len(ids)
                or any(key not in self.skill_set.skill_ids for key in ids)
            ):
                raise ValueError("Selected IDs must be unique members of the SkillSet")
            expected_count = len(ids)
        if self.assembled_skills is None:
            if self.assembled_hashes is not None or self.selected_skill_ids is not None:
                raise ValueError("Assembly hashes and selection require assembled snapshots")
        elif self.skill_set is not None and self.selected_skill_ids is None:
            raise ValueError("Assembled SkillSet requires selected IDs")
        elif len(self.assembled_skills) != expected_count or self.assembled_hashes != [
            snapshot.skill_hash for snapshot in self.assembled_skills
        ]:
            raise ValueError("Assembled snapshots and hashes must match the selected Skill count")
        return self


class Reward(BaseModel):
    """Define shared evaluation fields while allowing environment-specific metrics."""

    model_config = ConfigDict(extra="allow")
    """Preserve additional evaluation metrics when saving and loading results."""

    __pydantic_extra__: dict[str, JsonValue] = Field(init=False)
    """Additional JSON-serializable evaluation fields."""

    score: FiniteFloat
    """Finite evaluation score; its range is defined by the environment."""

    success: bool
    """Whether the task passed the environment's evaluation criteria."""

    detail: str
    """Detailed evaluation data as text, optionally containing encoded JSON."""


class Usage(BaseModel):
    """Record aggregate LLM usage for one rollout, including provider extensions."""

    model_config = ConfigDict(extra="allow")
    """Preserve additional usage counters and provider-specific breakdowns."""

    __pydantic_extra__: dict[str, JsonValue] = Field(init=False)
    """Additional JSON-serializable usage fields."""

    input_tokens: int = Field(default=0, ge=0, strict=True)
    """Total input tokens reported across the rollout's LLM calls."""

    output_tokens: int = Field(default=0, ge=0, strict=True)
    """Total output tokens reported across the rollout's LLM calls."""

    total_tokens: int = Field(default=0, ge=0, strict=True)
    """Total tokens reported by providers; zero by default, not inferred."""


class RolloutResult(BaseModel):
    """Return the agent output, sampled trajectory, and evaluated reward."""

    model_config = ConfigDict(extra="forbid")
    """Reject unknown fields when loading stored results."""

    output: JsonValue
    """Final agent answer or JSON-serializable task output."""

    trajectory: list[dict[str, JsonValue]]
    """Ordered agent interaction records, including tool calls where applicable."""

    reward: SerializeAsAny[Reward]
    """Evaluation outcome, including fields added by environment-specific subclasses."""

    usage: SerializeAsAny[Usage] = Field(default_factory=Usage)
    """LLM usage aggregated by the implementation, including evaluation calls."""

    started_at: AwareDatetime | None = None
    """UTC execution start time set by the environment; None before completion."""

    finished_at: AwareDatetime | None = None
    """UTC execution finish time set by the environment; None before completion."""

    duration_s: FiniteFloat | None = Field(default=None, ge=0)
    """Elapsed execution and scoring time in seconds, measured with a monotonic clock."""

    @model_validator(mode="after")
    def validate_timing(self) -> Self:
        """Require a complete timing record or an untimed implementation result.

        Returns:
            The validated result.

        Raises:
            ValueError: Only some timing fields are populated.
        """
        timing = (self.started_at, self.finished_at, self.duration_s)
        if any(value is not None for value in timing) and any(value is None for value in timing):
            raise ValueError("Timing fields must be populated together")
        return self
