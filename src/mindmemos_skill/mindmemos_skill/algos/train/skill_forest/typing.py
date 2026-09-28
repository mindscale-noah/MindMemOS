"""Run-local forest data and validated atomic patch contracts."""

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StrictInt, model_validator

from ....envs.typing import RolloutResult, Task
from ....skill import Skill


class RankingResponse(BaseModel):
    """Optimizer-selected indices from a bounded candidate edit pool."""

    selected_indices: list[StrictInt]
    """Zero-based indices in optimizer priority order."""


class Edit(BaseModel):
    """One exact textual edit to the Skill entrypoint."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    """Reject unknown patch operations and preserve generated evidence."""

    op: Literal["append", "insert_after", "replace", "delete"]
    """Operation applied sequentially to SKILL.md."""

    target: str = ""
    """Unique exact source substring required by non-append operations."""

    content: str = ""
    """Replacement or inserted Markdown; unused for deletion."""

    support_count: int | None = Field(default=None, ge=1)
    """Number of supporting trajectories or patches carried by reduction."""

    source_type: Literal["failure", "success"] | None = None
    """Outcome group supplying the edit evidence."""

    merge_level: int | None = Field(default=None, ge=1)
    """One-based hierarchy level assigned by patch aggregation."""

    @model_validator(mode="after")
    def validate_edit(self) -> Self:
        """Reject incomplete edits before modifying any content."""
        if self.op != "append" and not self.target:
            raise ValueError("Non-append edits require a target")
        if self.op != "delete" and not self.content.strip():
            raise ValueError("Non-delete edits require content")
        return self


class Patch(BaseModel):
    """A candidate edge containing zero or more atomic edits."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    """Keep model output restricted to the patch protocol."""

    reasoning: str = ""
    """Explanation of the generalizable changes."""

    edits: list[Edit] = Field(default_factory=list)
    """Ordered atomic edits; empty means no change is warranted."""


class ReflectionResponse(BaseModel):
    """Structured reflection result containing a proposed patch."""

    model_config = ConfigDict(extra="allow")
    """Retain source failure_summary or success_patterns without exposing private data."""

    batch_size: int = Field(default=0, ge=0)
    """Analyst-reported trajectory count, retained for response inspection."""

    patch: Patch
    """Nested reasoning and edits returned by reflection."""


class TreeSplit(BaseModel):
    """Stable task membership for one tree."""

    model_config = ConfigDict(frozen=True)
    """Prevent reassignment of tree membership."""

    train_ids: tuple[str, ...]
    """Tasks used once per parent expansion to collect reflection evidence."""

    validation_ids: tuple[str, ...]
    """Fixed tasks used for all parent-relative gates in this tree."""


class Node(BaseModel):
    """Complete candidate and its fully observed tree-local score vector."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)
    """Allow the public Skill data object without attaching tree behavior."""

    id: str
    """Stable run-local node identifier, also used for artifact directories."""

    tree_id: int
    """Index of the independent optimizer owning this node."""

    parent_id: str | None = None
    """Parent identifier; None only for a root."""

    depth: int = 0
    """Number of applied patches from the initial Skill."""

    skill: Skill
    """Complete candidate content with original auxiliary resources."""

    scores: dict[str, float]
    """Full tree-local validation vector for accepted nodes."""

    gain: float = 0.0
    """Strict validation improvement over the immediate parent."""


class ArtifactRecord(BaseModel):
    """Strict common contract for forest-owned JSON objects."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    """Reject misspelled fields and non-finite audit values."""


class ExecutionStatus(ArtifactRecord):
    """Shared lifecycle fields for runs and physical calls."""

    status: Literal["running", "completed", "failed"]
    """Current execution state."""

    duration_s: float | None = Field(default=None, ge=0)
    """Elapsed wall time for a terminal state."""

    error: str | None = None
    """Exception description when execution failed."""

    @model_validator(mode="after")
    def validate_state(self) -> Self:
        """Require terminal evidence and reject contradictory lifecycle fields."""
        if self.status == "running":
            if self.duration_s is not None or self.error is not None:
                raise ValueError("Running records cannot contain terminal evidence")
        elif self.duration_s is None:
            raise ValueError("Terminal records require duration_s")
        if (self.error is not None) != (self.status == "failed"):
            raise ValueError("Only failed records require error")
        return self


class RunStatus(ExecutionStatus):
    """Run lifecycle saved in status.json."""

    content_hash: str | None = None
    """Selected Skill identity for a completed run."""

    @model_validator(mode="after")
    def validate_selection(self) -> Self:
        """Require the selected identity only for completed runs."""
        if (self.content_hash is not None) != (self.status == "completed"):
            raise ValueError("Only completed runs require content_hash")
        return self


class RolloutCall(ExecutionStatus):
    """Physical environment invocation stored in call.json."""

    task_id: str
    """Public task identity."""

    content_hash: str
    """Skill identity submitted to the environment."""

    result: dict[str, JsonValue] | None = None
    """Full environment result, including environment-specific fields."""

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        """Validate the shared result contract without discarding extensions."""
        if (self.result is not None) != (self.status == "completed"):
            raise ValueError("Only completed rollout calls require result")
        if self.result is not None:
            RolloutResult.model_validate(
                {key: value for key, value in self.result.items() if key in RolloutResult.model_fields}
            )
        return self


class ModelMessage(ArtifactRecord):
    """One optimizer input message."""

    role: Literal["system", "user"]
    """Role in the optimizer request."""

    content: str
    """Prompt text sent to the optimizer."""


class ModelResponse(ArtifactRecord):
    """Text-only optimizer response using the AgentResult wire format."""

    output: str = Field(min_length=1)
    """Optimizer text returned to generation or merging."""

    trajectory: list[dict[str, JsonValue]]
    """Provider execution trace preserved verbatim."""

    usage: dict[str, JsonValue]
    """Provider usage counters and optional extensions."""


class ModelCall(ExecutionStatus):
    """Physical optimizer invocation stored in call.json."""

    messages: list[ModelMessage] = Field(min_length=2, max_length=2)
    """System and user prompts submitted for this call."""

    response: ModelResponse | None = None
    """Original text response for a completed call."""

    @model_validator(mode="after")
    def validate_response(self) -> Self:
        """Require a response only after completion."""
        if (self.response is not None) != (self.status == "completed"):
            raise ValueError("Only completed model calls require response")
        return self


class ForestNodeRecord(ArtifactRecord):
    """Node metadata in forest.json."""

    id: str
    """Run-local node identity."""

    tree_id: int = Field(ge=0)
    """Owning tree index."""

    parent_id: str | None
    """Parent identity; null for roots."""

    depth: int = Field(ge=0)
    """Distance from the shared initial Skill."""

    gain: float
    """Improvement over the parent."""

    scores: dict[str, float]
    """Observed validation rewards."""

    content_hash: str
    """Candidate Skill identity."""


class ForestRecord(ArtifactRecord):
    """Forest checkpoint in forest.json."""

    expansions: int = Field(ge=0)
    """Number of expanded parents."""

    nodes: list[ForestNodeRecord]
    """All roots and accepted descendants."""


class SharedTrainRecord(ArtifactRecord):
    """Shared evidence membership in shared_train.json."""

    task_ids: list[str]
    """Tasks whose root trajectories were reused."""


class InvalidDecision(ArtifactRecord):
    """Invalid candidate in decision.json."""

    status: Literal["invalid_patch"]
    """Patch parsing or application failed."""

    error: str
    """Failure description."""


class UnchangedDecision(ArtifactRecord):
    """Unchanged candidate in decision.json."""

    status: Literal["no_change"]
    """Patch did not change the Skill hash."""


class ScoredDecision(ArtifactRecord):
    """Validation gate outcome in decision.json."""

    status: Literal["accepted", "rejected"]
    """Candidate gate outcome."""

    scores: dict[str, float]
    """Observed validation rewards, possibly partial."""

    gain: float | None
    """Full validation gain; null when early rejection leaves scores incomplete."""

    evaluated: int = Field(ge=0)
    """Number of completed validation tasks."""

    expected: int = Field(ge=1)
    """Full validation task count."""

    early_rejected: bool
    """Whether validation stopped before completion."""

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        """Keep partial validation counts and acceptance consistent."""
        if self.evaluated != len(self.scores) or self.evaluated > self.expected:
            raise ValueError("Validation counts must match the score vector")
        if self.early_rejected != (self.evaluated < self.expected):
            raise ValueError("early_rejected must describe incomplete validation")
        if (self.gain is None) != self.early_rejected:
            raise ValueError("Only incomplete validation requires a null gain")
        if self.status == "accepted" and self.early_rejected:
            raise ValueError("Accepted candidates require complete validation")
        return self


class ScoresRecord(ArtifactRecord):
    """Full-pool evaluation in scores.json."""

    scores: dict[str, float]
    """Per-task evaluation rewards."""

    mean: float
    """Mean reward across tasks."""


class WinnerRecord(ArtifactRecord):
    """One entry in ensemble/winners.json."""

    id: str
    """Winning node identity."""

    tree_id: int = Field(ge=0)
    """Owning tree index."""

    score: float
    """Full-pool mean reward."""

    novelty: float
    """Distance from the initial Skill."""


class SelectionRecord(ArtifactRecord):
    """Selected winner in selection.json."""

    selected: str
    """Node returned by the select strategy."""


class MergePoolRecord(ArtifactRecord):
    """Eligible synthesis pool in merge_pool.json."""

    threshold: float
    """Minimum full-pool score for inclusion."""

    ids: list[str]
    """Ordered node identities admitted to merging."""


class TokenTotals(ArtifactRecord):
    """Aggregated normalized token counters."""

    input_tokens: int = Field(ge=0)
    """Reported input token sum."""

    output_tokens: int = Field(ge=0)
    """Reported output token sum."""

    total_tokens: int = Field(ge=0)
    """Reported total token sum."""


class CallTotals(ArtifactRecord):
    """Physical-call aggregate in summary.json."""

    calls: int = Field(ge=0)
    """Number of saved physical calls."""

    statuses: dict[Literal["running", "completed", "failed"], int]
    """Call counts by lifecycle state."""

    summed_call_duration_s: float = Field(ge=0)
    """Sum of durations; concurrent calls may overlap."""

    usage_present_calls: int = Field(ge=0)
    """Calls containing nonempty usage records."""

    usage: TokenTotals
    """Normalized reported token sums."""


class RunSummary(ArtifactRecord):
    """Audit aggregate in summary.json."""

    rollout: CallTotals
    """Physical environment calls."""

    model: CallTotals
    """Physical optimizer calls."""

    decisions: dict[Literal["invalid_patch", "no_change", "accepted", "rejected"], int]
    """Candidate counts by gate outcome."""


DecisionRecord = Annotated[InvalidDecision | UnchangedDecision | ScoredDecision, Field(discriminator="status")]
"""Candidate decision variants stored in decision.json."""


class SkillForestTrajectoryContext(BaseModel):
    """Expose completed training evidence without private scoring inputs."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    """Prevent rebinding context fields; adapters must not mutate nested evidence."""

    task: Task
    """Public task, preserving its concrete environment type."""

    result: RolloutResult
    """Completed training result; specialized adapters may use recorded grading feedback."""

    index: int = Field(default=1, ge=1)
    """One-based position in the current reflection batch."""


class SkillForestReflectionContext(BaseModel):
    """Inputs for one outcome-homogeneous reflection request."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    """Prevent reassignment of stage inputs."""

    skill: str
    """Unmodified parent Skill Markdown."""

    trajectories: tuple[SkillForestTrajectoryContext, ...]
    """Ordered completed training evidence, excluding RewardData."""

    success: bool
    """Shared environment success label for this batch."""

    edit_budget: int = Field(ge=1)
    """Maximum number of requested edits."""


class SkillForestPatchMergeContext(BaseModel):
    """Inputs for reducing patches from the same outcome stream."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    """Prevent reassignment of stage inputs."""

    skill: str
    """Unmodified parent Skill Markdown."""

    patches: tuple[Patch, ...]
    """Ordered patches being reduced."""

    success: bool
    """Whether the patches originate from successful trajectories."""

    level: int = Field(ge=1)
    """One-based reduction depth."""


class SkillForestFinalPatchContext(BaseModel):
    """Inputs for reconciling failure and success patch streams."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    """Prevent reassignment of stage inputs."""

    skill: str
    """Unmodified parent Skill Markdown."""

    failure: Patch
    """Reduced failure-derived patch, presented first."""

    success: Patch
    """Reduced success-derived patch, presented second."""


class SkillForestRankingContext(BaseModel):
    """Inputs for ranking one merged candidate patch."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    """Prevent reassignment of stage inputs."""

    skill: str
    """Unmodified parent Skill Markdown."""

    patch: Patch
    """Complete merged edit pool before budget selection."""

    edit_budget: int = Field(ge=1)
    """Maximum number of edits retained in optimizer priority order."""


class SkillForestSkillMergeContext(BaseModel):
    """Inputs for synthesizing two complete Skill candidates."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    """Prevent reassignment of stage inputs."""

    left: str
    """First candidate Markdown."""

    right: str
    """Second candidate Markdown."""

    left_score: float
    """First candidate's score on the shared training pool."""

    right_score: float
    """Second candidate's score on the same pool."""
