"""Run-local forest data and validated atomic patch contracts."""

from collections.abc import Callable, Coroutine, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal, Protocol, Self, TypedDict

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from ....envs.base import Environment
from ....envs.data.typing import DataSample
from ....envs.typing import RolloutResult, Task
from ....skill import Skill
from ....skill.typing import SkillSnapshot


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

    reference_payload: dict[str, JsonValue] | None = Field(default=None, exclude=True)
    """Original optimizer JSON, including field order and metadata, used only between generation stages."""

    @classmethod
    def from_reference(
        cls,
        payload: dict[str, JsonValue],
        *,
        source: str | None = None,
        support: int = 1,
        edit_budget: int | None = None,
        merge_level: int | None = None,
    ) -> Self:
        """Validate executable edits while preserving original JSON order and uninterpreted metadata."""
        raw = deepcopy(payload)
        edits = raw.get("edits")
        if not isinstance(edits, list):
            raise ValueError("Reference patch requires an edits list")
        if edit_budget is not None:
            edits = edits[:edit_budget]
            raw["edits"] = edits
        validated = []
        for edit in edits:
            if not isinstance(edit, dict):
                raise ValueError("Reference patch edits must be objects")
            if source is not None:
                edit.setdefault("source_type", source)
                edit.setdefault("support_count", support)
            if merge_level is not None:
                edit["merge_level"] = merge_level
            validated.append(
                Edit.model_validate({key: value for key, value in edit.items() if key in Edit.model_fields})
            )
        return cls(reasoning=raw.get("reasoning", ""), edits=validated, reference_payload=raw)

    def optimizer_input(self) -> dict[str, JsonValue]:
        """Return the preserved reference payload or the strict typed payload."""
        return self.reference_payload if self.reference_payload is not None else self.model_dump(exclude_unset=True)


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


class NodeCheckpoint(BaseModel):
    """Persist a patch-applied candidate and its optional completed validation."""

    model_config = ConfigDict(extra="forbid")
    """Reject unknown checkpoint fields."""

    id: str
    """Stable candidate identity within one forest run."""

    tree_id: int = Field(ge=0)
    """Tree that owns this candidate."""

    parent_id: str | None = None
    """Parent node identity, or None for an initial root."""

    depth: int = Field(default=0, ge=0)
    """Number of patches between this node and its root."""

    skill: SkillSnapshot
    """Complete Skill snapshot saved immediately after patch application."""

    status: Literal["generated", "root", "accepted", "rejected"]
    """Current node validation state."""

    scores: dict[str, float] = Field(default_factory=dict)
    """Completed validation scores; root and accepted nodes have full vectors."""

    gain: float = 0.0
    """Improvement over the parent after completed validation."""

    expanded: bool = False
    """Whether this node's child generation has fully completed."""

    @model_validator(mode="after")
    def validate_state(self) -> Self:
        """Keep root, generated and accepted node states internally consistent."""
        if self.status == "root":
            if self.parent_id is not None or self.depth != 0 or not self.scores:
                raise ValueError("A root requires a full score vector and no parent")
        elif self.parent_id is None or self.depth == 0:
            raise ValueError("A candidate requires a parent and positive depth")
        if self.status == "generated" and (self.scores or self.gain or self.expanded):
            raise ValueError("An unvalidated candidate cannot have scores or expanded children")
        if self.status == "accepted" and (not self.scores or self.gain <= 0):
            raise ValueError("An accepted candidate requires positive gain and scores")
        if self.status == "rejected" and self.expanded:
            raise ValueError("A rejected candidate cannot be expanded")
        return self


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


EvidenceBatch = Sequence[tuple[DataSample, RolloutResult]]
"""Ordered task and result pairs sharing the same success outcome."""

StopPredicate = Callable[[dict[str, RolloutResult]], bool]
"""Return True when remaining queued evaluations should be skipped."""


class RolloutExecutor(Protocol):
    """Keyword-only asynchronous execution accepted by the scheduler."""

    def __call__(
        self, *, env: Environment, sample: DataSample, skill: Skill, directory: Path
    ) -> Coroutine[Any, Any, RolloutResult]:
        """Execute one physical rollout.

        Args:
            env: Caller-owned execution environment.
            sample: Public task and private scoring inputs.
            skill: Candidate to evaluate.
            directory: Exclusive workspace for this call.

        Returns:
            Awaitable completed rollout.
        """
        ...


class RolloutLoader(Protocol):
    """Optional synchronous loader for an already completed rollout."""

    def __call__(self, *, env: Environment, sample: DataSample, skill: Skill, directory: Path) -> RolloutResult | None:
        """Load matching evidence if available.

        Args:
            env: Caller-owned execution environment.
            sample: Public task and private scoring inputs.
            skill: Candidate to evaluate.
            directory: Workspace that may contain a completed call.

        Returns:
            Completed rollout, or None when execution is required.
        """
        ...


class EnsembleCandidate(BaseModel):
    """A frozen winner or synthesized Skill with its evaluation owner."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)
    """Retain Skill objects while preventing candidate reassignment."""

    name: str
    """Stable label used for selection workspaces."""

    skill: Skill
    """Complete candidate evaluated over the training pool."""

    score: float
    """Mean success over the entire training pool."""

    owner: Node
    """Original tree winner that owns the evaluation workspace."""


class NodeSummary(TypedDict):
    """JSON metadata for one accepted forest node."""

    id: str
    """Run-local node identifier."""

    tree_id: int
    """Independent tree index."""

    parent_id: str | None
    """Parent identifier, or None for a root."""

    depth: int
    """Number of edits from the root."""

    gain: float
    """Validation improvement over the parent."""

    scores: dict[str, float]
    """Completed task-level validation scores."""

    content_hash: str
    """Digest of the candidate document."""


class WinnerScore(TypedDict):
    """Comparable selection metrics for a frozen tree winner."""

    id: str
    """Winner node identifier."""

    tree_id: int
    """Tree that produced the winner."""

    score: float
    """Mean success on the complete training pool."""

    novelty: float
    """TF-IDF distance from the initial Skill."""
