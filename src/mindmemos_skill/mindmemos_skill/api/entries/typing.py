"""Define HTTP request models, pagination cursors, and application callback types."""

from collections.abc import Awaitable, Callable
from typing import Annotated, Literal, Self

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    model_validator,
)

from ...config.config import FastAPIConfig
from ...envs.config import EnvironmentConfig
from ..auth import Nonempty, SkillHttpIdentity
from ..base import MindMemOSSkill
from ..typing import SkillHistoryCursor, StructuredSkillInfo, TrainingSplit


class HttpRequest(BaseModel):
    """Reject undeclared public HTTP fields."""

    model_config = ConfigDict(extra="forbid")
    """Do not silently accept server-owned or misspelled fields."""


class RegisterHttpRequest(HttpRequest):
    """Register portable source content without server-side file access."""

    skill: StructuredSkillInfo
    """Complete source text, Base64 resources, and commit metadata."""

    alias: Nonempty | None = None
    """Optional project-unique family shortcut."""


class SkillFilters(HttpRequest):
    """Optional Skill family fields and trajectory actor filters."""

    name: Nonempty | None = None
    """Case-sensitive whole-name wildcard pattern."""

    alias: Nonempty | None = None
    """Case-sensitive whole-alias wildcard pattern."""

    agent_id: Nonempty | None = None
    """Exact Agent identity on the same matching trajectory as user_id."""

    user_id: Nonempty | None = None
    """Exact user identity on the same matching trajectory as agent_id."""


class ListHttpRequest(HttpRequest):
    """Filter and continue a stable Skill family cursor within one project."""

    filters: SkillFilters = Field(default_factory=SkillFilters)
    """Optional conjunctive family and trajectory filters."""

    limit: int | None = Field(default=None, ge=1)
    """Maximum number of Skill families returned in this page."""

    cursor: str | None = Field(default=None, min_length=1, max_length=2048)
    """Opaque continuation token returned by the preceding list response."""


class ListCursor(BaseModel):
    """Server-owned list position bound to one project and filter set."""

    model_config = ConfigDict(extra="forbid")
    """Reject unknown cursor fields and incompatible future encodings."""

    version: Literal[1] = 1
    """Encoding contract version for safe future cursor changes."""

    project_id: Nonempty
    """Project authorized for the original list request."""

    filters: SkillFilters
    """Filters applied before cursor pagination."""

    snapshot_at: AwareDatetime
    """Inclusive creation cutoff captured at the first page."""

    last_created_at: AwareDatetime
    """Creation time of the last returned family."""

    last_skill_id: Nonempty
    """Identity breaking creation-time ties at the cursor position."""


class ShowHttpRequest(HttpRequest):
    """Select one family and optionally one historical version."""

    skill_id: Nonempty | None = None
    """Exact Skill family identity."""

    skill_version_id: Nonempty | None = None
    """Exact owned version; omission selects the family active version."""

    @model_validator(mode="after")
    def validate_selection(self):
        """Require a family or an exact immutable version."""
        if self.skill_id is None and self.skill_version_id is None:
            raise ValueError("Supply skill_id or skill_version_id")
        return self


class CommitHttpRequest(HttpRequest):
    """Publish complete structured source content."""

    skill_id: Nonempty
    """Target family identity."""

    skill: StructuredSkillInfo
    """Complete replacement content and commit metadata."""

    base_version_id: Nonempty | None = None
    """Explicit parent version; omission selects the active version."""

    active: bool = True
    """Whether the new version becomes active."""

    inherit_resources: bool = False
    """Explicitly inherit parent resources instead of replacing them."""


class HistoryHttpRequest(HttpRequest):
    """Page the immutable version history of one family."""

    skill_id: Nonempty
    """Exact Skill family identity."""

    limit: int | None = Field(default=None, ge=1)
    """Maximum number of versions in this page."""

    offset: int = Field(default=0, ge=0)
    """Number of prior versions to skip."""

    cursor: str | None = None
    """Opaque continuation token; cannot be combined with a nonzero offset."""


class HistoryCursor(SkillHistoryCursor):
    """History position bound to an authorized project and family."""

    project_id: Nonempty
    """Project that created this continuation token."""

    skill_id: Nonempty
    """Family whose versions are being paged."""


class DiffHttpRequest(HttpRequest):
    """Select two owned versions in the same authorized project."""

    skill_id: Nonempty
    """Source Skill family identity."""

    from_skill_version_id: Nonempty | None = None
    """Source version; omission selects the source family active version."""

    to_skill_id: Nonempty | None = None
    """Target family; omission uses the source family."""

    to_skill_version_id: Nonempty
    """Exact target version belonging to the target family."""


class ActivateHttpRequest(HttpRequest):
    """Select a previously saved version as active."""

    skill_id: Nonempty | None = None
    """Exact Skill family identity."""

    skill_version_id: Nonempty
    """Exact version belonging to the family."""

    expected_active_version_id: Nonempty | None = None
    """Optional previous pointer used to reject stale activation requests."""


class UpdateInfoHttpRequest(HttpRequest):
    """Change family display fields without creating a source version."""

    skill_id: Nonempty
    """Exact Skill family identity."""

    name: Nonempty | None = None
    """Replacement name; omitted leaves the current name unchanged."""

    alias: Nonempty | None = None
    """Replacement alias; explicit null clears the alias."""

    metadata: dict[str, JsonValue] | None = None
    """Replacement family metadata; empty object clears it."""

    @model_validator(mode="after")
    def validate_changes(self) -> Self:
        """Distinguish omitted values from explicit null and reject empty edits."""
        if "name" in self.model_fields_set and self.name is None:
            raise ValueError("name cannot be null")
        if "metadata" in self.model_fields_set and self.metadata is None:
            raise ValueError("metadata cannot be null")
        if not {"name", "alias", "metadata"} & self.model_fields_set:
            raise ValueError("At least one family field must be supplied")
        return self


class DeleteHttpRequest(HttpRequest):
    """Delete one unreferenced Skill family."""

    skill_id: Nonempty
    """Exact Skill family identity."""


class VersionResolveHttpRequest(HttpRequest):
    """Resolve a source hash only when the matching version is unambiguous."""

    content_hash: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")] | None = None
    """Optional exact source text digest."""

    skill_hash: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")] | None = None
    """Optional complete source digest."""

    skill_id: Nonempty | None = None
    """Optional exact family constraint."""

    skill_version_id: Nonempty | None = None
    """Optional exact immutable version identity."""

    @model_validator(mode="after")
    def validate_lookup(self) -> Self:
        """Require at least one hash or immutable version identity."""
        if self.content_hash is None and self.skill_hash is None and self.skill_version_id is None:
            raise ValueError("Supply content_hash, skill_hash, or skill_version_id")
        return self


class SkillResolveHttpRequest(HttpRequest):
    """Select a current or immutable Skill version from one reference."""

    skill_ref: Nonempty
    """Skill alias, family ID or unique prefix, or version ID or unique prefix."""


class TrajectoryReportHttpRequest(HttpRequest):
    """Report one actual Skill version used in an Agent execution."""

    user_id: Nonempty
    """Business user identity supplied by the Harness context."""

    agent_id: Nonempty
    """Agent identity that performed the execution."""

    messages: list[dict[str, JsonValue]] = Field(min_length=1)
    """Ordered interaction messages with role and content per item."""

    skill_version_id: Nonempty
    """Exact loaded version resolved by the client before reporting."""

    session_id: Nonempty | None = None
    """Optional Harness session identity."""

    started_at: AwareDatetime | None = None
    """Actual execution start time when known."""

    finished_at: AwareDatetime | None = None
    """Actual execution finish time when known."""

    status: Nonempty = "completed"
    """Caller-declared execution status."""

    @model_validator(mode="after")
    def validate_messages(self) -> Self:
        """Require the minimum message shape and ordered timestamps."""
        if any(not message.get("role") or "content" not in message for message in self.messages):
            raise ValueError("Each message requires role and content")
        if self.started_at and self.finished_at and self.finished_at < self.started_at:
            raise ValueError("finished_at must not precede started_at")
        return self


class TrajectoryShowHttpRequest(HttpRequest):
    """Read one stored trajectory report."""

    trajectory_id: Nonempty
    """Server-generated trajectory identity."""


class TrajectoryListHttpRequest(HttpRequest):
    """Page reported trajectories bound to one Skill family."""

    skill_id: Nonempty
    """Exact family identity."""

    skill_version_id: Nonempty | None = None
    """Optional exact version filter."""

    status: Nonempty | None = None
    """Optional caller-declared execution status filter."""

    received_at_from: AwareDatetime | None = None
    """Inclusive server receipt time lower bound."""

    received_at_to: AwareDatetime | None = None
    """Inclusive server receipt time upper bound."""

    limit: int | None = Field(default=None, ge=1)
    """Maximum number of trajectory summaries returned."""

    cursor: Nonempty | None = None
    """Opaque server pagination cursor."""


class TrajectoryStatisticsHttpRequest(HttpRequest):
    """Read fixed-size aggregates for an exact owned Skill version."""

    skill_id: Nonempty
    """Owning Skill family or alias."""

    skill_version_id: Nonempty
    """Exact version to count; date windows are server-owned."""


class TrajectoryListCursor(BaseModel):
    """Server-owned continuation position for one trajectory filter set."""

    model_config = ConfigDict(extra="forbid")
    """Reject unknown fields and unsupported cursor encodings."""

    version: Literal[1] = 1
    """Cursor encoding contract version."""

    project_id: Nonempty
    """Project authorized for the original page."""

    skill_id: Nonempty
    """Exact family requested on the original page."""

    skill_version_id: Nonempty | None = None
    """Exact version filter from the original page."""

    status: Nonempty | None = None
    """Status filter from the original page."""

    received_at_from: AwareDatetime | None = None
    """Lower receipt bound from the original page."""

    received_at_to: AwareDatetime | None = None
    """Upper receipt bound from the original page."""

    snapshot_at: AwareDatetime
    """First-page receipt cutoff."""

    last_received_at: AwareDatetime
    """Receipt time of the last returned attempt."""

    last_trajectory_id: Nonempty
    """Identity breaking ties at the last receipt time."""


class OrganizeHttpRequest(HttpRequest):
    """Organize explicit Skill references into a new or existing graph."""

    skill_refs: list[Nonempty]
    """Exact versions or family references resolved by the management pipeline."""

    graph_id: Nonempty | None = None
    """Existing graph identity; omission creates a new graph."""

    alias: Nonempty | None = None
    """Optional alias for a new graph."""

    commit_message: str | None = None
    """Explanation stored with the publication."""

    force_extract_scene: bool = Field(default=False, strict=True)
    """Refresh version Scene data even when the saved cache is valid."""


class GraphResolveHttpRequest(HttpRequest):
    """Resolve a project-scoped graph ID, alias, or unique ID prefix."""

    graph_ref: Nonempty
    """Graph reference selected by the caller."""


class GraphSnapshotsHttpRequest(HttpRequest):
    """Select one graph within the authenticated project."""

    graph_id: Nonempty
    """Exact graph identity."""


class GraphSnapshotHttpRequest(GraphSnapshotsHttpRequest):
    """Select one immutable snapshot of a project-owned graph."""

    graph_version_id: Nonempty
    """Exact immutable graph snapshot identity."""


class RecommendHttpRequest(GraphResolveHttpRequest):
    """Recommend Skill versions from one immutable published graph."""

    graph_version_id: Nonempty | None = None
    """Exact snapshot; omission pins the graph's current published revision."""

    query: Nonempty
    """Task instruction used for graph retrieval."""

    top_k: int | None = Field(default=None, ge=1)
    """Maximum recommended Skill versions, subject to project policy."""

    strategy: Literal["graph", "vector"] = "graph"
    """Graph propagation or direct vector similarity."""

    include_scenes: bool = Field(default=False, strict=True)
    """Vector-only option matching Scene nodes as well as Skill nodes."""

    @model_validator(mode="after")
    def validate_strategy(self):
        """Reject vector-only options in graph mode and blank instructions."""
        if self.strategy == "graph" and self.include_scenes:
            raise ValueError("include_scenes requires the vector strategy")
        if not self.query.strip():
            raise ValueError("Query must not be blank")
        return self


class GraphMembersHttpRequest(GraphResolveHttpRequest):
    """Select a nonempty batch of Skill references in an existing graph."""

    skill_refs: list[str] = Field(min_length=1)
    """Family or version references; duplicate families are invalid."""

    commit_message: str | None = None
    """Explanation stored with the new snapshot."""

    @model_validator(mode="after")
    def validate_members(self):
        """Reject blank or repeated references before making a business call."""
        if any(not reference.strip() for reference in self.skill_refs):
            raise ValueError("Skill references must not be blank")
        if len(self.skill_refs) != len(set(self.skill_refs)):
            raise ValueError("Skill references must be unique")
        return self


class AddGraphMembersHttpRequest(GraphMembersHttpRequest):
    """Add new families without replacing existing graph members."""

    force_extract_scene: bool = Field(default=False, strict=True)
    """Refresh Scene data for the newly selected versions."""


class RemoveGraphMembersHttpRequest(GraphMembersHttpRequest):
    """Remove graph members without deleting canonical Skill versions."""


class EvolveHttpRequest(HttpRequest):
    """Queue one exact version or a collection through the common evolution endpoint."""

    skill_version_id: Nonempty | None = None
    """Exact baseline version; exclusive with skill_version_ids."""

    auto_accept: bool = True
    """Whether the evolved version becomes the active Skill version."""

    pipeline: Nonempty | None = None
    """Permitted evolution pipeline; omission uses the configured HTTP default."""

    skill_version_ids: list[Nonempty] | None = Field(default=None, min_length=1)
    """Exact input version collection; exclusive with skill_version_id."""

    editable_skills: list[Nonempty] | None = None
    """Editable input Skill IDs; None permits all input members."""

    allow_create_skills: bool = True
    """Whether collection evolution may create independent Skills."""

    @model_validator(mode="after")
    def validate_selection(self) -> Self:
        """Reuse the SDK wire contract after normalizing HTTP identity strings."""
        from mindmemos_sdk.skills import EvolveRequest

        EvolveRequest.model_validate(self.model_dump())
        return self


class TaskShowHttpRequest(HttpRequest):
    """Read one project-scoped asynchronous business task."""

    task_id: Nonempty
    """Server-generated task identity."""


class TrainingHttpRequest(HttpRequest):
    """Accept serialized samples before the selected environment restores their types."""

    algo: Nonempty
    """Configured training pipeline name."""

    env: EnvironmentConfig
    """Environment implementation and configuration supplied for this run."""

    project_id: Nonempty | None = None
    """Ignored caller hint; authenticated project scope replaces it."""

    init_skill_version_id: Nonempty
    """Exact parent version used by the training run."""

    auto_accept: bool = True
    """Whether the trained version becomes the active Skill version."""

    work_dir: Nonempty
    """Stable output directory for this training run."""

    tasks: list[dict[str, JsonValue]] = Field(min_length=1)
    """Public tasks and private scoring data parsed using the selected environment."""

    split: TrainingSplit
    """Explicit train, validation, and test task membership."""


IdentityResolver = Callable[[str], SkillHttpIdentity | Awaitable[SkillHttpIdentity]]
"""Resolve a bearer credential into an HTTP identity, synchronously or asynchronously."""

ServiceFactory = Callable[[], Awaitable[tuple[MindMemOSSkill, FastAPIConfig]]]
"""Build a service and its HTTP policy asynchronously for application startup."""
