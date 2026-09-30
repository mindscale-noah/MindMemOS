"""Transport-independent inputs and results of Skill service operations."""

from typing import Annotated, Self
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, JsonValue, StringConstraints, model_validator

from ..algos.manage.skill_manage import SceneRecord
from ..envs.data.typing import DataSample, SplitManifest
from ..envs.typing import Task
from ..persistence import SkillRecord, SkillVersionRecord
from ..persistence.db_services.skills.typing import VersionNaming
from ..skill import SkillSnapshot

Nonempty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class SkillPipelineRequest(BaseModel):
    """Select one existing Skill for evolution or training.

    Backend choices and evidence preparation policies belong to the concrete
    pipeline configuration. Implementations needing additional per-call inputs
    may define typed request extensions and must validate them before I/O.
    This contract does not prescribe trajectory queries or dataset selection.
    """

    model_config = ConfigDict(extra="forbid")
    """Reject undeclared inputs rather than silently ignoring selection options."""

    pipeline: Nonempty | None = None
    """Configured instance name; omission requires exactly one available pipeline."""

    project_id: Nonempty
    """Caller-selected repository scope; not proof of authorization."""

    skill_ref: Nonempty
    """Skill identity or reference resolved within the selected project."""

    base_version_id: Nonempty | None = None
    """Owned parent version; None captures the current version at run start."""

    auto_accept: bool = True
    """Whether to select the evolved version as the active Skill version."""


class TrainingSplit(BaseModel):
    """Assign task IDs to disjoint training, validation, and test sets."""

    model_config = ConfigDict(extra="forbid")
    """Accept only the three supported split names."""

    train: list[str] = Field(min_length=1)
    """Ordered task IDs used to optimize candidates; at least one is required."""

    val: list[str] = Field(default_factory=list)
    """Ordered task IDs used for candidate selection; omitted means no validation."""

    test: list[str] = Field(default_factory=list)
    """Held-out task IDs evaluated only after final selection; omitted means no test."""

    @model_validator(mode="after")
    def validate_membership(self) -> Self:
        """Reject blank or repeated IDs using the shared split contract.

        Returns:
            The validated, disjoint split selection.

        Raises:
            ValueError: An ID is blank or occurs more than once across the splits.
        """
        SplitManifest(root={"train": self.train, "val": self.val, "test": self.test})
        return self


class TrainingRequest(BaseModel):
    """Supply complete tasks, explicit split membership, and an input Skill version."""

    model_config = ConfigDict(extra="forbid")
    """Reject undeclared inputs, including obsolete Skill references."""

    algo: str
    """Configured instance name; omission requires exactly one available pipeline."""

    environment: Nonempty | None = None
    """Configured training environment profile; omission selects the sole profile."""

    project_id: Nonempty | None = None
    """Repository scope; omission is allowed only through a locally bound service."""

    base_version_id: Nonempty = Field(...)
    """Required input version, used as the published version's parent."""

    auto_accept: bool = True
    """Whether to select the trained version as the active Skill version."""

    work_dir: Nonempty
    """Stable run directory passed to the trainer; matching runs resume automatically."""

    tasks: list[DataSample] = Field(min_length=1)
    """Materialized public tasks paired with private scoring inputs; task IDs must be unique.

    Reuse DataSample so environment-specific task and scoring fields are retained.
    Private reward data must never be injected into the agent's public task.
    """

    split: TrainingSplit
    """Explicit assignment of every supplied task to exactly one supported split."""

    @model_validator(mode="after")
    def validate_tasks(self) -> Self:
        """Require unique tasks and an exact partition of the supplied task IDs.

        Returns:
            The validated training input with its original task order preserved.

        Raises:
            ValueError: Task IDs repeat, a split references missing tasks, or tasks
                are omitted from the split assignment.
        """
        task_ids = [sample.task.id for sample in self.tasks]
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("Training task IDs must be unique")
        selected = set(self.split.train + self.split.val + self.split.test)
        available = set(task_ids)
        if missing := selected - available:
            raise ValueError(f"Split references unknown task IDs: {sorted(missing)}")
        if unassigned := available - selected:
            raise ValueError(f"Tasks must be assigned to a split: {sorted(unassigned)}")
        return self


class SkillPipelineResult(BaseModel):
    """Identify a completed run whose output Skill version was persisted.

    Return only after successful persistence. Candidates, pending jobs, and
    failed or conflicting commits are not successful results of this contract.
    Algorithm-specific metrics may be added through typed result extensions.
    """

    model_config = ConfigDict(extra="forbid")
    """Reject undeclared result fields and incomplete identity contracts."""

    project_id: Nonempty
    """Repository scope containing the resulting Skill."""

    skill_id: Nonempty
    """Resolved identity of the evolved or trained Skill."""

    base_version_id: Nonempty
    """Version captured before execution and used as the committed parent."""

    version_id: Nonempty
    """Persisted output version, whether or not it became active."""

    run_id: Nonempty
    """Pipeline execution identity linking inputs, artifacts, and saved output."""


class ProjectRequest(BaseModel):
    """Select the local repository project explicitly."""

    model_config = ConfigDict(extra="forbid")
    """Reject unknown request fields."""

    project_id: Nonempty | None = None
    """Explicit scope; omission is allowed only by a service assembled for local use."""


class SkillRequest(ProjectRequest):
    """Resolve one Skill by identity or alias within a project."""

    skill_ref: Nonempty
    """Exact alias, full Skill ID, or unambiguous Skill ID prefix."""


class RegisterRequest(ProjectRequest, VersionNaming):
    """Register a new central identity from a directory or portable snapshot."""

    source: SkillSnapshot | Nonempty
    """Complete snapshot or local directory/SKILL.md path, read only for this request."""

    name: Nonempty
    """Initial display name; duplicate display names are allowed."""

    alias: Nonempty | None = None
    """Optional project-unique shortcut."""

    message: str | None = None
    """Initial registration and first-version explanation."""

    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    """Mutable user metadata stored on the Skill family."""

    description: str | None = None
    """Server-parsed description from the Skill document when available."""

    file_meta: dict[str, JsonValue] | None = None
    """Server-parsed version file metadata when available."""

    require_frontmatter: bool = False
    """Validate SKILL.md and derive description and metadata from the source."""

    registration_source: str | None = None
    """Optional provenance reference; defaults to the import path when provided."""

    @model_validator(mode="after")
    def validate_initial_version(self):
        """Reject increments without a parent version."""
        if self.version_bump != "patch":
            raise ValueError("Initial registration has no version to increment")
        return self


class ListRequest(ProjectRequest):
    """Read a bounded page of central Skills."""

    limit: int = Field(default=100, ge=0, le=1000)
    """Maximum number of returned records."""

    offset: int = Field(default=0, ge=0)
    """Number of sorted records to skip."""

    snapshot_at: AwareDatetime | None = None
    """First-page cutoff for a stable cursor walk; absent uses legacy offset paging."""

    after_created_at: AwareDatetime | None = None
    """Creation time of the last returned Skill in cursor order."""

    after_skill_id: Nonempty | None = None
    """Stable identity breaking ties at the cursor creation time."""

    @model_validator(mode="after")
    def validate_cursor(self) -> Self:
        """Require a complete cursor without combining it with offset paging."""
        if (self.after_created_at is None) != (self.after_skill_id is None):
            raise ValueError("Cursor position requires both creation time and Skill ID")
        if self.after_skill_id is not None and self.snapshot_at is None:
            raise ValueError("Cursor position requires a snapshot cutoff")
        if self.snapshot_at is not None and self.offset != 0:
            raise ValueError("Cursor paging cannot use offset")
        return self


class ReportTrajectoryRequest(ProjectRequest):
    """Report one Agent execution using an exact registered Skill version."""

    user_id: Nonempty
    """Business user identity from the Harness context."""

    agent_id: Nonempty
    """Agent identity from the Harness context."""

    messages: list[dict[str, JsonValue]] = Field(min_length=1)
    """Ordered interaction messages for this physical execution."""

    skill_version_id: Nonempty
    """Exact Skill version loaded by the Agent."""

    session_id: Nonempty | None = None
    """Optional logical Agent session identity."""

    status: Nonempty = "completed"
    """Caller-declared execution outcome."""

    started_at: AwareDatetime | None = None
    """Actual Agent start time when known."""

    finished_at: AwareDatetime | None = None
    """Actual Agent finish time when known."""

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        """Require message roles and forward-moving execution times."""
        if any(not message.get("role") or "content" not in message for message in self.messages):
            raise ValueError("Each message requires role and content")
        if self.started_at and self.finished_at and self.finished_at < self.started_at:
            raise ValueError("finished_at must not precede started_at")
        return self


class ShowTrajectoryRequest(ProjectRequest):
    """Read one trajectory within its owning project."""

    trajectory_id: Nonempty
    """Exact server-generated or local trajectory identity."""


class ListTrajectoriesRequest(ProjectRequest):
    """Read one stable page of reported trajectories for an exact family."""

    skill_id: Nonempty
    """Exact Skill family identity."""

    skill_version_id: Nonempty | None = None
    """Optional exact loaded version filter."""

    status: Nonempty | None = None
    """Optional reported execution status filter."""

    received_at_from: AwareDatetime | None = None
    """Inclusive lower bound for server receipt time."""

    received_at_to: AwareDatetime | None = None
    """Inclusive upper bound for server receipt time."""

    snapshot_at: AwareDatetime
    """First-page receipt cutoff for stable scrolling."""

    after_received_at: AwareDatetime | None = None
    """Last receipt time from the preceding page."""

    after_trajectory_id: Nonempty | None = None
    """Last identity breaking receipt-time ties."""

    limit: int = Field(default=20, ge=1, le=101)
    """Maximum number of records including one lookahead row."""

    @model_validator(mode="after")
    def validate_cursor(self) -> Self:
        """Require both parts of a receipt cursor and ordered bounds."""
        if (self.after_received_at is None) != (self.after_trajectory_id is None):
            raise ValueError("Trajectory cursor requires receipt time and ID")
        if (
            self.received_at_from is not None
            and self.received_at_to is not None
            and self.received_at_to < self.received_at_from
        ):
            raise ValueError("Receipt time bounds are reversed")
        return self


class ShowRequest(SkillRequest):
    """Inspect the selected or explicitly requested version."""

    version_id: Nonempty | None = None
    """Owned local version ID; omitted selects the current version."""


class VersionResolveRequest(ProjectRequest):
    """Find one project-owned version matching a source bundle hash."""

    content_hash: Nonempty
    """Canonical source bundle digest computed by the client."""

    skill_id: Nonempty | None = None
    """Optional exact Skill family identity."""

    skill_version_id: Nonempty | None = None
    """Optional exact version identity that must belong to skill_id."""


class VersionResolution(BaseModel):
    """Exact immutable source version selected by a hash lookup."""

    skill_id: str
    """Owning Skill family identity."""

    skill_version_id: str
    """Matching exact version identity."""

    content_hash: str
    """Stored canonical source bundle hash."""

    version_label: str | None
    """Immutable display label of the matched version."""

    is_current: bool
    """Whether this version is the family current version."""


class ExportRequest(ShowRequest):
    """Export stored content without binding or tracking the destination."""

    directory: Nonempty
    """Application directory receiving the complete stored bundle and resources."""

    overwrite: bool = False
    """Permit replacing a nonempty destination, including removal of stale files."""


class CommitRequest(SkillRequest, VersionNaming):
    """Create a child snapshot and atomically select it."""

    source: SkillSnapshot | Nonempty
    """Complete snapshot or local directory/SKILL.md path to read once."""

    base_version_id: Nonempty | None = None
    """Owned parent version; omitted captures the current version before reading source."""

    message: str | None = None
    """Explanation stored with the new version."""

    active: bool = True
    """Whether the new version becomes the family current version."""

    inherit_resources: bool = False
    """Whether to retain parent resources when only a source bundle is supplied."""

    description: str | None = None
    """Server-parsed description from the new Skill document when available."""

    file_meta: dict[str, JsonValue] | None = None
    """Server-parsed version file metadata when available."""

    require_frontmatter: bool = False
    """Validate SKILL.md and derive description and metadata from the source."""

    @model_validator(mode="after")
    def validate_inactive_label(self):
        """Require a lookup label for an inactive publication."""
        if not self.active and self.version_label is None:
            raise ValueError("Inactive commits require an explicit version label")
        return self


class HistoryRequest(SkillRequest):
    """Read a bounded page of immutable version snapshots."""

    limit: int = Field(default=100, ge=0, le=1000)
    """Maximum number of versions returned."""

    offset: int = Field(default=0, ge=0)
    """Number of chronologically sorted versions to skip."""


class DiffRequest(SkillRequest):
    """Compare two project-owned versions without changing current selections."""

    to_version_id: Nonempty
    """Target version to compare."""

    from_version_id: Nonempty | None = None
    """Starting version; omitted selects the current version."""

    to_skill_ref: Nonempty | None = None
    """Optional target family; omission uses the starting family."""


class SwitchRequest(SkillRequest):
    """Select an existing version without changing external files."""

    version_id: Nonempty
    """Owned version to make current."""


class SetMetaRequest(SkillRequest):
    """Change display metadata without creating a version."""

    name: Nonempty | None = None
    """Replacement display name; omitted leaves the name unchanged."""

    alias: Nonempty | None = None
    """Replacement alias; omitted leaves the alias unchanged."""

    clear_alias: bool = False
    """Remove the current alias explicitly."""

    metadata: dict[str, JsonValue] | None = None
    """Replacement family metadata; None leaves the value unchanged."""

    @model_validator(mode="after")
    def validate_changes(self):
        """Reject contradictory and empty metadata edits."""
        if self.alias is not None and self.clear_alias:
            raise ValueError("alias and clear_alias cannot be combined")
        if self.name is None and self.alias is None and not self.clear_alias and self.metadata is None:
            raise ValueError("Supply name, alias, metadata, or clear_alias")
        return self


class SkillDetail(BaseModel):
    """A central identity and the inspected version, which may be historical."""

    skill: SkillRecord
    """Current central metadata and selected-version pointer."""

    version: SkillVersionRecord
    """Complete requested version snapshot."""


class OrganizeRequest(ProjectRequest):
    """Select exact project-owned Skill versions for one graph publication."""

    graph_id: Nonempty
    """Logical graph identity within the selected project."""

    actor_id: Nonempty
    """Trusted caller identity recorded on the graph revision."""

    message: Nonempty | None = None
    """Optional explanation recorded on the successful graph revision."""

    skill_version_ids: list[Nonempty]
    """Complete version scope; an empty list publishes an empty graph."""

    scene_records: dict[str, SceneRecord] = Field(default_factory=dict)
    """Explicit scenes keyed by Skill family ID; omitted means no scenes."""


class OrganizeResult(BaseModel):
    """Identify the immutable graph published for an explicit version scope."""

    project_id: Nonempty
    """Project that owns the graph."""

    graph_id: Nonempty
    """Logical graph identity within the project."""

    graph_version_id: UUID
    """Immutable graph snapshot available for later recommendation."""

    skill_version_ids: list[Nonempty]
    """Exact input versions in caller order."""


class RecommendRequest(ProjectRequest):
    """Select one immutable organized graph for a public task."""

    graph_id: Nonempty
    """Logical graph identity within the selected project."""

    graph_version_id: UUID | None = None
    """Pinned graph snapshot; omission selects and pins the current graph head."""

    task: Task
    """Public task used for retrieval; no private reward data is accepted."""

    limit: int = Field(default=5, ge=1, strict=True)
    """Maximum number of recommended Skill versions."""


class RecommendResult(BaseModel):
    """Ordered Skill versions selected from one pinned graph snapshot."""

    project_id: Nonempty
    """Project that owns the graph and every selected version."""

    graph_id: Nonempty
    """Logical graph identity within the project."""

    graph_version_id: UUID
    """Immutable graph snapshot used for retrieval."""

    skill_version_ids: list[Nonempty]
    """Selected exact version IDs in descending recommendation order."""


class SkillPageRequest(ProjectRequest):
    """Filter one stable page of Skill families across repository evidence."""

    name_pattern: Nonempty | None = None
    """Optional case-sensitive wildcard pattern for family names."""

    alias_pattern: Nonempty | None = None
    """Optional case-sensitive wildcard pattern for family aliases."""

    user_id: Nonempty | None = None
    """Optional user identity required on a reported trajectory."""

    agent_id: Nonempty | None = None
    """Optional Agent identity required on that same reported trajectory."""

    limit: int = Field(default=20, ge=1, le=100)
    """Maximum number of matching families in this page."""

    snapshot_at: AwareDatetime
    """First-page cutoff reused by subsequent pages."""

    after_created_at: AwareDatetime | None = None
    """Creation time of the previous page's last family."""

    after_skill_id: Nonempty | None = None
    """Stable identity breaking ties at the previous page boundary."""

    @model_validator(mode="after")
    def validate_position(self) -> "SkillPageRequest":
        """Require both fields of a continuation position."""
        if (self.after_created_at is None) != (self.after_skill_id is None):
            raise ValueError("Skill page position requires creation time and Skill ID")
        return self


class SkillPage(BaseModel):
    """Return matched families and whether another page exists."""

    model_config = ConfigDict(extra="forbid")
    """Reject undeclared page output fields."""

    items: list[SkillRecord]
    """Families matching all requested filters in stable order."""

    has_more: bool
    """Whether at least one further matching family exists."""
