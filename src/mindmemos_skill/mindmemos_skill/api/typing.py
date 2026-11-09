"""Transport-independent inputs and results of Skill service operations."""

from typing import Annotated, Literal, Self

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    JsonValue,
    SecretStr,
    StringConstraints,
    TypeAdapter,
    ValidationInfo,
    field_validator,
    model_validator,
)

from mindmemos_skill.persistence.db_services.skills.typing import UserVersionLabel
from mindmemos_skill.skill import SkillContent, decode_resources
from mindmemos_skill.skill.typing import SkillSnapshot, parse_skill_header

from ..envs.config import EnvironmentConfig
from ..envs.data.typing import DataSample, SplitManifest
from ..persistence import EvaluationRecord, SkillRecord, SkillVersionRecord
from ..persistence.db_services.graphs.typing import GraphNodeType

Nonempty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class SkillListCursor(BaseModel):
    """Stable Skill family list position within one creation snapshot."""

    snapshot_at: AwareDatetime | None = None
    """First-page creation cutoff, or None for offset paging."""

    after_created_at: AwareDatetime | None = None
    """Creation time of the last family returned by the preceding page."""

    after_skill_id: Nonempty | None = None
    """Skill identity breaking ties at the preceding page boundary."""

    offset: int = Field(default=0, ge=0)
    """Number of sorted records to skip during offset paging."""

    @model_validator(mode="after")
    def validate_position(self) -> Self:
        """Reject incomplete positions and mixed pagination modes."""
        if (self.after_created_at is None) != (self.after_skill_id is None):
            raise ValueError("Skill cursor requires creation time and Skill ID")
        if self.after_created_at is not None and self.snapshot_at is None:
            raise ValueError("Continuation requires snapshot_at")
        if self.snapshot_at is not None and self.offset:
            raise ValueError("Snapshot pagination cannot include an offset")
        return self


class SkillHistoryCursor(BaseModel):
    """Stable chronological position in one family's immutable history."""

    offset: int = Field(default=0, ge=0)
    """Number of versions to skip in offset mode."""

    snapshot_at: AwareDatetime | None = None
    """Inclusive first-page creation cutoff for stable pagination."""

    after_created_at: AwareDatetime | None = None
    """Creation time of the last returned version."""

    after_version_id: Nonempty | None = None
    """Version identity breaking ties at the continuation boundary."""

    @model_validator(mode="after")
    def validate_position(self) -> Self:
        """Reject incomplete continuation positions and mixed modes."""
        if (self.after_created_at is None) != (self.after_version_id is None):
            raise ValueError("History cursor requires creation time and version ID")
        if self.after_created_at is not None and self.snapshot_at is None:
            raise ValueError("Continuation requires snapshot_at")
        if self.snapshot_at is not None and self.offset:
            raise ValueError("Snapshot pagination cannot include an offset")
        return self


class TrajectoryListCursor(BaseModel):
    """Stable receipt-order position for reported trajectory pages."""

    snapshot_at: AwareDatetime | None = None
    """First-page receipt cutoff."""

    after_received_at: AwareDatetime | None = None
    """Receipt time of the previous page's last trajectory."""

    after_trajectory_id: Nonempty | None = None
    """Identity breaking ties at the preceding receipt position."""

    received_at_from: AwareDatetime | None = None
    """Inclusive lower receipt-time bound."""

    received_at_to: AwareDatetime | None = None
    """Inclusive upper receipt-time bound."""

    limit: int | None = Field(default=None, ge=1)
    """Maximum returned rows including a lookahead row; None returns all matches."""


class RequestAuth(BaseModel):
    model_config = ConfigDict(extra="forbid")
    """Reject unknown request fields."""

    project_id: Nonempty
    """Project scope checked against the trusted caller identity by the service."""

    user_id: Nonempty | None = None
    """Business user identity from the Harness context."""

    agent_id: Nonempty | None = None
    """Agent identity from the Harness context."""


class SkillCommit(BaseModel):
    """Commit metadata shared by structured and local Skill inputs."""

    model_config = ConfigDict(extra="forbid")
    """Reject obsolete or misspelled content fields."""

    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    """User metadata stored on the Skill family, including extra frontmatter fields."""

    commit_message: str | None = None
    """Initial registration and first-version explanation."""

    commit_version: UserVersionLabel | None = None
    """Optional user-assigned version label; None selects automatic numbering."""

    version_bump: Literal["patch", "minor", "major"] = "patch"
    """Component incremented by automatic numbering; lower components reset."""

    commit_source: Literal["user", "algo"] = "user"
    """Whether a user or an algorithm submitted the content."""


class FileSkillInfo(SkillCommit):
    """Read a standard Skill from a local path in the Python process."""

    path: Nonempty
    """Directory or SKILL.md path resolved only by the local caller."""


class StructuredSkillInfo(SkillCommit):
    """Complete portable source content for registration or a version commit."""

    content: str
    """Full SKILL.md text, preserved without newline or metadata rewriting."""

    resources: dict[str, str] = Field(default_factory=dict)
    """Auxiliary file paths mapped to standard Base64 of their original bytes."""

    name: Nonempty | None = None
    """Initial display name; duplicate display names are allowed."""

    description: str | None = None
    """Version description parsed from standard frontmatter or supplied explicitly."""

    standard_format: bool = True
    """Require valid name and description in frontmatter when true."""

    @model_validator(mode="after")
    def validate_source(self) -> Self:
        """Validate source files and resolve display metadata without changing text."""
        source = SkillContent(
            content=self.content,
            resources=decode_resources(self.resources),
            name=self.name,
            description=self.description,
            standard_format=self.standard_format,
        )
        self.name = source.name
        self.description = source.description
        if self.standard_format:
            _, _, header_metadata = parse_skill_header(self.content)
            self.metadata = {**header_metadata, **self.metadata}
        return self


class SkillPipelineRequest(RequestAuth):
    """Select one existing Skill for evolution.

    Backend choices and evidence preparation policies belong to the concrete
    pipeline configuration. Implementations needing additional per-call inputs
    may define typed request extensions and must validate them before I/O.
    This contract does not prescribe trajectory queries or dataset selection.
    """

    model_config = ConfigDict(extra="forbid")
    """Reject undeclared inputs rather than silently ignoring selection options."""

    pipeline: Nonempty | None = None
    """Configured instance name; omission requires exactly one available pipeline."""

    skill_ref: Nonempty
    """Skill identity or reference resolved within the selected project."""

    base_version_id: Nonempty | None = None
    """Owned parent version; None captures the active version at run start."""

    auto_accept: bool = True
    """Whether to select the evolved version as the active Skill version."""

    run_id: str | None = Field(default=None, pattern=r"^evolve_[0-9a-f]{32}$")
    """Stable evolution identity for retries; None starts an independent run."""


class CollectionEvolutionRequest(RequestAuth):
    """Select exact readable Skill versions for collection evolution.

    Backend choices and evidence preparation policies belong to the concrete
    pipeline configuration. Implementations needing additional per-call inputs
    may define typed request extensions and must validate them before I/O.
    This contract does not prescribe trajectory queries or dataset selection.
    """

    model_config = ConfigDict(extra="forbid")
    """Reject undeclared inputs rather than silently ignoring selection options."""

    pipeline: Nonempty | None = None
    """Configured instance name; omission requires exactly one available pipeline."""

    skill_version_ids: list[Nonempty] = Field(min_length=1)
    """Exact readable input versions, with at most one version per Skill family."""

    editable_skills: list[Nonempty] | None = None
    """Input Skill IDs allowed to change; None selects all, [] selects none."""

    allow_create_skills: bool = True
    """Whether the algorithm may propose new independent Skills."""

    auto_accept: bool = True
    """Whether saved output versions replace the active versions of existing Skills."""

    run_id: str | None = Field(default=None, pattern=r"^evolve_[0-9a-f]{32}$")
    """Stable evolution identity for retries; None starts an independent run."""

    @model_validator(mode="after")
    def validate_collection(self) -> Self:
        """Reject duplicate version and editable member identities."""
        if len(set(self.skill_version_ids)) != len(self.skill_version_ids):
            raise ValueError("Evolution input versions must be unique")
        if self.editable_skills is not None and len(set(self.editable_skills)) != len(self.editable_skills):
            raise ValueError("Editable Skill IDs must be unique")
        return self


EvolutionRequest = SkillPipelineRequest | CollectionEvolutionRequest
"""Prepared single-Skill or collection input accepted by evolution pipelines."""

EVOLUTION_REQUEST_ADAPTER = TypeAdapter(EvolutionRequest)
"""Validate stored and library evolution requests without algorithm-specific dispatch."""


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


class TrainingRequest(RequestAuth):
    """Supply complete tasks, explicit split membership, and an input Skill version."""

    model_config = ConfigDict(extra="forbid", use_attribute_docstrings=True)
    """Reject undeclared inputs, including obsolete Skill references."""

    algo: Nonempty
    """Registered training algorithm name, such as treeskill or skill_forest."""

    env: EnvironmentConfig
    """Environment implementation and its complete configuration for this run."""

    init_skill_version_id: Nonempty = Field(...)
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

    @field_validator("tasks", mode="before")
    @classmethod
    def restore_samples(cls, value: object, info: ValidationInfo) -> object:
        """Restore serialized samples using the selected environment's data models.

        Args:
            value: Materialized samples or JSON-compatible sample dictionaries.
            info: Validated fields, including the environment selection.

        Returns:
            Samples retaining their environment-specific task and scoring fields.
        """
        from ..envs import Environment

        env = info.data.get("env")
        if env is None or not isinstance(value, list):
            return value
        implementation = Environment.resolve(env.class_name)
        samples = []
        for sample in value:
            if isinstance(sample, dict) and "task" in sample and "reward_data" in sample:
                sample = {
                    **sample,
                    "task": implementation.task_type.model_validate(sample["task"]),
                    "reward_data": implementation.reward_data_type.model_validate(sample["reward_data"]),
                }
            samples.append(sample)
        return samples

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


class AlgoSkillEvolveResult(BaseModel):
    """Identify a completed run whose output Skill version was persisted.

    Return only after successful persistence. Candidates, pending jobs, and
    failed or conflicting commits are not successful results of this contract.
    Algorithm-specific metrics may be added through typed result extensions.
    """

    model_config = ConfigDict(extra="forbid")
    """Reject undeclared result fields and incomplete identity contracts."""

    skill_id: Nonempty
    """Resolved identity of the evolved or trained Skill."""

    base_version_id: Nonempty
    """Version captured before execution and used as the committed parent."""

    version_id: Nonempty
    """Persisted output version, whether or not it became active."""

    run_id: Nonempty
    """Pipeline execution identity linking inputs, artifacts, and saved output."""

    content: str
    """Exact algorithm-produced document stored in the output version."""

    resources: dict[str, str]
    """Complete auxiliary files encoded as canonical Base64."""

    content_hash: Nonempty
    """Digest of the exact output document."""

    skill_hash: Nonempty
    """Digest of the output document and auxiliary files."""


class EvolutionPublishedSkill(BaseModel):
    """One immutable member outcome in a published evolution collection."""

    model_config = ConfigDict(extra="forbid")
    """Reject undeclared outcome data."""

    key: str = Field(min_length=1)
    """Original input ID or stable run-local creation key."""

    skill_id: str = Field(min_length=1)
    """Actual repository identity after publication."""

    action: Literal["updated", "unchanged", "created"]
    """Repository effect published for this member."""

    base_version_id: str | None = None
    """Exact input parent; None only for a newly created family."""

    version_id: str = Field(min_length=1)
    """Persisted immutable output version, whether or not it became active."""

    snapshot: SkillSnapshot
    """Complete persisted output."""

    @model_validator(mode="after")
    def validate_outcome(self) -> Self:
        """Require a parent version for every existing member."""
        if (self.action == "created") != (self.base_version_id is None):
            raise ValueError("Only newly created families may omit a parent input version")
        return self


class EvolutionRunResult(BaseModel):
    """Identify the persisted versions and snapshots of a completed collection evolution."""

    model_config = ConfigDict(extra="forbid")
    """Require explicit collection outcomes rather than a synthetic primary version."""

    run_id: str = Field(pattern=r"^evolve_[0-9a-f]{32}$")
    """Stable execution identity used to inspect this publication."""

    input_version_ids: dict[str, str]
    """Original Skill IDs mapped to their exact input versions."""

    skills: dict[str, SkillSnapshot]
    """Persisted complete input outputs, including unchanged members."""

    created_skills: dict[str, SkillSnapshot] = Field(default_factory=dict)
    """New complete outputs keyed by run-local names; members contains their repository identities."""

    members: list[EvolutionPublishedSkill] = Field(min_length=1)
    """Persisted repository identities and outcomes for every returned Skill."""


EvolutionResult = AlgoSkillEvolveResult | EvolutionRunResult
"""Persisted single-Skill or collection evolution outcome."""


class ProjectRequest(RequestAuth):
    """Select an explicitly authorized project for a repository operation."""


class SkillRequest(ProjectRequest):
    """Resolve one Skill family by identity or alias within a project."""

    skill_ref: Nonempty
    """Exact alias, full Skill ID, or unambiguous Skill ID prefix."""


class RegisterRequest(RequestAuth):
    """Register a new central identity from portable structured source content."""

    skill: StructuredSkillInfo
    """Validated source content and commit metadata."""

    alias: Nonempty | None = None
    """Optional project-unique shortcut."""

    @model_validator(mode="after")
    def validate_initial_version(self):
        """Reject increments without a parent version."""
        if self.skill.version_bump != "patch":
            raise ValueError("Initial registration has no version to increment")
        return self


class RegisterFromLocalRequest(RequestAuth):
    """Register a standard Skill read by the local Python process."""

    skill: FileSkillInfo
    """Local source path and shared commit metadata."""

    alias: Nonempty | None = None
    """Optional project-unique family shortcut."""


class RegisterResult(BaseModel):
    """Identify a registered Skill and its initial immutable version."""

    skill_id: Nonempty
    """Stable Skill family identity."""

    skill_version_id: Nonempty
    """Initial exact version identity."""

    content_hash: str
    """Digest of the exact SKILL.md text."""

    skill_hash: str
    """Digest of the full source content and resources."""


class ListRequest(RequestAuth):
    """Read a bounded page of central Skills."""

    limit: int | None = Field(default=None, ge=0)
    """Maximum number of returned records. None means no limit."""

    cursor: SkillListCursor = Field(default_factory=SkillListCursor)
    """Snapshot cutoff and optional continuation position."""

    @model_validator(mode="after")
    def validate_cursor(self) -> Self:
        """Require a complete cursor without combining it with offset paging."""
        if (self.cursor.after_created_at is None) != (self.cursor.after_skill_id is None):
            raise ValueError("Cursor position requires both creation time and Skill ID")
        if self.cursor.after_skill_id is not None and self.cursor.snapshot_at is None:
            raise ValueError("Cursor position requires a snapshot cutoff")
        if self.cursor.snapshot_at is not None and self.cursor.offset != 0:
            raise ValueError("Cursor paging cannot use offset")
        return self


class ReportTrajectoryRequest(RequestAuth):
    """Report one Agent execution using an exact registered Skill version."""

    messages: list[dict[str, JsonValue]] = Field(min_length=1)
    """Ordered interaction messages for this physical execution."""

    session_id: Nonempty | None = None
    """Optional logical Agent session identity."""

    skill_version_id: Nonempty
    """Exact Skill version loaded by the Agent."""

    status: Nonempty = "completed"
    """Caller-declared execution outcome."""

    started_at: AwareDatetime | None = None
    """Actual Agent start time when known."""

    finished_at: AwareDatetime | None = None
    """Actual Agent finish time when known."""

    usage: dict[str, JsonValue] | None = None
    """Number of tokens consumed by the trajectory."""

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        """Require message roles and forward-moving execution times."""
        if any(not message.get("role") or "content" not in message for message in self.messages):
            raise ValueError("Each message requires role and content")
        if self.started_at and self.finished_at and self.finished_at < self.started_at:
            raise ValueError("finished_at must not precede started_at")
        return self


class ShowTrajectoryRequest(RequestAuth):
    """Read one trajectory within its owning project."""

    trajectory_id: Nonempty
    """Exact server-generated or local trajectory identity."""


class ListTrajectoriesRequest(RequestAuth):
    """Read one stable page of reported trajectories for an exact family."""

    skill_id: Nonempty
    """Exact Skill family identity."""

    skill_version_id: Nonempty | None = None
    """Optional exact loaded version filter."""

    status: Nonempty | None = None
    """Optional reported execution status filter."""

    cursor: TrajectoryListCursor = Field(default_factory=TrajectoryListCursor)
    """Receipt cutoff, time bounds, and optional continuation position."""

    @model_validator(mode="after")
    def validate_cursor(self) -> Self:
        """Require both parts of a receipt cursor and ordered bounds."""
        if (self.cursor.after_received_at is None) != (self.cursor.after_trajectory_id is None):
            raise ValueError("Trajectory cursor requires receipt time and ID")
        if (
            self.cursor.received_at_from is not None
            and self.cursor.received_at_to is not None
            and self.cursor.received_at_to < self.cursor.received_at_from
        ):
            raise ValueError("Receipt time bounds are reversed")
        return self


class ShowRequest(RequestAuth):
    """Inspect an exact version or the selected version of one family."""

    skill_id: Nonempty | None = None
    """Family identity; required when version_id is omitted."""

    version_id: Nonempty | None = None
    """Owned local version ID; omitted selects the active version."""

    @model_validator(mode="after")
    def validate_selection(self) -> Self:
        """Require an exact version or a family for default selection."""
        if self.skill_id is None and self.version_id is None:
            raise ValueError("Supply skill_id or version_id")
        return self


class VersionResolveRequest(RequestAuth):
    """Find one project-owned version matching supplied hashes or an exact ID."""

    content_hash: Nonempty | None = None
    """Optional digest of the exact source text."""

    skill_hash: Nonempty | None = None
    """Optional digest of the source text and all resources."""

    skill_id: Nonempty | None = None
    """Optional exact Skill family identity."""

    skill_version_id: Nonempty | None = None
    """Optional exact version identity, checked against every supplied constraint."""

    @model_validator(mode="after")
    def validate_lookup(self) -> Self:
        """Require at least one immutable lookup constraint."""
        if self.content_hash is None and self.skill_hash is None and self.skill_version_id is None:
            raise ValueError("Supply content_hash, skill_hash, or skill_version_id")
        return self


class VersionResolution(BaseModel):
    """Exact immutable source version selected by a hash lookup."""

    skill_id: str
    """Owning Skill family identity."""

    skill_version_id: str
    """Matching exact version identity."""

    alias: str | None
    """Optional family shortcut."""

    commit_message: str | None
    """Explanation saved with this version."""

    content_hash: str
    """Stored exact-text digest."""

    skill_hash: str
    """Stored complete-source digest."""

    version: str | None
    """Immutable display label of the matched version."""

    is_active: bool
    """Whether this version is the family active version."""


class ExportRequest(ShowRequest):
    """Export stored content without binding or tracking the destination."""

    directory: Nonempty
    """Parent directory receiving a named Skill folder with the complete document and resources."""

    overwrite: bool = False
    """Permit replacing the named Skill child, including removal of stale files."""


class CommitRequest(RequestAuth):
    """Create a child snapshot and atomically select it."""

    skill_id: Nonempty
    """Identifier of the Skill to which the version is committed."""

    skill: StructuredSkillInfo
    """Complete structured source content and commit metadata."""

    base_version_id: Nonempty | None = None
    """Expected parent version; None selects the active version at commit time."""

    active: bool = True
    """Whether the published version replaces the active version."""

    inherit_resources: bool = False
    """Reuse all parent resources; supplied resources must be empty when true."""

    @model_validator(mode="after")
    def validate_resource_inheritance(self) -> Self:
        """Reject ambiguous replacement and inheritance in the same commit."""
        if self.inherit_resources and self.skill.resources:
            raise ValueError("Resource inheritance cannot include replacement resources")
        return self


class CommitFromLocalRequest(RequestAuth):
    """Commit a standard local Skill through the structured write pipeline."""

    skill_id: Nonempty
    """Target Skill family identity."""

    skill: FileSkillInfo
    """Local source path and shared commit metadata."""

    base_version_id: Nonempty | None = None
    """Explicit parent version; None captures the active version."""

    active: bool = True
    """Whether the published version becomes active."""

    inherit_resources: bool = False
    """Reuse parent resources only when the local directory has no resources."""


class HistoryRequest(RequestAuth):
    """Read immutable versions belonging to one Skill family."""

    skill_id: Nonempty
    """Family identity or alias whose history is requested."""

    limit: int | None = Field(default=None, ge=0)
    """Maximum number of versions returned."""

    cursor: SkillHistoryCursor = Field(default_factory=SkillHistoryCursor)
    """Chronological position within this family's history."""


class DiffRequest(RequestAuth):
    """Compare two project-owned versions without changing active selections."""

    to_version_id: Nonempty
    """Target version to compare."""

    from_version_id: Nonempty | None = None
    """Starting version; omitted selects the active version."""


class ActiveRequest(RequestAuth):
    """Select an existing version without changing external files."""

    version_id: Nonempty
    """Owned version to make active."""

    expected_active_version_id: Nonempty | None = None
    """Optional previous pointer for detecting stale client activation requests."""


class SetMetaRequest(RequestAuth):
    """Change display metadata without creating a version."""

    skill_id: Nonempty

    name: Nonempty | None = None
    """Replacement display name; omitted leaves the name unchanged."""

    alias: Nonempty | None = None
    """Replacement alias; omitted leaves the alias unchanged."""

    clear_alias: bool = False
    """Remove the family alias when true."""

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


class TrajectoryStatisticsRequest(RequestAuth):
    """Query bounded aggregates for one owned Skill version."""

    skill_id: Nonempty
    """Owning Skill family or alias."""

    version_id: Nonempty
    """Exact Skill version whose stored trajectories are counted."""


class SkillDetail(BaseModel):
    """A central identity and the inspected version, which may be historical."""

    skill: SkillRecord
    """Current central metadata and selected-version pointer."""

    version: SkillVersionRecord
    """Complete requested version snapshot."""


class OrganizeRequest(RequestAuth):
    """Select project-owned Skill versions for a complete graph publication."""

    skill_refs: list[Nonempty]
    """Family or version references resolved once before construction."""

    graph_id: Nonempty | None = None
    """Existing graph identity; None creates a new graph."""

    alias: Nonempty | None = None
    """Alias for a new graph; cannot rename an existing graph."""

    commit_message: Nonempty | None = None
    """Explanation recorded with the published snapshot."""

    force_extract_scene: bool = Field(default=False, strict=True)
    """Refresh Scene data even when the selected version has a valid cache."""


class GraphMembersRequest(RequestAuth):
    """Select a nonempty batch of members in an existing graph."""

    graph_ref: Nonempty
    """Project-owned graph family identity or alias, excluding snapshot references."""

    skill_refs: list[Nonempty] = Field(min_length=1)
    """Skill family or version references; repeated families are invalid."""

    commit_message: Nonempty | None = None
    """Explanation recorded with the new snapshot."""

    @model_validator(mode="after")
    def validate_members(self):
        """Reject duplicate references before resolving their canonical families."""
        if len(self.skill_refs) != len(set(self.skill_refs)):
            raise ValueError("Skill references must be unique")
        return self


class AddGraphMembersRequest(GraphMembersRequest):
    """Add new families without replacing any existing graph member."""

    force_extract_scene: bool = Field(default=False, strict=True)
    """Refresh Scene data for new members even when cached data is valid."""


class RemoveGraphMembersRequest(GraphMembersRequest):
    """Remove selected families without deleting their canonical Skills."""


class OrganizeResult(BaseModel):
    """Identify the published graph and its complete immutable member scope."""

    graph_id: Nonempty
    """Public graph family identity."""

    graph_version_id: Nonempty
    """Immutable snapshot created or reused by the operation."""

    skill_version_ids: list[Nonempty]
    """Exact versions of all members in the resulting graph."""


class SkillAnalyzeRequest(RequestAuth):
    """Run or reuse static analysis for one exact stored Skill version."""

    skill_version_id: Nonempty
    """Exact version belonging to the authorized project."""

    request_id: Nonempty | None = None
    """Optional trace identity excluded from the reusable result key."""

    force_refresh: bool = Field(default=False, strict=True)
    """Skip a completed cache hit and execute the configured analyzer again."""


class TrajectoryAnalyzeRequest(RequestAuth):
    """Run or reuse analysis for one exact stored trajectory."""

    trajectory_id: Nonempty
    """Exact physical attempt belonging to the authorized project."""

    request_id: Nonempty | None = None
    """Optional trace identity excluded from the reusable result key."""

    attribute: bool | None = None
    """Optional per-request override for failure attribution."""

    verifier_score: FiniteFloat | None = None
    """Optional finite verifier score that overrides the stored reward."""

    force_refresh: bool = Field(default=False, strict=True)
    """Skip a completed cache hit and execute the configured analyzer again."""


class EvaluationQueryRequest(RequestAuth):
    """Read completed evaluations for exact Skill versions."""

    skill_version_ids: list[Nonempty] = Field(min_length=1)
    """Exact versions returned by the caller or a preceding recommendation."""

    judge_types: list[Literal["static_skill", "trajectory"]] | None = None
    """Optional analysis families to include."""

    limit: int = Field(default=200, ge=1, le=1000, strict=True)
    """Maximum number of evaluation rows returned."""


class EvaluationQueryResult(BaseModel):
    """Completed evaluation rows in requested version order."""

    items: list[EvaluationRecord]
    """Stored results with stable version, judge type, and recency ordering."""


class RecommendationEvaluation(EvaluationRecord):
    """Public evaluation projection attached to recommended versions."""


class RecommendRequest(RequestAuth):
    """Retrieve Skills from one pinned graph using a public instruction."""

    graph_id: Nonempty
    """Project-owned graph identity."""

    graph_version_id: Nonempty | None = None
    """Immutable snapshot; None pins the current graph head."""

    task: str = Field(min_length=1)
    """Public instruction used for retrieval."""

    top_k: int = Field(default=5, ge=1, strict=True)
    """Maximum number of unique Skill versions to return."""

    strategy: Literal["graph", "vector"] = "graph"
    """Graph propagation or direct cosine-similarity retrieval."""

    include_scenes: bool = Field(default=False, strict=True)
    """Vector-only option allowing Scene nodes to contribute owner matches."""

    include_evaluation: bool = Field(default=False, strict=True)
    """Opt in to completed evaluation rows attached after retrieval."""

    @model_validator(mode="after")
    def validate_strategy(self):
        """Reject vector-only options in graph mode and empty query text."""
        if self.strategy == "graph" and self.include_scenes:
            raise ValueError("include_scenes requires the vector strategy")
        if not self.task.strip():
            raise ValueError("Task instruction must not be blank")
        return self


class GraphRecommendationScore(BaseModel):
    """Core graph score components for one exact Skill version."""

    skill_version_id: Nonempty
    """Exact selected version corresponding to the result's Skill list."""

    ppr_score: float = Field(ge=0, le=1, allow_inf_nan=False)
    """Normalized Skill-node PageRank mass."""

    scene_boost: float = Field(ge=0, allow_inf_nan=False)
    """Weighted contribution from child Scene nodes."""

    graph_score: float = Field(ge=0, allow_inf_nan=False)
    """Skill PageRank mass plus the Scene contribution."""

    rerank_score: float = Field(ge=0, allow_inf_nan=False)
    """Final field-based score used to order graph candidates."""

    @model_validator(mode="after")
    def validate_total(self):
        """Require the graph total to match its two components."""
        from math import isclose

        if not isclose(self.graph_score, self.ppr_score + self.scene_boost, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("graph_score must equal ppr_score plus scene_boost")
        return self


class VectorRecommendationScore(BaseModel):
    """Winning cosine-similarity node for one exact Skill version."""

    skill_version_id: Nonempty
    """Exact selected owner version."""

    similarity: float = Field(ge=-1, le=1, allow_inf_nan=False)
    """Cosine similarity of the best matching node for this owner."""

    node_id: Nonempty
    """Winning Skill or Scene node within the pinned snapshot."""

    node_type: GraphNodeType
    """Type of the winning node."""


class RecommendResult(BaseModel):
    """Selected versions and strategy-specific scores from one immutable snapshot."""

    skills: list[SkillVersionRecord]
    """Selected versions in descending score order."""

    graph_version_id: Nonempty
    """Snapshot actually used throughout the query."""

    strategy: Literal["graph", "vector"]
    """Retrieval strategy determining the score meaning."""

    score_type: Literal["graph_components", "cosine_similarity"]
    """Explicit score units; graph scores are not cosine similarities."""

    scores: list[GraphRecommendationScore] | list[VectorRecommendationScore]
    """Core scores in exactly the same order as skills."""

    include_scenes: bool = False
    """Whether vector retrieval searched Scene nodes."""

    evaluations: list[RecommendationEvaluation] | None = None
    """Optional completed evaluations in recommendation version order."""

    @model_validator(mode="after")
    def validate_scores(self):
        """Keep score types, version identities, and result order consistent."""
        graph = self.strategy == "graph"
        expected_type = GraphRecommendationScore if graph else VectorRecommendationScore
        expected_units = "graph_components" if graph else "cosine_similarity"
        if self.score_type != expected_units or (graph and self.include_scenes):
            raise ValueError("Score semantics differ from the retrieval strategy")
        if any(not isinstance(score, expected_type) for score in self.scores):
            raise ValueError("Score fields differ from the retrieval strategy")
        if [score.skill_version_id for score in self.scores] != [skill.version_id for skill in self.skills]:
            raise ValueError("Scores must match selected versions in order")
        if len({skill.version_id for skill in self.skills}) != len(self.skills):
            raise ValueError("Recommended versions must be unique")
        selected = {skill.version_id for skill in self.skills}
        if self.evaluations is not None:
            if any(item.skill_version_id not in selected for item in self.evaluations):
                raise ValueError("Evaluations must reference recommended versions")
            if len({item.evaluation_id for item in self.evaluations}) != len(self.evaluations):
                raise ValueError("Recommendation evaluations must be unique")
        return self


class SkillPageRequest(RequestAuth):
    """Filter one stable page of Skill families across repository evidence."""

    name_pattern: Nonempty | None = None
    """Optional case-sensitive wildcard pattern for family names."""

    alias_pattern: Nonempty | None = None
    """Optional case-sensitive wildcard pattern for family aliases."""

    limit: int | None = Field(default=None, ge=1)
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


class LocalKeyFile(BaseModel):
    """Credentials for one locally deployed Skill project."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    """Reject ambiguous or misspelled key file fields."""

    project_id: str = Field(min_length=1)
    """Project assigned to all requests carrying this key."""

    api_key: SecretStr
    """Bearer credential accepted by this local deployment."""

    enabled: bool = True
    """Whether this credential is allowed to authenticate."""

    scopes: list[Literal["skill:read", "skill:write"]] = Field(default_factory=lambda: ["skill:write"])
    """Explicit Skill permissions; omitted fields preserve existing local-key behavior."""


class LocalKeyTable(BaseModel):
    """Credentials for all projects accepted by one local Skill deployment."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    """Reject unknown table fields without exposing credentials."""

    api_keys: list[LocalKeyFile]
    """Project credentials and their explicit enabled state and permissions."""
