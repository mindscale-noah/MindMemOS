"""Public request and response types for the standalone Skill service."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, JsonValue, StringConstraints, model_validator

from .config import EnvironmentConfig

NonemptyIdentity = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
"""Nonblank service identity, normalized like the HTTP request boundary."""


class SkillRequestBody(BaseModel):
    """Validate the fields sent to one Skill endpoint."""

    model_config = ConfigDict(extra="forbid")
    """Reject unknown public fields before sending a request."""


class StructuredSkillInfo(SkillRequestBody):
    """Portable source document, resources, and publication metadata."""

    content: str
    """Exact SKILL.md text, including its original newlines."""

    resources: dict[str, str] = Field(default_factory=dict)
    """Canonical Base64 resource bytes keyed by relative path."""

    name: str | None = None
    """Name parsed from frontmatter, or required for nonstandard content."""

    description: str | None = None
    """Description parsed from frontmatter, or required for nonstandard content."""

    standard_format: bool = True
    """Whether the server must validate standard SKILL.md frontmatter."""

    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    """User metadata saved with this publication."""

    commit_message: str | None = None
    """Explanation saved with the version."""

    commit_version: str | None = None
    """Explicit semantic label; None allocates it automatically."""

    version_bump: Literal["patch", "minor", "major"] = "patch"
    """Version component incremented during automatic labeling."""

    commit_source: Literal["user", "algo"] = "user"
    """Origin of this publication."""

    @model_validator(mode="after")
    def validate_resources(self):
        """Reject invalid resource paths or Base64 before HTTP I/O."""
        from .content import decode_resources

        decode_resources(self.resources)
        return self


class RegisterRequest(SkillRequestBody):
    """Register structured content as a new family."""

    skill: StructuredSkillInfo
    """Complete portable source and commit metadata."""

    alias: str | None = None
    """Optional project-unique family shortcut."""


class ListRequest(SkillRequestBody):
    """Page Skill families within the authenticated project."""

    filters: dict[str, str] = Field(default_factory=dict)
    """Optional name, alias, user_id, and agent_id filters."""

    limit: int | None = Field(default=None, ge=1)
    """Maximum number of families in one page."""

    cursor: str | None = None
    """Opaque continuation cursor from the preceding page."""


class ShowRequest(SkillRequestBody):
    """Read one family and a selected version."""

    skill_id: str | None = None
    """Exact family identity."""

    skill_version_id: str | None = None
    """Historical version identity, or None for the active version."""

    @model_validator(mode="after")
    def validate_selection(self):
        """Require a family or an exact immutable version."""
        if self.skill_id is None and self.skill_version_id is None:
            raise ValueError("Supply skill_id or skill_version_id")
        return self


class CommitRequest(SkillRequestBody):
    """Publish structured content as a child of an existing version."""

    skill_id: str
    """Target family identity."""

    skill: StructuredSkillInfo
    """Complete replacement source and commit metadata."""

    base_version_id: str | None = None
    """Explicit parent; omission selects the active version."""

    active: bool = True
    """Whether to activate the new version."""

    inherit_resources: bool = False
    """Inherit parent resources explicitly; replacement resources must be empty."""

    @model_validator(mode="after")
    def validate_inheritance(self):
        """Reject conflicting inheritance and replacement instructions."""
        if self.inherit_resources and self.skill.resources:
            raise ValueError("Resource inheritance cannot include replacement resources")
        return self


class GraphSnapshotsRequest(SkillRequestBody):
    """List immutable snapshots of one project-owned graph."""

    graph_id: str
    """Exact graph identity in the authorized project."""


class GraphSnapshotRequest(GraphSnapshotsRequest):
    """Read the contents of one immutable graph snapshot."""

    graph_version_id: str
    """Exact graph snapshot identity."""


class HistoryRequest(SkillRequestBody):
    """Page immutable versions of one family."""

    skill_id: str
    """Exact family identity."""

    limit: int | None = Field(default=None, ge=1)
    """Maximum number of versions in one page."""

    offset: int = 0
    """Number of earlier versions to skip."""

    cursor: str | None = None
    """Opaque history continuation token from the preceding response."""


class DiffRequest(SkillRequestBody):
    """Compare two stored versions."""

    skill_id: str
    """Source family identity."""

    to_skill_version_id: str
    """Exact target version identity."""

    from_skill_version_id: str | None = None
    """Source version identity, or None for the active version."""

    to_skill_id: str | None = None
    """Target family identity, or None for the source family."""


class ActivateRequest(SkillRequestBody):
    """Make an existing version current."""

    skill_id: str | None = None
    """Exact family identity."""

    skill_version_id: str
    """Exact owned version identity."""

    expected_active_version_id: str | None = None
    """Optional pointer used to reject stale activation requests."""


class UpdateInfoRequest(SkillRequestBody):
    """Change family fields without creating a version."""

    skill_id: str
    """Exact family identity."""

    name: str | None = None
    """New display name, or None when unchanged."""

    alias: str | None = None
    """New alias; explicit None clears it."""

    metadata: dict[str, JsonValue] | None = None
    """Replacement family metadata, or None when unchanged."""


class DeleteRequest(SkillRequestBody):
    """Delete an unreferenced family."""

    skill_id: str
    """Exact family identity."""


class VersionResolveRequest(SkillRequestBody):
    """Resolve a source hash to an exact version."""

    content_hash: str | None = None
    """Optional exact-text SHA-256 digest."""

    skill_hash: str | None = None
    """Optional complete-source SHA-256 digest."""

    skill_id: str | None = None
    """Optional family identity narrowing the lookup."""

    skill_version_id: str | None = None
    """Optional exact version identity requiring a family identity."""


class SkillResolveRequest(SkillRequestBody):
    """Resolve a Skill alias, family, or immutable version."""

    skill_ref: str = Field(min_length=1)
    """Project-owned alias or unique prefixed Skill identity."""


class TrajectoryReportRequest(SkillRequestBody):
    """Report one execution with its exact loaded Skill versions."""

    user_id: str
    """Business user who requested the execution."""

    agent_id: str
    """Agent that performed the execution."""

    messages: list[dict[str, JsonValue]]
    """Ordered execution messages with role and content."""

    skill_version_ids: list[NonemptyIdentity]
    """Exact loaded versions. Empty means no exact version was identified."""

    event_id: NonemptyIdentity | None = None
    """Stable project-unique identity for replaying an identical Skill report."""

    session_id: str | None = None
    """Optional Harness session identity."""

    started_at: datetime | None = None
    """Actual execution start time when known."""

    finished_at: datetime | None = None
    """Actual execution finish time when known."""

    status: str = "completed"
    """Caller-declared execution status."""


class TrajectoryStatisticsRequest(SkillRequestBody):
    """Query all-time totals and fixed thirty-day counts without record content."""

    skill_id: str = Field(min_length=1)
    """Owning Skill family identity or alias."""

    skill_version_id: str = Field(min_length=1)
    """Exact Skill version to aggregate."""


class TrajectoryShowRequest(SkillRequestBody):
    """Read one stored execution attempt."""

    trajectory_id: str
    """Server-generated attempt identity."""


class TrajectoryListRequest(SkillRequestBody):
    """Page execution attempts for one Skill family."""

    skill_id: str
    """Exact family identity."""

    skill_version_id: str | None = None
    """Optional exact version filter."""

    status: str | None = None
    """Optional execution status filter."""

    received_at_from: datetime | None = None
    """Inclusive server receipt time lower bound."""

    received_at_to: datetime | None = None
    """Inclusive server receipt time upper bound."""

    limit: int | None = Field(default=None, ge=1)
    """Maximum number of attempts in one page."""

    cursor: str | None = None
    """Opaque continuation cursor from the preceding page."""


class OrganizeRequest(SkillRequestBody):
    """Publish an explicit Skill collection into a new or existing graph."""

    skill_refs: list[str]
    """Project-owned family or immutable version references."""

    graph_id: str | None = None
    """Existing graph identity; omission creates one."""

    alias: str | None = None
    """Optional alias for a new graph."""

    commit_message: str | None = None
    """Explanation stored with the graph publication."""

    force_extract_scene: bool = Field(default=False, strict=True)
    """Refresh version Scene data even when the saved cache is valid."""


class GraphResolveRequest(SkillRequestBody):
    """Resolve a project-scoped graph reference."""

    graph_ref: str = Field(min_length=1)
    """Graph ID, alias, or unique ID prefix."""


class RecommendRequest(GraphResolveRequest):
    """Request Skill recommendations from one graph."""

    graph_version_id: str | None = None
    """Optional exact published snapshot overriding the current graph pointer."""

    query: str
    """Task instruction for graph retrieval."""

    top_k: int | None = None
    """Maximum number of recommendations when supplied."""

    strategy: Literal["graph", "vector"] = "graph"
    """Graph propagation or direct vector similarity."""

    include_scenes: bool = Field(default=False, strict=True)
    """Vector-only option matching Scene nodes as well as Skill nodes."""

    include_evaluation: bool = Field(default=False, strict=True)
    """Attach stored evaluations without changing recommendation ordering."""

    @model_validator(mode="after")
    def validate_strategy(self):
        """Reject vector-only options in graph mode and blank instructions."""
        if self.strategy == "graph" and self.include_scenes:
            raise ValueError("include_scenes requires the vector strategy")
        if not self.query.strip():
            raise ValueError("Query must not be blank")
        return self


class SkillAnalyzeRequest(SkillRequestBody):
    """Run or reuse static scoring for an exact Skill version."""

    skill_version_id: NonemptyIdentity
    """Exact registered version to assess."""

    request_id: NonemptyIdentity | None = None
    """Optional trace identity, excluded from the evaluation cache key."""

    force_refresh: bool = Field(default=False, strict=True)
    """Recompute even when a completed evaluation is cached."""


class TrajectoryAnalyzeRequest(SkillRequestBody):
    """Run or reuse analysis of one stored execution."""

    trajectory_id: NonemptyIdentity
    """Exact stored trajectory to assess."""

    request_id: NonemptyIdentity | None = None
    """Optional trace identity, excluded from the evaluation cache key."""

    attribute: bool | None = None
    """Override failure attribution; None uses the service configuration."""

    verifier_score: FiniteFloat | None = None
    """Optional finite verifier score overriding the stored reward."""

    force_refresh: bool = Field(default=False, strict=True)
    """Recompute even when a completed evaluation is cached."""


class EvaluationQueryRequest(SkillRequestBody):
    """Read completed evaluations without running an analyzer."""

    skill_version_ids: list[NonemptyIdentity] = Field(min_length=1)
    """Exact versions whose stored evaluations should be returned."""

    judge_types: list[Literal["static_skill", "trajectory"]] | None = None
    """Optional analyzer families to include; None includes both."""

    limit: int = Field(default=200, ge=1, le=1000, strict=True)
    """Maximum number of evaluation records returned."""


class GraphMembersRequest(GraphResolveRequest):
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


class AddGraphMembersRequest(GraphMembersRequest):
    """Add new families without replacing existing graph members."""

    force_extract_scene: bool = Field(default=False, strict=True)
    """Refresh Scene data for the newly selected versions."""


class RemoveGraphMembersRequest(GraphMembersRequest):
    """Remove graph members without deleting canonical Skill versions."""


class EvolveRequest(SkillRequestBody):
    """Submit exact Skill versions to one configured evolution pipeline."""

    skill_version_ids: list[NonemptyIdentity] = Field(min_length=1)
    """One or more exact input versions, with at most one version per Skill family."""

    auto_accept: bool = True
    """Whether to activate the published candidate version."""

    pipeline: NonemptyIdentity | None = None
    """Allowed server pipeline name; omission uses the configured HTTP default."""

    editable_skills: list[NonemptyIdentity] | None = None
    """Input Skill IDs whose contents may change; None permits all, [] permits none."""

    allow_create_skills: bool = True
    """Whether independent new Skill candidates may be created."""

    @model_validator(mode="after")
    def validate_selection(self):
        """Reject duplicate version and editable Skill IDs."""
        for values in (self.skill_version_ids, self.editable_skills):
            if values is not None and len(values) != len(set(values)):
                raise ValueError("Evolution identities must be unique")
        return self


class TaskShowRequest(SkillRequestBody):
    """Read one asynchronous business task."""

    task_id: str
    """Server-generated task identity."""


class TrainingSplitRequest(SkillRequestBody):
    """Assign each training task to exactly one data split."""

    train: list[str] = Field(min_length=1)
    """Ordered task IDs used to optimize the Skill."""

    val: list[str] = Field(default_factory=list)
    """Ordered task IDs used for candidate selection."""

    test: list[str] = Field(default_factory=list)
    """Held-out task IDs evaluated after training."""


class TrainingRequest(SkillRequestBody):
    """Submit a Skill training run to the authorized service project."""

    init_skill_version_id: str
    """Current stored version used as the training input and commit parent."""

    work_dir: str
    """Stable server-side run directory used for resumable training."""

    tasks: list[dict[str, JsonValue]] = Field(min_length=1)
    """Public tasks paired with private reward data for the server-side environment."""

    split: TrainingSplitRequest
    """Exact assignment of submitted task IDs to train, validation, and test."""

    algo: str
    """Configured training algorithm profile."""

    env: EnvironmentConfig
    """Environment implementation and configuration supplied for this run."""

    auto_accept: bool = True
    """Activate the trained version after successful publication."""


class TaskProgressRequest(SkillRequestBody):
    """Read task logs after one committed log sequence number."""

    task_id: str
    """Server-generated training task identity."""

    after: int = Field(default=0, ge=0)
    """Return task logs with a sequence number above this value; gaps are allowed."""


SkillData = dict[str, Any] | list[dict[str, Any]]
"""Decoded data payload returned by a Skill service endpoint."""


class CLISkillInfo(StructuredSkillInfo):
    """Validate direct CLI content before uploading it to the service."""

    @model_validator(mode="after")
    def validate_content(self):
        """Require standard frontmatter or explicit nonstandard identities."""
        from .content import parse_frontmatter

        if self.standard_format:
            header = parse_frontmatter(self.content)
            if self.name is not None and self.name != header["name"]:
                raise ValueError("Skill name differs from frontmatter")
            if self.description is not None and self.description != header["description"]:
                raise ValueError("Skill description differs from frontmatter")
        elif not self.name or not self.name.strip() or not self.description or not self.description.strip():
            raise ValueError("Nonstandard Skills require nonempty name and description")
        return self


class CLIRegisterInput(SkillRequestBody):
    """Keep local registration options separate from a complete JSON Skill."""

    source: str | None = None
    """Local directory or SKILL.md path; mutually exclusive with json_skill."""

    json_skill: CLISkillInfo | None = None
    """Complete structured input without local file or metadata overrides."""

    version: str | None = None
    """Local initial label; omission uses v0.0.1."""

    message: str | None = None
    """Local publication explanation; omission uses Initial registration."""

    metadata_json: dict[str, JsonValue] | None = None
    """Local publication metadata; omission uses an empty mapping."""

    include: str | None = None
    """Local resource inclusion patterns."""

    exclude: str | None = None
    """Local resource exclusion patterns."""

    @model_validator(mode="after")
    def validate_input_mode(self):
        """Require exactly one source and reject overrides of JSON content."""
        if (self.source is None) == (self.json_skill is None):
            raise ValueError("Supply either a local path or --json-skill")
        if self.json_skill is not None and any(
            value is not None for value in (self.version, self.message, self.metadata_json, self.include, self.exclude)
        ):
            raise ValueError("--json-skill cannot be combined with local registration options")
        return self


class CLICommitInput(SkillRequestBody):
    """Validate a complete JSON commit or a local source with publication options."""

    source: str | None = None
    """Local directory or SKILL.md path; mutually exclusive with json_skill."""

    json_skill: CLISkillInfo | None = None
    """Complete replacement source, resources, and publication metadata."""

    message: str | None = None
    """Required publication explanation for local sources."""

    version: str | None = None
    """Explicit local version label; mutually exclusive with bump."""

    bump: Literal["patch", "minor", "major"] | None = None
    """Local automatic version increment; omission uses patch."""

    resources_dir: str | None = None
    """Local replacement resource directory; omission inherits parent resources."""

    clear_resources: bool = False
    """Explicitly clear local resources instead of inheriting them."""

    @model_validator(mode="after")
    def validate_input_mode(self):
        """Reject mixed inputs and require the selected source's commit message."""
        if (self.source is None) == (self.json_skill is None):
            raise ValueError("Supply either --from or --json-skill")
        if self.json_skill is not None:
            if self.clear_resources or any(
                value is not None for value in (self.message, self.version, self.bump, self.resources_dir)
            ):
                raise ValueError("--json-skill cannot be combined with local commit options")
            message = self.json_skill.commit_message
        else:
            if self.version is not None and self.bump is not None:
                raise ValueError("--version and --bump are mutually exclusive")
            if self.resources_dir is not None and self.clear_resources:
                raise ValueError("--resources-dir and --clear-resources are mutually exclusive")
            message = self.message
        if message is None or not message.strip():
            raise ValueError("Supply a nonempty --message for local sources or commit_message in --json-skill")
        return self


class CLITrainingRequest(TrainingRequest):
    """Validate CLI task inputs and split references before submitting training."""

    @model_validator(mode="after")
    def validate_tasks(self):
        """Reject malformed samples, duplicate identities, and overlapping splits."""
        ids = []
        for sample in self.tasks:
            task = sample.get("task")
            if not isinstance(task, dict):
                raise ValueError("Each sample requires a task object")
            task_id, instruction = task.get("id"), task.get("instruction")
            if not isinstance(task_id, str) or not task_id.strip():
                raise ValueError("Task IDs must be nonempty strings")
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError("Task instructions must be nonempty strings")
            if not isinstance(sample.get("reward_data"), dict):
                raise ValueError("Each sample requires a reward_data object")
            ids.append(task_id)
        split_ids = self.split.train + self.split.val + self.split.test
        if len(set(ids)) != len(ids) or len(set(split_ids)) != len(split_ids) or not set(split_ids) <= set(ids):
            raise ValueError("Task IDs must be unique and splits must be disjoint subsets of supplied tasks")
        return self
