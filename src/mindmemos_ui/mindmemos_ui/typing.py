"""Validated workbench requests."""

from typing import Literal

from mindmemos_skill.persistence.db_services.skills.typing import UserVersionLabel
from mindmemos_skill.skill import decode_resources
from pydantic import BaseModel, ConfigDict, Field, JsonValue, StrictBool, model_validator


class RequestModel(BaseModel):
    """Reject unsupported request fields."""

    model_config = ConfigDict(extra="forbid")
    """Catch accidental or misspelled input fields."""


class SkillListPage(BaseModel):
    """One browser-visible page with portable Skill metadata."""

    items: list[dict[str, JsonValue]]
    """SDK family JSON rows extended with active_version_id, active_version_label,
    implementation, latest_version_id, latest_version_label, latest_implementation;
    optional metadata retains the service's arbitrary JSON keys.
    """

    next_cursor: str | None
    """Opaque service cursor for the next page; None marks the final page."""

    has_more: bool
    """Whether the service supplied a continuation cursor."""


class MemoryUpdateRequest(RequestModel):
    """Edit the body of one existing memory without changing its metadata."""

    content: str = Field(min_length=1, pattern=r"\S")
    """Non-blank replacement text, preserving its whitespace and line breaks."""


class MemorySearchRequest(RequestModel):
    """Search text submitted from the read-only Memories page."""

    query: str = Field(min_length=1, max_length=2000, pattern=r"\S")
    """Non-empty natural-language query sent to the Memory service."""

    top_k: int = Field(default=20, ge=1)
    """Maximum number of memories returned by the search."""

    user_id: str | None = Field(default=None, min_length=1, pattern=r"\S")
    """Optional user filter; None uses the configured default user."""


class DirectorySelectionRequest(RequestModel):
    """Initial location for the host's native directory chooser."""

    home: str | None = Field(default=None, max_length=4096)
    """Preferred directory; omission uses the active UI workspace."""


class EnvironmentVariable(RequestModel):
    """One literal environment assignment stored alongside a YAML document."""

    name: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    """Portable environment variable name."""

    value: str = Field(max_length=100000)
    """Literal value; credentials remain local to the selected home."""


class ServiceConfigUpdate(RequestModel):
    """Complete service document with optimistic concurrency protection."""

    content: str = Field(min_length=1, max_length=1000000)
    """YAML source, including comments preserved on save."""

    revision: str | None = None
    """Hash of the last loaded file; null means the file must not exist."""

    env: list[EnvironmentVariable] = Field(default_factory=list, max_length=200)
    """Environment assignments saved to the sibling .env file."""

    @model_validator(mode="after")
    def unique_names(self):
        """Reject duplicate assignments instead of silently keeping the last value."""
        if len({item.name for item in self.env}) != len(self.env):
            raise ValueError("Environment variable names must be unique")
        return self


class SDKSettingsUpdate(RequestModel):
    """Complete SDK v2 JSON document with optimistic concurrency protection."""

    content: str = Field(min_length=1, max_length=1000000)
    """JSON source for a complete SDK v2 settings document."""

    revision: str | None = None
    """Hash of the last loaded settings file; None requires a missing file."""


class RegisterRequest(RequestModel):
    """Local registration input."""

    path: str = Field(min_length=1)
    """Source Skill directory or SKILL.md path."""

    version_label: UserVersionLabel | None = None
    """Optional semantic label for the initial version; defaults to v1.0.0."""

    message: str | None = None
    """Registration explanation."""

    alias: str | None = None
    """Optional local lookup alias."""


class VersionRef(RequestModel):
    """Explicit version identity and source namespace."""

    source: str = Field(pattern="^(graph|local)$")
    """Local repository or graph navigation source."""

    skill_id: str
    """Central Skill identity."""

    version_id: str
    """Immutable version in the selected namespace."""


class DiffRequest(RequestModel):
    """Two snapshots to compare."""

    before: VersionRef
    """Baseline version."""

    after: VersionRef
    """Target version."""


class AssembleRequest(RequestModel):
    """Assemble one or two immutable versions for the same public task."""

    versions: list[VersionRef] = Field(min_length=1, max_length=2)
    """Source versions in display order; duplicate identities are assembled once."""

    instruction: str = Field(min_length=1, max_length=100000, pattern=r"\S")
    """Public task instruction passed unchanged to each Skill."""


class ExportRequest(RequestModel):
    """Inputs for a reviewed export plan."""

    version: VersionRef
    """Source snapshot."""

    directory: str = Field(min_length=1)
    """Final destination Skill directory."""


class ConfirmRequest(RequestModel):
    """A one-use operation preview."""

    plan_id: str
    """Server-owned preview identity."""


class ActiveRequest(RequestModel):
    """A selected version to make active without changing external files."""

    skill_id: str
    """Local repository Skill identity."""
    version_id: str
    """Local version to select without modifying imported files."""


class CommitVersionRequest(RequestModel):
    """Shared version identity and publication options."""

    skill_id: str
    """Central Skill identity."""

    active: bool = True
    """Whether to make the committed version active; preserves existing callers' behavior."""

    base_version_id: str
    """Explicit parent version for optimistic concurrency."""

    message: str | None = None
    """Explanation for the new version."""

    version_label: UserVersionLabel | None = None
    """Explicit user version alias; omitted for automatic allocation."""

    version_bump: Literal["patch", "minor", "major"] = "patch"
    """Numeric component to increment when a label is not supplied."""


class CommitRequest(CommitVersionRequest):
    """Create a child version from a local source directory."""

    path: str = Field(min_length=1)
    """Complete source directory or SKILL.md path."""


class EditCommitRequest(CommitVersionRequest):
    """Publish the complete source snapshot edited in the browser."""

    content: str = Field(min_length=1)
    """Full replacement SKILL.md text, preserving whitespace."""

    resources: dict[str, str]
    """All retained auxiliary files, mapping relative paths to canonical Base64 bytes."""

    @model_validator(mode="after")
    def validate_resources(self):
        """Reject unsafe paths and invalid encodings before publication."""
        decode_resources(self.resources)
        return self


class ConfigImportRequest(RequestModel):
    """Import a small ZIP containing YAML and its companion environment file."""

    archive: str = Field(max_length=3000000)
    """Base64-encoded ZIP bytes, never extracted to the filesystem."""


class AlgorithmDirectoryRequest(RequestModel):
    """Select a local algorithm output directory."""

    path: str = Field(min_length=1, max_length=4096)
    """Directory on the machine hosting the workbench."""


class ForestSkill(BaseModel):
    """Display-only subset of a checkpoint Skill."""

    name: str = ""
    """Human-readable Skill name."""

    content: str = ""
    """Saved Skill text, without runtime connection configuration."""


class ForestCheckpoint(BaseModel):
    """Version-independent display fields from a saved node."""

    id: str
    """Node identity within its training run."""

    tree_id: int
    """Tree index recorded by the trainer."""

    parent_id: str | None = None
    """Parent identity, absent for a root."""

    depth: int = 0
    """Distance from the root."""

    status: Literal["root", "generated", "accepted", "rejected"]
    """Persisted decision; generated does not prove a running process."""

    scores: dict[str, float] = Field(default_factory=dict)
    """Validated task IDs mapped to binary success scores."""

    gain: float = 0
    """Saved change in mean validation success; partial rejection uses a sentinel zero."""

    expanded: bool = False
    """Whether candidate expansion was completed."""

    skill: ForestSkill = Field(default_factory=ForestSkill)
    """Safe fields used by the node inspector."""


class ForestNodeView(BaseModel):
    """Compact node view derived from saved evidence."""

    winner: Literal["tree", "final"] | None = None
    """Official selection role, absent until a completed result identifies the node."""

    checkpoint: ForestCheckpoint
    """Saved checkpoint with only display fields retained."""

    completed: int
    """Distinct validation tasks with successful execution records."""

    total: int | None
    """Expected validation tasks, or None when unavailable."""

    score: float | None
    """Mean over observed validation successes, including partial observations."""

    gain: float | None
    """Final gain only when the complete validation set is available."""

    errors: int = 0
    """Invalid or failed evaluation records, excluded from completed tasks."""


class ForestProgress(BaseModel):
    """Training milestones inferred from saved artifacts, not process liveness."""

    stage: Literal["root", "search", "select", "completed", "unknown"] = "unknown"
    """Latest phase evidenced by files in this training root."""

    roots_completed: int = 0
    """Number of saved root checkpoints."""

    roots_total: int | None = None
    """Expected roots from the persisted split manifest."""

    expansions_completed: int = 0
    """Parents whose candidate expansion has been committed."""

    expansion_budget: int | None = None
    """Maximum parent expansions, not a guaranteed final count."""


class ForestRunView(BaseModel):
    """One independent training root; runs are never merged."""

    path: str
    """Absolute training root path."""

    progress: ForestProgress = Field(default_factory=ForestProgress)
    """Observed stage and committed work counts."""

    nodes: list[ForestNodeView]
    """Readable checkpoints in this run."""


class ForestReport(BaseModel):
    """Read-only snapshot of discovered Skill-Forest runs."""

    runs: list[ForestRunView]
    """Independent runs discovered beneath the selected directory."""

    warnings: list[str]
    """Incomplete, inaccessible, or bounded evidence notices."""


class ForestSplit(BaseModel):
    """Validation membership stored in a training manifest."""

    validation_ids: list[str]
    """Task IDs expected for one tree's validation."""


class ForestTrainerBudget(BaseModel):
    """Display-only subset of the persisted trainer settings."""

    expansion_budget: int | None = Field(default=None, ge=0)
    """Configured upper limit on parent expansions."""


class ForestManifest(BaseModel):
    """Only the non-sensitive split fields needed for progress totals."""

    trainer: ForestTrainerBudget = Field(default_factory=ForestTrainerBudget)
    """Non-sensitive settings needed to show search progress."""

    splits: list[ForestSplit]
    """Validation splits ordered by tree index."""


class ForestReward(BaseModel):
    """Binary validation outcome used by the trainer."""

    success: StrictBool
    """Whether evaluation passed, distinct from execution success."""


class ForestEvaluationResult(BaseModel):
    """Safe subset of an incremental evaluation result."""

    reward: ForestReward
    """Completed environment evaluation."""

    error: JsonValue = None
    """Explicit execution error, if present in an external artifact."""


class ForestEvaluationRecord(BaseModel):
    """One completed task entry from evaluation.jsonl."""

    task_id: str
    """Identity used to deduplicate validation observations."""

    result: ForestEvaluationResult
    """Scored outcome without trajectories or model configuration."""


class ForestWinnerRecord(BaseModel):
    """One officially selected tree winner."""

    id: str
    """Winner checkpoint identity."""

    tree_id: int
    """Tree owning the winner."""


class ForestSelectionReport(BaseModel):
    """Final selection identity from the algorithm report."""

    selected_node_id: str | None = None
    """Final winner ID; None for synthesized ensemble outputs."""


class ForestTrainingSelection(BaseModel):
    """Saved per-tree winners and final selection."""

    winners: list[ForestWinnerRecord]
    """Official tree winners, not inferred from displayed scores."""

    algorithm_report: ForestSelectionReport
    """Final selected node recorded by the trainer."""


class ForestSelectionResult(BaseModel):
    """Minimal completed-result fields used for winner highlighting."""

    status: str
    """Only completed results authorize winner markers."""

    training: ForestTrainingSelection
    """Authoritative selection evidence within this training root."""
