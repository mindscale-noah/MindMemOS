"""Transport-independent inputs and persisted results of Skill pipelines."""

from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from ..envs.data.typing import DataSample, SplitManifest
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
    """Expected current version; None captures the current version at run start.

    An explicit value must match the current version before execution. It is
    not a request to branch from history. The captured version remains the
    parent throughout the run and must be checked again when committing.
    """


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

    pipeline: Nonempty | None = None
    """Configured instance name; omission requires exactly one available pipeline."""

    project_id: Nonempty | None = None
    """Repository scope; omission is allowed only through a locally bound service."""

    base_version_id: Nonempty = Field(...)
    """Required input version, checked as the current parent before training and commit."""

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
    """Identify a completed run whose selected Skill version was persisted.

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
    """Persisted version selected by this successful operation."""

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


class ShowRequest(SkillRequest):
    """Inspect the selected or explicitly requested version."""

    version_id: Nonempty | None = None
    """Owned local version ID; omitted selects the current version."""


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
    """Expected current parent; omitted captures the current version before reading source."""

    message: str | None = None
    """Explanation stored with the new version."""


class HistoryRequest(SkillRequest):
    """Read a bounded page of immutable version snapshots."""

    limit: int = Field(default=100, ge=0, le=1000)
    """Maximum number of versions returned."""

    offset: int = Field(default=0, ge=0)
    """Number of chronologically sorted versions to skip."""


class DiffRequest(SkillRequest):
    """Compare two owned versions without changing the current selection."""

    to_version_id: Nonempty
    """Target version to compare."""

    from_version_id: Nonempty | None = None
    """Starting version; omitted selects the current version."""


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

    @model_validator(mode="after")
    def validate_changes(self):
        """Reject contradictory and empty metadata edits."""
        if self.alias is not None and self.clear_alias:
            raise ValueError("alias and clear_alias cannot be combined")
        if self.name is None and self.alias is None and not self.clear_alias:
            raise ValueError("Supply name, alias, or clear_alias")
        return self


class SkillDetail(BaseModel):
    """A central identity and the inspected version, which may be historical."""

    skill: SkillRecord
    """Current central metadata and selected-version pointer."""

    version: SkillVersionRecord
    """Complete requested version snapshot."""


class FileChange(BaseModel):
    """One added, removed, or modified file in a version comparison."""

    path: str
    """Relative bundle or resource path."""

    change: str
    """Added, deleted, or modified."""

    binary: bool
    """Whether either side is non-textual."""

    diff: str
    """Unified text diff; empty for binary changes."""


class DiffResult(BaseModel):
    """Content and implementation changes between owned versions."""

    from_version_id: str
    """Resolved starting version."""

    to_version_id: str
    """Resolved target version."""

    files: list[FileChange]
    """Changed files only, in path order."""

    implementation_changed: bool
    """Whether implementation name, configuration schema, or fixed options differ."""
