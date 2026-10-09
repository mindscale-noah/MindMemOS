"""Validated workbench requests."""

from typing import Literal

from mindmemos_skill.persistence.db_services.skills.typing import UserVersionLabel
from mindmemos_skill.skill import decode_resources
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator


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
