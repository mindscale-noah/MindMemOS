"""Validated workbench requests."""

from typing import Literal

from mindmemos_skill.persistence.db_services.skills.typing import UserVersionLabel
from pydantic import BaseModel, ConfigDict, Field, model_validator


class RequestModel(BaseModel):
    """Reject unsupported request fields."""

    model_config = ConfigDict(extra="forbid")
    """Catch accidental or misspelled input fields."""


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


class RegisterRequest(RequestModel):
    """Local registration input."""

    path: str = Field(min_length=1)
    """Source Skill directory or SKILL.md path."""

    version_label: UserVersionLabel | None = None
    """Optional semantic label for the initial version; defaults to v1.0.0."""

    name: str = Field(min_length=1)
    """Central display name."""

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


class SwitchRequest(RequestModel):
    """A selected version to make current without changing external files."""

    skill_id: str
    """Local repository Skill identity."""
    version_id: str
    """Local version to select without modifying imported files."""


class CommitRequest(RequestModel):
    """Create a child version from a local source directory."""

    skill_id: str
    """Central Skill identity."""

    path: str = Field(min_length=1)
    """Complete source directory or SKILL.md path."""

    base_version_id: str
    """Expected current version for optimistic concurrency."""

    message: str | None = None
    """Explanation for the new version."""

    version_label: UserVersionLabel | None = None
    """Explicit user version alias; omitted for automatic allocation."""

    version_bump: Literal["patch", "minor", "major"] = "patch"
    """Numeric component to increment when a label is not supplied."""


class ConfigImportRequest(RequestModel):
    """Import a small ZIP containing YAML and its companion environment file."""

    archive: str = Field(max_length=3000000)
    """Base64-encoded ZIP bytes, never extracted to the filesystem."""
