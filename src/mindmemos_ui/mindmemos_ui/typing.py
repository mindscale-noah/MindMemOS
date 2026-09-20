"""Validated workbench requests."""

from typing import Literal

from mindmemos_skill.persistence.db_services.skills.typing import UserVersionLabel
from pydantic import BaseModel, ConfigDict, Field


class RequestModel(BaseModel):
    """Reject unsupported request fields."""

    model_config = ConfigDict(extra="forbid")
    """Catch accidental or misspelled input fields."""


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
