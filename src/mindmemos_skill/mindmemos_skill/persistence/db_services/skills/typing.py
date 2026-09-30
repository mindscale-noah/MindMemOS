"""Validated records owned by this business data service."""

from typing import Annotated, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, JsonValue, StringConstraints, model_validator

from ....common.time_utils import utcnow
from ...typing import RecordModel
from .versions import ALGORITHM_PATTERN, USER_VERSION_PATTERN


class SkillRecord(RecordModel):
    """One centrally managed Skill with exactly one selected version."""

    project_id: str = Field(min_length=1)
    """Local project owning the Skill."""

    skill_id: str = Field(min_length=1)
    """Stable central identity, independent of import and export paths."""

    name: str = Field(min_length=1)
    """Mutable display name; not a unique identity."""

    alias: str | None = Field(default=None, min_length=1)
    """Optional exact-match shortcut, unique within the local project."""

    registration_source: str | None = None
    """Initial source reference for provenance only; never watched or synchronized."""

    registration_message: str | None = None
    """Initial registration explanation, separate from per-version commit messages."""

    current_version_id: str = Field(min_length=1)
    """Only selected version for default reads and exports; belongs to this Skill."""

    created_at: AwareDatetime = Field(default_factory=utcnow)
    """Time the Skill was registered in the central repository."""

    updated_at: AwareDatetime = Field(default_factory=utcnow)
    """Last metadata or selected-version change; supplied by the writer."""

    @model_validator(mode="after")
    def validate_timing(self):
        """Reject registration timestamps in reverse order."""
        if self.updated_at < self.created_at:
            raise ValueError("updated_at must not precede created_at")
        return self


class SkillVersionRecord(RecordModel):
    """Complete insertable record for skill_versions."""

    project_id: str = Field(min_length=1)
    """Project owning this immutable local Skill version."""

    version_id: str = Field(min_length=1)
    """Local version identity; never inferred from a remote version ID."""

    skill_id: str = Field(min_length=1)
    """Local Skill family shared by all of its versions."""

    name: str = Field(min_length=1)
    """Human-readable Skill name."""

    description: str | None = None
    """Optional explanation of the Skill."""

    parent_version_ids: list[str] = Field(default_factory=list)
    """Ordered local parent IDs; remote single-parent projection is separate."""

    bundle: str = Field(min_length=1)
    """Serialized Skill bundle, preserved without cloud-specific rewriting."""

    resources: str = "{}"
    """Serialized auxiliary files outside the Skill bundle."""

    content_hash: str = Field(min_length=1)
    """Hash of the local canonical bundle; not the remote hash."""

    local_snapshot_hash: str | None = None
    """Optional hash including local auxiliary resources."""

    implementation: str = "static"
    """Registered concrete Skill implementation name."""

    config_schema_version: int = Field(default=1, ge=1)
    """Version of the fixed Skill configuration format."""

    config: dict[str, JsonValue] = Field(default_factory=dict)
    """Fixed Skill implementation options; mutable content belongs in bundle."""

    status: str = "draft"
    """Local lifecycle state; remote lifecycle is stored separately."""

    version_revision: int = Field(default=0, ge=0)
    """Local metadata revision for conditional updates."""

    version_label: str | None = None
    """Unique semantic display label; None remains readable for pre-upgrade writers."""

    commit_message: str | None = None
    """Reason for creating the version."""

    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    """Portable Skill metadata."""

    local_metadata: dict[str, JsonValue] = Field(default_factory=dict)
    """Local-only annotations."""

    origin: str = "local"
    """Local provenance such as local, imported, or evolved."""

    created_at: AwareDatetime = Field(default_factory=utcnow)
    """Time this local version was created."""

    updated_at: AwareDatetime = Field(default_factory=utcnow)
    """Time local metadata was last changed."""

    @model_validator(mode="after")
    def validate_lineage(self):
        """Reject invalid local lineage and reversed timestamps."""
        if self.version_id in self.parent_version_ids or len(set(self.parent_version_ids)) != len(
            self.parent_version_ids
        ):
            raise ValueError("Parents must be unique and cannot include this version")
        if any(not isinstance(parent, str) or not parent for parent in self.parent_version_ids):
            raise ValueError("Parent IDs must be nonempty strings")
        if self.updated_at < self.created_at:
            raise ValueError("updated_at must not precede created_at")
        return self


UserVersionLabel = Annotated[str, StringConstraints(pattern=USER_VERSION_PATTERN, max_length=96)]
"""Validated human-assigned semantic label without a reserved suffix."""

AlgorithmName = Annotated[str, StringConstraints(pattern=ALGORITHM_PATTERN, max_length=48)]
"""Stable algorithm identifier from the publishing pipeline."""


class VersionNaming(BaseModel):
    """Select automatic semantic numbering or an explicit human version label."""

    model_config = ConfigDict(extra="forbid")
    """Reject misspelled version options."""

    version_label: UserVersionLabel | None = None
    """Explicit user label; None allocates a label automatically."""

    version_bump: Literal["patch", "minor", "major"] = "patch"
    """Component incremented by automatic numbering; lower components reset."""

    algorithm: AlgorithmName | None = None
    """Publishing algorithm identity; only pipeline callers set this reserved namespace."""

    @model_validator(mode="after")
    def validate_mode(self):
        """Disallow manual names in the algorithm namespace or ambiguous increments."""
        if self.version_label is not None and (self.algorithm is not None or self.version_bump != "patch"):
            raise ValueError("An explicit label cannot be combined with algorithm or non-default increment")
        if self.algorithm is not None and self.version_bump != "patch":
            raise ValueError("Algorithm publications increment the patch component")
        return self
