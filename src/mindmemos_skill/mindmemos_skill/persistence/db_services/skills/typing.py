"""Validated records owned by this business data service."""

from typing import Annotated, Literal

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictInt,
    StringConstraints,
    field_validator,
    model_validator,
)

from ....common.time_utils import utcnow
from ....scene.typing import ProvidedSceneRecord
from ....skill import SkillSnapshot
from ...typing import RecordModel
from .versions import ALGORITHM_PATTERN, USER_VERSION_PATTERN


class SkillRecord(RecordModel):
    """One centrally managed Skill and its selected version."""

    project_id: str = Field(min_length=1)
    """Local project owning the Skill."""

    skill_id: str = Field(min_length=1)
    """Stable central identity, independent of import and export paths."""

    name: str = Field(min_length=1)
    """Mutable display name; not a unique identity."""

    alias: str | None = Field(default=None, min_length=1)
    """Optional exact-match shortcut, unique within the local project."""

    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    """Mutable user metadata belonging to the Skill family, separate from version metadata."""

    registration_source: str | None = None
    """Initial source reference for provenance only; never watched or synchronized."""

    registration_message: str | None = None
    """Initial registration explanation, separate from per-version commit messages."""

    active_version_id: str | None = Field(min_length=1)
    """Only selected version for default reads and exports; belongs to this Skill."""

    created_at: AwareDatetime = Field(default_factory=utcnow)
    """Time the Skill was registered in the central repository."""

    updated_at: AwareDatetime = Field(default_factory=utcnow)
    """Last metadata or selected-version change; supplied by the writer."""

    is_deleted: bool = False
    """Whether this Skill is hidden from business operations."""

    deleted_at: AwareDatetime | None = None
    """Time of soft deletion; None while active."""

    @model_validator(mode="after")
    def validate_timing(self):
        """Reject registration timestamps in reverse order."""
        if self.updated_at < self.created_at:
            raise ValueError("updated_at must not precede created_at")
        return self


class SkillVersionWrite(RecordModel):
    """Skill version input whose project is supplied by the repository scope."""

    version_id: str = Field(min_length=1)
    """Immutable Skill version identity."""

    skill_id: str = Field(min_length=1)
    """Local Skill family shared by all of its versions."""

    name: str = Field(min_length=1)
    """Human-readable Skill name."""

    description: str = Field(min_length=1)
    """Immutable version description."""

    standard_format: bool = True
    """Whether the source requires standard frontmatter."""

    parent_version_ids: list[str] = Field(default_factory=list)
    """Ordered parent version identities."""

    content: str
    """Exact SKILL.md text, including original newlines."""

    resources: dict[str, str] = Field(default_factory=dict)
    """Relative resource paths mapped to canonical Base64 bytes."""

    content_hash: str = Field(min_length=1)
    """Hash of the exact UTF-8 document."""

    skill_hash: str
    """Digest of complete source text and resource bytes."""

    implementation: str = "static"
    """Registered concrete Skill implementation name."""

    config_schema_version: int = Field(default=1, ge=1)
    """Version of the fixed Skill configuration format."""

    config: dict[str, JsonValue] = Field(default_factory=dict)
    """Fixed Skill implementation options; mutable content belongs in content."""

    status: str = "draft"
    """Skill version lifecycle state."""

    version_revision: int = Field(default=0, ge=0)
    """Local metadata revision for conditional updates."""

    version_label: str | None = None
    """Unique semantic display label."""

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
        self.snapshot()
        if self.version_id in self.parent_version_ids or len(set(self.parent_version_ids)) != len(
            self.parent_version_ids
        ):
            raise ValueError("Parents must be unique and cannot include this version")
        if any(not isinstance(parent, str) or not parent for parent in self.parent_version_ids):
            raise ValueError("Parent IDs must be nonempty strings")
        if self.updated_at < self.created_at:
            raise ValueError("updated_at must not precede created_at")
        return self

    def snapshot(self) -> SkillSnapshot:
        """Return portable source content from this already loaded version.

        Returns:
            Source fields without repository identity or mutable metadata.
        """
        return SkillSnapshot(
            implementation=self.implementation,
            config_schema_version=self.config_schema_version,
            config=self.config,
            name=self.name,
            description=self.description,
            standard_format=self.standard_format,
            content=self.content,
            resources=self.resources,
            content_hash=self.content_hash,
            skill_hash=self.skill_hash,
        )


class SkillVersionRecord(SkillVersionWrite):
    """Complete persisted record with its project ownership."""

    project_id: str = Field(min_length=1)
    """Project owning this immutable local Skill version."""


class SkillVersionSummary(RecordModel):
    """Projected version fields returned by a history page."""

    version_id: str = Field(min_length=1)
    """Immutable local version identity."""

    version_label: str | None
    """Display label, if assigned to this version."""

    commit_message: str | None = None
    """Author's submission message; None when no message was supplied."""

    parent_version_ids: list[str]
    """Ordered parent version identities."""

    content_hash: str = Field(min_length=1)
    """Hash of the exact UTF-8 document."""

    skill_hash: str
    """Digest of the full document and resource bytes."""

    implementation: str
    """Registered Skill implementation name."""

    status: str
    """Local lifecycle state."""

    origin: str
    """Local provenance of this version."""

    created_at: AwareDatetime
    """Time this version was created."""


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


SCENE_SCHEMA_VERSION = 1
"""Only supported persisted Scene format."""


class SceneData(BaseModel):
    """Version-owned validated Scene cache."""

    model_config = ConfigDict(extra="forbid")
    """Reject undeclared cache fields."""

    schema_version: StrictInt = SCENE_SCHEMA_VERSION
    """Strict format version; strings and booleans are invalid."""

    record: ProvidedSceneRecord | None
    """Validated Scene semantics; None caches a confirmed insufficient-source result."""

    @field_validator("schema_version")
    @classmethod
    def validate_schema(cls, value):
        """Reject unsupported cache versions without conversion."""
        if value != SCENE_SCHEMA_VERSION:
            raise ValueError("Unsupported Scene schema version")
        return value
