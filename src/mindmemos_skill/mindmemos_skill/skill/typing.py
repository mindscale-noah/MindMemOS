"""Validated Skill content and portable source snapshots."""

from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter, field_validator, model_validator

from .content import content_hash, decode_resources, encode_resources, skill_hash, validate_resource_paths


def parse_skill_header(content: str) -> tuple[str, str, dict[str, JsonValue]]:
    """Read required name and description from a standard SKILL.md frontmatter.

    Args:
        content: Full source text, including its original line endings.

    Returns:
        Parsed name, description, and additional YAML metadata.

    Raises:
        ValueError: Frontmatter or required fields are invalid.
    """
    lines = content.splitlines(keepends=True)
    if not lines or lines[0].rstrip("\r\n") != "---":
        raise ValueError("Standard SKILL.md must begin with YAML frontmatter")
    closing = next((index for index, line in enumerate(lines[1:], start=1) if line.rstrip("\r\n") == "---"), None)
    if closing is None:
        raise ValueError("Standard SKILL.md frontmatter has no closing delimiter")
    try:
        parsed = yaml.safe_load("".join(lines[1:closing]))
    except yaml.YAMLError as error:
        raise ValueError("SKILL.md frontmatter is invalid YAML") from error
    if not isinstance(parsed, dict) or any(not isinstance(key, str) for key in parsed):
        raise ValueError("SKILL.md frontmatter must be a mapping with text keys")
    name, description = parsed.get("name"), parsed.get("description")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("SKILL.md frontmatter requires a nonempty name")
    if not isinstance(description, str) or not description.strip():
        raise ValueError("SKILL.md frontmatter requires a nonempty description")
    extra = TypeAdapter(dict[str, JsonValue]).validate_python(
        {key: value for key, value in parsed.items() if key not in {"name", "description"}}
    )
    return name.strip(), description.strip(), extra


class SkillContent(BaseModel):
    """Complete source files and display metadata for one Skill version."""

    model_config = ConfigDict(extra="forbid")
    """Reject unknown content fields."""

    content: str
    """Full unchanged SKILL.md text, including original newline characters."""

    resources: dict[str, bytes] = Field(default_factory=dict)
    """Auxiliary relative paths and original file bytes, excluding SKILL.md."""

    name: str | None = None
    """Version display name; derived from standard frontmatter when omitted."""

    description: str | None = None
    """Version description; derived from standard frontmatter when omitted."""

    standard_format: bool = True
    """Require name and description in frontmatter when true."""

    @field_validator("resources")
    @classmethod
    def validate_resources(cls, resources: dict[str, bytes]) -> dict[str, bytes]:
        """Validate every auxiliary file path.

        Args:
            resources: Source file paths and their original bytes.

        Returns:
            The validated mapping.
        """
        validate_resource_paths(resources)
        return resources

    @model_validator(mode="after")
    def validate_metadata(self) -> Self:
        """Resolve standard frontmatter or require explicit nonstandard metadata."""
        if self.standard_format:
            name, description, _ = parse_skill_header(self.content)
            if self.name is not None and self.name != name:
                raise ValueError("Skill name disagrees with SKILL.md frontmatter")
            if self.description is not None and self.description != description:
                raise ValueError("Skill description disagrees with SKILL.md frontmatter")
            self.name = name
            self.description = description
        elif not self.name or not self.name.strip() or not self.description or not self.description.strip():
            raise ValueError("Nonstandard Skill content requires name and description")
        return self

    @property
    def content_hash(self) -> str:
        """Return the digest of the exact stored document text."""
        return content_hash(self.content)

    @property
    def skill_hash(self) -> str:
        """Return the digest of document text and all resource paths and bytes."""
        return skill_hash(self.content, self.resources)


class SkillSnapshot(BaseModel):
    """Portable source content with both hashes checked at restoration."""

    model_config = ConfigDict(extra="forbid")
    """Reject undeclared snapshot fields."""

    implementation: str = Field(default="static", min_length=1)
    """Runtime Skill implementation name, excluded from source hashes."""

    config_schema_version: int = Field(default=1, ge=1)
    """Version of fixed implementation configuration, excluded from hashes."""

    config: dict[str, JsonValue] = Field(default_factory=dict)
    """Fixed runtime options, excluded from source hashes."""

    name: str = Field(min_length=1)
    """Version display name, excluded from source hashes."""

    description: str = Field(min_length=1)
    """Version description, excluded from source hashes."""

    standard_format: bool = True
    """Whether original content was validated as standard SKILL.md format."""

    content: str
    """Unchanged full SKILL.md text."""

    resources: dict[str, str] = Field(default_factory=dict)
    """Auxiliary files encoded as standard Base64 by relative path."""

    content_hash: str
    """SHA-256 digest of UTF-8 content."""

    skill_hash: str
    """SHA-256 digest of versioned canonical source content and resources."""

    @model_validator(mode="after")
    def validate_hashes(self) -> Self:
        """Reject snapshots whose content or resources changed after hashing."""
        resources = decode_resources(self.resources)
        SkillContent(
            content=self.content,
            resources=resources,
            name=self.name,
            description=self.description,
            standard_format=self.standard_format,
        )
        if content_hash(self.content) != self.content_hash:
            raise ValueError("Skill snapshot content hash mismatch")
        if skill_hash(self.content, resources) != self.skill_hash:
            raise ValueError("Skill snapshot source hash mismatch")
        return self

    @classmethod
    def from_content(
        cls,
        source: SkillContent,
        *,
        implementation: str = "static",
        config_schema_version: int = 1,
        config: dict[str, JsonValue] | None = None,
    ) -> Self:
        """Capture a validated source without changing its text or bytes.

        Args:
            source: Version content and display metadata.
            implementation: Runtime implementation name.
            config_schema_version: Version of fixed runtime configuration.
            config: Fixed runtime configuration, excluded from source hashes.

        Returns:
            Portable snapshot with both canonical digests.
        """
        source = SkillContent.model_validate(source.model_dump())
        return cls(
            implementation=implementation,
            config_schema_version=config_schema_version,
            config=config or {},
            name=source.name,
            description=source.description,
            standard_format=source.standard_format,
            content=source.content,
            resources=encode_resources(source.resources),
            content_hash=source.content_hash,
            skill_hash=source.skill_hash,
        )

    def to_content(self) -> SkillContent:
        """Restore source files while checking both stored digests.

        Returns:
            Original Skill content and resource bytes.
        """
        self.validate_hashes()
        return SkillContent(
            content=self.content,
            resources=decode_resources(self.resources),
            name=self.name,
            description=self.description,
            standard_format=self.standard_format,
        )


class SkillSetIdentity(BaseModel):
    """Record collection selection settings and ordered candidate identities."""

    model_config = ConfigDict(extra="forbid")
    """Reject unrecognized persisted collection settings."""

    implementation: str = Field(min_length=1)
    """Fully qualified collection class implementing task-time selection."""

    skill_ids: list[str]
    """Ordered candidate IDs corresponding to the source Skill snapshots."""

    max_concurrency: int = Field(gt=0, strict=True)
    """Maximum simultaneous member assemblies within one collection call."""

    config: dict[str, JsonValue] = Field(default_factory=dict)
    """Non-secret selection settings supplied by the concrete collection."""

    @field_validator("skill_ids")
    @classmethod
    def validate_ids(cls, values: list[str]) -> list[str]:
        """Validate unique, nonempty candidate IDs.

        Args:
            values: Candidate identities in injection order.
        """
        if any(not value.strip() for value in values) or len(set(values)) != len(values):
            raise ValueError("Skill IDs must be unique nonempty strings")
        return values


class FileChange(BaseModel):
    """One changed document or resource in a source comparison."""

    path: str
    """Relative file path, including SKILL.md for the source document."""

    change: Literal["added", "deleted", "modified"]
    """How the file differs between the source snapshots."""

    binary: bool
    """Whether either file contains non-text bytes or NUL characters."""

    diff: str
    """Unified text diff; empty for binary files."""


class SnapshotDiff(BaseModel):
    """Source file, display metadata, and runtime configuration differences."""

    files: list[FileChange]
    """Changed files ordered by relative path."""

    metadata_changed: bool
    """Whether name, description, or standard-format policy differs."""

    implementation_changed: bool
    """Whether implementation identity or fixed runtime configuration differs."""


class DiffResult(BaseModel):
    """Content and implementation changes between owned Skill versions."""

    from_version_id: str
    """Resolved starting version."""

    from_skill_id: str | None = None
    """Source family identity for comparisons across families."""

    to_version_id: str
    """Resolved target version."""

    to_skill_id: str | None = None
    """Target family identity for comparisons across families."""

    files: list[FileChange]
    """Changed files only, in path order."""

    implementation_changed: bool
    """Whether implementation name, configuration schema, or fixed options differ."""
