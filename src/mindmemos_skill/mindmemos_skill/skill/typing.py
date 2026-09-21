"""Portable Skill content and implementation snapshots."""

import base64
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator


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


class SkillBundle(BaseModel):
    """Own the mutable text files managed by a Skill."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    """Reject unknown fields and validate replacement file mappings."""

    files: dict[str, str]
    """Managed relative paths and UTF-8 text; currently exactly SKILL.md."""

    @field_validator("files")
    @classmethod
    def normalize_files(cls, files: dict[str, str]) -> dict[str, str]:
        """Validate the managed paths and normalize line endings.

        Args:
            files: Source paths and text supplied by the caller.
        """
        if set(files) != {"SKILL.md"}:
            raise ValueError("A Skill bundle must contain exactly SKILL.md")
        return {path: content.replace("\r\n", "\n").replace("\r", "\n") for path, content in files.items()}

    def serialize(self) -> str:
        """Return the canonical JSON representation used by the SDK."""
        files = self.normalize_files(self.files)
        records = [{"path": path, "content": files[path]} for path in sorted(files)]
        return json.dumps(records, ensure_ascii=False, separators=(",", ":"), sort_keys=True)

    @classmethod
    def deserialize(cls, content: str) -> Self:
        """Read canonical bundle JSON without interpreting invalid JSON as Markdown.

        Args:
            content: Serialized list of path/content records.

        Raises:
            ValueError: Records are malformed, duplicated, or unsupported.
        """
        records = json.loads(content)
        if not isinstance(records, list):
            raise ValueError("A serialized bundle must be a list of path/content records")
        files = {}
        for record in records:
            if not isinstance(record, dict) or set(record) != {"path", "content"}:
                raise ValueError("Each bundle record must contain path and content")
            path, text = record["path"], record["content"]
            if not isinstance(path, str) or not isinstance(text, str) or path in files:
                raise ValueError("Bundle paths must be unique strings and contents must be strings")
            files[path] = text
        return cls(files=files)

    @property
    def content_hash(self) -> str:
        """Compute SHA-256 from canonical UTF-8 bundle JSON."""
        return hashlib.sha256(self.serialize().encode("utf-8")).hexdigest()

    @classmethod
    def from_path(cls, path: str | Path) -> Self:
        """Read SKILL.md or a directory containing it.

        Args:
            path: Skill file or directory; auxiliary files are ignored.
        """
        from .io import read_bundle

        return cls(files=read_bundle(path))

    def export(self, directory: str | Path) -> Path:
        """Write managed files without changing auxiliary files.

        Args:
            directory: Destination Skill directory, created when absent.

        Returns:
            Absolute destination directory.
        """
        from .io import write_bundle

        return write_bundle(self.normalize_files(self.files), directory)


class SkillResources(BaseModel):
    """Preserve auxiliary files unchanged across bundle assembly and optimization."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    """Validate replacement mappings without interpreting auxiliary file content."""

    files: dict[str, bytes] = Field(default_factory=dict)
    """Relative paths and original bytes, excluding the root SKILL.md."""

    @field_validator("files")
    @classmethod
    def validate_paths(cls, files: dict[str, bytes]) -> dict[str, bytes]:
        """Require portable paths contained within the Skill directory.

        Args:
            files: Auxiliary file paths and unmodified bytes.
        """
        for name in files:
            path = PurePosixPath(name)
            if (
                not path.parts
                or path.is_absolute()
                or "\\" in name
                or ".." in path.parts
                or path.as_posix() != name
                or path.parts[0] == "SKILL.md"
            ):
                raise ValueError("Resource paths must be relative and must not replace SKILL.md")
            if any(parent.as_posix() in files for parent in path.parents):
                raise ValueError("A resource cannot be both a file and a directory")
        return dict(files)

    def serialize(self) -> str:
        """Encode resource bytes as a canonical JSON path-to-base64 mapping."""
        files = self.validate_paths(self.files)
        return json.dumps(
            {path: base64.b64encode(content).decode("ascii") for path, content in files.items()},
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    @classmethod
    def deserialize(cls, content: str) -> Self:
        """Restore auxiliary bytes from their persistence representation.

        Args:
            content: JSON object mapping relative paths to base64 strings.
        """
        records = json.loads(content)
        if not isinstance(records, dict) or any(not isinstance(value, str) for value in records.values()):
            raise ValueError("Resources must be a JSON path-to-base64 mapping")
        return cls(files={path: base64.b64decode(value, validate=True) for path, value in records.items()})


class SkillSnapshot(BaseModel):
    """Describe a source Skill independently of database identities."""

    model_config = ConfigDict(extra="forbid")
    """Reject unrecognized implementation or content fields."""

    implementation: str = Field(min_length=1)
    """Source implementation metadata; reading content does not load an implementation."""

    config_schema_version: int = Field(default=1, ge=1)
    """Version of the fixed implementation configuration format."""

    config: dict[str, JsonValue] = Field(default_factory=dict)
    """Fixed implementation options; optimizable content belongs in bundle."""

    bundle: str
    """Canonical serialized source bundle, before task-specific assembly."""

    content_hash: str
    """SHA-256 identity of the source bundle, never the assembled output."""

    resources: str = "{}"
    """Serialized auxiliary bytes, preserved independently of bundle assembly."""

    local_snapshot_hash: str | None = None
    """Optional hash of both source bundle and resources for local identity."""

    def compute_local_hash(self) -> str:
        """Hash canonical content and resources without implementation metadata."""
        content = json.dumps(
            {
                "bundle": SkillBundle.deserialize(self.bundle).serialize(),
                "resources": SkillResources.deserialize(self.resources).serialize(),
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @model_validator(mode="after")
    def validate_content(self) -> Self:
        """Reject a corrupt source snapshot before restoring or resuming it."""
        if SkillBundle.deserialize(self.bundle).content_hash != self.content_hash:
            raise ValueError("Skill content hash does not match its bundle")
        actual_hash = self.compute_local_hash()
        if self.local_snapshot_hash is not None and self.local_snapshot_hash != actual_hash:
            raise ValueError("Skill local snapshot hash does not match its resources and bundle")
        return self
