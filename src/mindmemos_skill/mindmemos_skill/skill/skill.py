"""Source Skill value object independent of persistence and runtime backends."""

from copy import copy
from pathlib import Path
from typing import TYPE_CHECKING, Self

from mindmemos_sdk.skills.files import resolve_skill_export_directory

from .io import export_skill, read_skill
from .typing import SkillContent, SkillSnapshot

if TYPE_CHECKING:
    from ..envs.typing import Task


class Skill:
    """Own one immutable-view Skill source and its runtime snapshot metadata."""

    def __init__(self, source: SkillContent, *, template: SkillSnapshot | None = None):
        """Copy Skill content without binding to an external directory.

        Args:
            source: Complete version content and metadata.
            template: Optional runtime configuration retained from a snapshot.
        """
        self._source = SkillContent.model_validate(source.model_dump()).model_copy(deep=True)
        self._template = template.model_copy(deep=True) if template is not None else None

    @property
    def source(self) -> SkillContent:
        """Return an isolated view of complete source files."""
        return self._source.model_copy(deep=True)

    def with_content(self, content: str, *, name: str | None = None, description: str | None = None) -> Self:
        """Create a candidate with replacement text and unchanged resources.

        Args:
            content: Complete replacement SKILL.md text.
            name: Explicit display name for nonstandard content.
            description: Explicit display description for nonstandard content.

        Returns:
            New Skill preserving source resources and runtime settings.
        """
        source = SkillContent(
            content=content,
            resources=self._source.resources.copy(),
            name=name if self._source.standard_format or name is not None else self._source.name,
            description=description
            if self._source.standard_format or description is not None
            else self._source.description,
            standard_format=self._source.standard_format,
        )
        candidate = copy(self)
        candidate._source = source
        return candidate

    @property
    def content(self) -> str:
        """Return the full source document without assembly or rewriting."""
        return self._source.content

    @property
    def resources(self) -> dict[str, bytes]:
        """Return an isolated mapping of auxiliary source bytes."""
        return dict(self._source.resources)

    def with_resources(self, resources: dict[str, bytes]) -> Self:
        """Replace the complete resource set without changing the document.

        Args:
            resources: Complete replacement resource paths and original bytes.

        Returns:
            Independent candidate retaining runtime dependencies and source metadata.
        """
        values = self._source.model_dump()
        values["resources"] = resources
        candidate = copy(self)
        candidate._source = SkillContent.model_validate(values)
        return candidate

    async def assemble(self, task: "Task") -> Self:
        """Return an isolated task-local copy while retaining runtime dependencies.

        Args:
            task: Public task input for implementations that customize source text.

        Returns:
            A Skill of the same concrete class with independently owned source files.
        """
        candidate = copy(self)
        candidate._source = self._source.model_copy(deep=True)
        return candidate

    def snapshot(self) -> SkillSnapshot:
        """Capture content and opaque runtime settings with both source hashes."""
        template = self._template
        return SkillSnapshot.from_content(
            self._source,
            implementation=template.implementation if template else "static",
            config_schema_version=template.config_schema_version if template else 1,
            config=template.config if template else None,
        )

    @classmethod
    def from_snapshot(cls, snapshot: SkillSnapshot) -> Self:
        """Restore a stored Skill after validating both source hashes.

        Args:
            snapshot: Complete portable snapshot.

        Returns:
            Restored Skill with original text, resources, and runtime settings.
        """
        validated = SkillSnapshot.model_validate(snapshot.model_dump())
        return cls(validated.to_content(), template=validated)

    @classmethod
    def from_path(cls, path: str | Path) -> Self:
        """Read a standard Skill directory without newline conversion.

        Args:
            path: Skill directory or its SKILL.md entrypoint.

        Returns:
            Validated source Skill.
        """
        return cls(read_skill(path))

    def resolve_export_directory(self, directory: str | Path) -> Path:
        """Resolve the named export folder without writing any files.

        Args:
            directory: Parent directory for the exported Skill.

        Returns:
            Absolute path of the named Skill child directory.
        """
        return resolve_skill_export_directory(
            self._source.content,
            directory,
            name=self._source.name,
            standard_format=self._source.standard_format,
        )

    def export(self, directory: str | Path, *, overwrite: bool = False) -> Path:
        """Write complete source files into a dedicated directory.

        Args:
            directory: Parent directory for the named Skill child.
            overwrite: Replace the named child, removing stale files, when true.

        Returns:
            Absolute destination path.
        """
        return export_skill(self._source, directory, overwrite=overwrite)
