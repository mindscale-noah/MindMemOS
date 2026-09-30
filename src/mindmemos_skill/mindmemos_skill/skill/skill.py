"""Shared Skill content, assembly, and portable restoration contract."""

from copy import copy
from pathlib import Path
from typing import TYPE_CHECKING, Self

if TYPE_CHECKING:
    from ..envs.typing import Task

from .typing import SkillBundle, SkillResources, SkillSnapshot


class Skill:
    """Own Skill content and resources independently of runtime backends."""

    def __init__(self, *, bundle: SkillBundle, resources: SkillResources | None = None):
        """Own copies of source content and auxiliary files.

        Args:
            bundle: Source content, independent of any particular task.
            resources: Auxiliary files preserved unchanged by assembly.
        """
        self._bundle = SkillBundle.deserialize(bundle.serialize())
        self._resources = (
            SkillResources.deserialize(resources.serialize()) if resources is not None else SkillResources()
        )
        self._snapshot_template: SkillSnapshot | None = None

    @property
    def bundle(self) -> SkillBundle:
        """Return an independent source bundle; changes require with_bundle."""
        return self._bundle.model_copy(deep=True)

    @property
    def resources(self) -> SkillResources:
        """Return an isolated view of unchanged reference documents and assets."""
        return self._resources.model_copy(deep=True)

    @classmethod
    def from_path(cls, path: str | Path) -> Self:
        """Load source content and auxiliary files from a Skill directory.

        Args:
            path: A Skill directory or its SKILL.md entrypoint.
        """
        from .io import read_resources

        return cls(bundle=SkillBundle.from_path(path), resources=read_resources(path))

    def export(self, directory: str | Path, *, overwrite: bool = False) -> Path:
        """Export current bundle and unchanged auxiliary files together.

        Args:
            directory: Dedicated application directory receiving the complete Skill.
            overwrite: Replace a nonempty directory and remove stale files; false rejects it.

        Returns:
            Absolute directory containing SKILL.md and its reference files.
        """
        from .io import export_skill

        return export_skill(self._bundle, self._resources, directory, overwrite=overwrite)

    def with_bundle(self, bundle: SkillBundle) -> Self:
        """Create a candidate with new content, retaining borrowed dependencies.

        Args:
            bundle: Replacement source content. The original Skill is unchanged.
        """
        candidate = copy(self)
        candidate._bundle = SkillBundle.deserialize(bundle.serialize())
        return candidate

    def assemble(self, task: "Task") -> Self:
        """Return a complete task-local Skill, with unchanged content by default.

        Args:
            task: Public task input only, without private scoring data.

        Returns:
            A Skill copy preserving its concrete class, resources, and dependencies. Overrides may replace only its bundle.
        """
        return self.with_bundle(self.bundle)

    def snapshot(self) -> SkillSnapshot:
        """Capture content while retaining opaque metadata from existing snapshots."""
        metadata = self._snapshot_template
        snapshot = SkillSnapshot(
            implementation=metadata.implementation if metadata is not None else "static",
            config_schema_version=metadata.config_schema_version if metadata is not None else 1,
            config=metadata.model_copy(deep=True).config if metadata is not None else {},
            bundle=self._bundle.serialize(),
            content_hash=self._bundle.content_hash,
            resources=self._resources.serialize(),
        )
        snapshot.local_snapshot_hash = snapshot.compute_local_hash()
        return snapshot

    @classmethod
    def from_snapshot(cls, snapshot: SkillSnapshot) -> Self:
        """Restore data without resolving implementations or constructing backends.

        Args:
            snapshot: Source content and opaque implementation metadata. Existing
                metadata is retained for lossless storage round trips.
        """
        snapshot = SkillSnapshot.model_validate(snapshot.model_dump())
        skill = cls(
            bundle=SkillBundle.deserialize(snapshot.bundle),
            resources=SkillResources.deserialize(snapshot.resources),
        )
        skill._snapshot_template = snapshot.model_copy(deep=True)
        return skill
