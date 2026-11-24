"""Shared Skill version snapshot conversion and content comparison."""

from ..persistence import SkillVersionRecord
from ..skill import SkillSnapshot, compare_snapshots
from ..skill.typing import DiffResult


def snapshot_from_version(version: SkillVersionRecord) -> SkillSnapshot:
    """Validate stored content before exporting or comparing it.

    Args:
        version: Stored source version, including implementation metadata.
    """
    return SkillSnapshot(
        implementation=version.implementation,
        config_schema_version=version.config_schema_version,
        config=version.config,
        name=version.name,
        description=version.description,
        standard_format=version.standard_format,
        content=version.content,
        content_hash=version.content_hash,
        resources=version.resources,
        skill_hash=version.skill_hash,
    )


def compare_versions(before: SkillVersionRecord, after: SkillVersionRecord) -> DiffResult:
    """Compare managed text and resources, including binary additions and removals.

    Args:
        before: Starting version.
        after: Target version.
    """
    comparison = compare_snapshots(snapshot_from_version(before), snapshot_from_version(after))
    return DiffResult(
        from_version_id=before.version_id,
        from_skill_id=before.skill_id,
        to_version_id=after.version_id,
        to_skill_id=after.skill_id,
        files=comparison.files,
        implementation_changed=comparison.implementation_changed,
    )
