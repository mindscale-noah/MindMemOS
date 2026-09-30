"""Repository snapshot conversion and version comparison."""

from difflib import unified_diff

from ..persistence import SkillVersionRecord
from ..skill import SkillBundle, SkillResources, SkillSnapshot
from .typing import DiffResult, FileChange


def snapshot_from_version(version: SkillVersionRecord) -> SkillSnapshot:
    """Validate stored content before exporting or comparing it.

    Args:
        version: Stored source version, including implementation metadata.
    """
    return SkillSnapshot(
        implementation=version.implementation,
        config_schema_version=version.config_schema_version,
        config=version.config,
        bundle=version.bundle,
        content_hash=version.content_hash,
        resources=version.resources,
        local_snapshot_hash=version.local_snapshot_hash,
    )


def compare_versions(before: SkillVersionRecord, after: SkillVersionRecord) -> DiffResult:
    """Compare managed text and resources, including binary additions and removals.

    Args:
        before: Starting version.
        after: Target version.
    """
    left, right = _files(snapshot_from_version(before)), _files(snapshot_from_version(after))
    changes = []
    for path in sorted(left.keys() | right.keys()):
        if left.get(path) == right.get(path):
            continue
        old, new = left.get(path, b""), right.get(path, b"")
        binary = b"\0" in old or b"\0" in new
        diff = ""
        try:
            old_text, new_text = old.decode("utf-8"), new.decode("utf-8")
        except UnicodeDecodeError:
            binary = True
        if not binary:
            diff = "".join(
                unified_diff(
                    old_text.splitlines(keepends=True),
                    new_text.splitlines(keepends=True),
                    fromfile=f"{before.version_label or before.version_id}/{path}",
                    tofile=f"{after.version_label or after.version_id}/{path}",
                )
            )
        changes.append(
            FileChange(
                path=path,
                change="added" if path not in left else "deleted" if path not in right else "modified",
                binary=binary,
                diff=diff,
            )
        )
    return DiffResult(
        from_version_id=before.version_id,
        to_version_id=after.version_id,
        files=changes,
        implementation_changed=(before.implementation, before.config_schema_version, before.config)
        != (after.implementation, after.config_schema_version, after.config),
    )


def _files(snapshot):
    """Return complete source files without constructing a runtime Skill."""
    files = SkillResources.deserialize(snapshot.resources).files
    files.update({name: text.encode("utf-8") for name, text in SkillBundle.deserialize(snapshot.bundle).files.items()})
    return files
