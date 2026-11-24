"""Compare complete Skill snapshots without loading runtime implementations."""

from difflib import unified_diff

from .typing import FileChange, SkillSnapshot, SnapshotDiff


def compare_snapshots(before: SkillSnapshot, after: SkillSnapshot) -> SnapshotDiff:
    """Compare validated source files, metadata, and implementation settings.

    Args:
        before: Original complete source snapshot.
        after: Replacement complete source snapshot.

    Returns:
        File changes in path order and independent metadata/configuration flags.

    Raises:
        ValueError: Either source snapshot fails integrity validation.
    """
    left_source, right_source = before.to_content(), after.to_content()
    left = {"SKILL.md": left_source.content.encode("utf-8"), **left_source.resources}
    right = {"SKILL.md": right_source.content.encode("utf-8"), **right_source.resources}
    changes = []
    for path in sorted(left.keys() | right.keys()):
        if path in left and path in right and left[path] == right[path]:
            continue
        old, new = left.get(path, b""), right.get(path, b"")
        binary = b"\0" in old or b"\0" in new
        text = ""
        try:
            old_text, new_text = old.decode("utf-8"), new.decode("utf-8")
        except UnicodeDecodeError:
            binary = True
        if not binary:
            text = "".join(
                unified_diff(
                    old_text.splitlines(keepends=True),
                    new_text.splitlines(keepends=True),
                    fromfile=f"before/{path}",
                    tofile=f"after/{path}",
                )
            )
        changes.append(
            FileChange(
                path=path,
                change="added" if path not in left else "deleted" if path not in right else "modified",
                binary=binary,
                diff=text,
            )
        )
    return SnapshotDiff(
        files=changes,
        metadata_changed=(before.name, before.description, before.standard_format)
        != (after.name, after.description, after.standard_format),
        implementation_changed=(before.implementation, before.config_schema_version, before.config)
        != (after.implementation, after.config_schema_version, after.config),
    )
