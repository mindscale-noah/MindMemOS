"""Local source collection and transactional Skill exports."""

import fnmatch
import os
import shutil
import tempfile
from pathlib import Path

from .content import decode_resources, encode_resources, parse_frontmatter


def _reject_symlinks(path: Path) -> None:
    """Reject symlinks at the path or any of its parent components."""
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError(f"Symlinks are not supported: {path}")


def read_source(source: str) -> tuple[Path, str, dict]:
    """Read an exact standard source document and its frontmatter.

    Args:
        source: Directory containing SKILL.md, or the document itself.

    Returns:
        Root directory, original text, and validated frontmatter.
    """
    path = Path(source).absolute()
    _reject_symlinks(path)
    document = path / "SKILL.md" if path.is_dir() else path
    if document.name != "SKILL.md" or not document.is_file():
        raise ValueError("Source must be a directory containing SKILL.md or a SKILL.md file")
    _reject_symlinks(document)
    content = document.read_bytes().decode("utf-8")
    return document.parent, content, parse_frontmatter(content)


def _patterns(value: str | None) -> list[str]:
    """Parse optional comma-separated patterns and reject empty entries."""
    if value is None:
        return []
    patterns = [item.strip() for item in value.split(",")]
    if not all(patterns):
        raise ValueError("Include/exclude patterns cannot be empty")
    return patterns


def _matches(name: str, patterns: list[str]) -> bool:
    """Match a path or any parent directory, including root-level ** globs."""
    candidates = [name, *(parent.as_posix() for parent in Path(name).parents if str(parent) != ".")]
    for pattern in patterns:
        variants = [pattern.rstrip("/")]
        while variants[-1].startswith("**/"):
            variants.append(variants[-1][3:])
        if any(fnmatch.fnmatchcase(candidate, variant) for candidate in candidates for variant in variants):
            return True
    return False


def collect_resources(root: Path, include: str | None = None, exclude: str | None = None) -> dict[str, str]:
    """Collect selected regular files as portable Base64 resources.

    Args:
        root: Source root for relative paths.
        include: Optional comma-separated inclusion globs.
        exclude: Optional exclusion globs, applied after inclusion.

    Returns:
        Encoded resource mapping; SKILL.md is always excluded.
    """
    root = root.absolute()
    _reject_symlinks(root)
    if not root.is_dir():
        raise ValueError(f"Resource directory does not exist: {root}")
    included, excluded = _patterns(include), _patterns(exclude)
    resources = {}
    for directory, directories, files in os.walk(root, followlinks=False):
        for name in list(directories):
            path = Path(directory) / name
            relative = path.relative_to(root).as_posix()
            if _matches(relative, excluded):
                directories.remove(name)
            elif path.is_symlink():
                raise ValueError(f"Symlink resource directory is not supported: {relative}")
        for name in files:
            path = Path(directory) / name
            relative = path.relative_to(root).as_posix()
            if (
                relative == "SKILL.md"
                or (included and not _matches(relative, included))
                or _matches(relative, excluded)
            ):
                continue
            _reject_symlinks(path)
            if not path.is_file():
                raise ValueError(f"Resource must be a regular file: {relative}")
            resources[relative] = path.read_bytes()
    return encode_resources(resources)


def export_source(version: dict, destination: str, overwrite: bool) -> dict:
    """Write a version in a staging directory before replacing the destination.

    Args:
        version: HTTP version data with content and encoded resources.
        destination: Local export directory.
        overwrite: Whether an existing nonempty directory may be replaced.

    Returns:
        Destination and exported version identity.
    """
    target = Path(destination).absolute()
    _reject_symlinks(target)
    if target.exists() and (not target.is_dir() or (any(target.iterdir()) and not overwrite)):
        raise ValueError("Export destination must be an empty directory, or use --overwrite")
    resources = decode_resources(version["resources"])
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".skill-export-", dir=target.parent))
    backup = None
    try:
        (staging / "SKILL.md").write_bytes(version["content"].encode("utf-8"))
        for name, raw in resources.items():
            path = staging / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        if target.exists():
            backup_path = Path(tempfile.mkdtemp(prefix=".skill-backup-", dir=target.parent))
            backup_path.rmdir()
            target.rename(backup_path)
            backup = backup_path
        try:
            staging.rename(target)
        except OSError:
            if backup is not None:
                backup.rename(target)
                backup = None
            raise
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        if backup is not None:
            shutil.rmtree(backup)
    return {"directory": str(target), "skill_version_id": version["skill_version_id"], "resources": len(resources)}
