"""Shared Skill export paths and atomic source-file publishing."""

import os
import re
import shutil
from pathlib import Path
from tempfile import mkdtemp
from uuid import uuid4

from .content import parse_frontmatter, validate_resource_paths


def resolve_skill_export_directory(
    content: str,
    directory: str | Path,
    *,
    name: str | None = None,
    standard_format: bool = True,
) -> Path:
    """Resolve the named Skill child without creating files.

    Args:
        content: Complete SKILL.md text; standard frontmatter owns the name.
        directory: Parent directory under which to create the Skill folder.
        name: Required explicit name for nonstandard content.
        standard_format: Whether to read the name from standard frontmatter.

    Returns:
        Absolute path of the named Skill child directory.

    Raises:
        ValueError: The name is unsafe or the destination uses symbolic links.
    """
    if standard_format:
        name = parse_frontmatter(content)["name"]
    if (
        not isinstance(name, str)
        or not name.strip()
        or name != name.strip()
        or name in {".", ".."}
        or any(character in name for character in '/\\<>:"|?*')
        or any(ord(character) < 32 for character in name)
        or name.endswith(".")
        or re.fullmatch(r"(?:CON|PRN|AUX|NUL|COM[1-9¹²³]|LPT[1-9¹²³])(?:\..*)?", name, re.IGNORECASE)
    ):
        raise ValueError("Skill name must be a valid single directory name")
    raw = Path(directory).expanduser().absolute() / name
    if any(path.is_symlink() for path in (raw, *raw.parents)):
        raise ValueError("Export destination must not use symbolic links")
    target = raw.resolve()
    if target == Path.home().resolve() or target == Path.cwd() or target in Path.cwd().parents:
        raise ValueError("Export destination must be a dedicated Skill directory")
    return target


def export_skill_files(
    content: str,
    resources: dict[str, bytes],
    directory: str | Path,
    *,
    name: str | None = None,
    standard_format: bool = True,
    overwrite: bool = False,
) -> Path:
    """Atomically write complete source files under the Skill's name.

    Args:
        content: Original SKILL.md text, including its newline characters.
        resources: Auxiliary relative file paths and original bytes.
        directory: Parent directory for the named Skill child.
        name: Explicit name for nonstandard content.
        standard_format: Whether standard frontmatter supplies the name.
        overwrite: Replace only the named child, including stale files.

    Returns:
        Absolute path of the exported Skill directory.

    Raises:
        ValueError: Source paths are unsafe or the child is already populated.
        OSError: Files cannot be written or published.
    """
    validate_resource_paths(resources)
    target = resolve_skill_export_directory(content, directory, name=name, standard_format=standard_format)
    if target.exists() and (not target.is_dir() or (not overwrite and any(target.iterdir()))):
        raise ValueError("Export destination exists and is not empty; use overwrite to replace the Skill directory")
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(mkdtemp(prefix=f".{target.name}-export-", dir=target.parent))
    backup = target.parent / f".{target.name}-backup-{uuid4().hex}"
    try:
        for relative, data in {"SKILL.md": content.encode("utf-8"), **resources}.items():
            destination = stage / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
        # Recheck links and destination occupancy immediately before replacement.
        resolve_skill_export_directory(content, directory, name=name, standard_format=standard_format)
        if target.exists():
            if not target.is_dir() or (not overwrite and any(target.iterdir())):
                raise ValueError("Export destination changed during export")
            os.replace(target, backup)
        try:
            os.replace(stage, target)
        except BaseException:
            if backup.exists():
                os.replace(backup, target)
            raise
        if backup.exists():
            shutil.rmtree(backup)
        return target
    finally:
        if stage.exists():
            shutil.rmtree(stage)
