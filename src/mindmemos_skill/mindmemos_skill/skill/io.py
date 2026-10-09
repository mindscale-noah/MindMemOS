"""Read and atomically export complete Skill source files."""

import os
import shutil
from pathlib import Path
from tempfile import mkdtemp
from uuid import uuid4

from .typing import SkillContent


def read_skill(path: str | Path) -> SkillContent:
    """Read a standard local Skill without changing its source bytes.

    Args:
        path: Skill directory or its SKILL.md file.

    Returns:
        Validated content and every auxiliary file.

    Raises:
        ValueError: The entrypoint or any resource is a symbolic link.
        OSError: The source cannot be read.
    """
    source = Path(path).expanduser()
    if source.is_symlink():
        raise ValueError("Skill source must not be a symbolic link")
    entrypoint = source / "SKILL.md" if source.is_dir() else source
    if entrypoint.name != "SKILL.md":
        raise ValueError("Expected a SKILL.md file or its containing directory")
    if entrypoint.is_symlink():
        raise ValueError("SKILL.md must not be a symbolic link")
    content = entrypoint.read_bytes().decode("utf-8")
    root = entrypoint.parent
    resources = {}
    for resource in sorted(root.rglob("*")):
        if resource.is_symlink():
            raise ValueError("Skill resources must not be symbolic links")
        if resource.is_file() and resource != entrypoint:
            resources[resource.relative_to(root).as_posix()] = resource.read_bytes()
    return SkillContent(content=content, resources=resources)


def export_skill(source: SkillContent, directory: str | Path, *, overwrite: bool = False) -> Path:
    """Atomically export the unchanged document and auxiliary bytes.

    Args:
        source: Validated Skill source to export.
        directory: Dedicated destination Skill directory.
        overwrite: Replace a nonempty destination when true.

    Returns:
        Absolute destination directory.

    Raises:
        ValueError: Destination is unsafe or already populated without overwrite.
    """
    source = SkillContent.model_validate(source.model_dump())
    raw = Path(directory).expanduser().absolute()
    if raw.is_symlink():
        raise ValueError("Export destination must not be a symbolic link")
    target = raw.resolve()
    if target == Path.home() or target == Path.cwd() or target in Path.cwd().parents or target.parent == target:
        raise ValueError("Export destination must be a dedicated Skill directory")
    if target.exists() and (not target.is_dir() or (not overwrite and any(target.iterdir()))):
        raise ValueError("Export destination exists and is not empty")
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(mkdtemp(prefix=f".{target.name}-export-", dir=target.parent))
    backup = target.parent / f".{target.name}-backup-{uuid4().hex}"
    try:
        files = {"SKILL.md": source.content.encode("utf-8"), **source.resources}
        for name, content in files.items():
            destination = stage / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
        if target.exists():
            if not overwrite and any(target.iterdir()):
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
