"""Read and atomically replace managed Skill files."""

import os
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile, mkdtemp
from uuid import uuid4

from ..errors import SkillConflictError
from .typing import SkillBundle, SkillResources


def read_bundle(path: str | Path) -> dict[str, str]:
    """Read only the managed Skill entrypoint.

    Args:
        path: A SKILL.md file or its containing directory.

    Returns:
        The managed file mapping.
    """
    source = Path(path).expanduser()
    if source.is_dir():
        source = source / "SKILL.md"
    if source.name != "SKILL.md":
        raise ValueError("Expected a SKILL.md file or its containing directory")
    if source.is_symlink():
        raise ValueError("SKILL.md must not be a symbolic link")
    return {"SKILL.md": source.read_text(encoding="utf-8")}


def write_bundle(files: dict[str, str], directory: str | Path) -> Path:
    """Replace managed files without following file symlinks or deleting resources.

    Args:
        files: Validated bundle file mapping.
        directory: Destination directory, created if needed.

    Returns:
        Absolute destination directory.
    """
    files = SkillBundle(files=files).files
    return _write_files({name: content.encode("utf-8") for name, content in files.items()}, directory)


def read_resources(path: str | Path) -> SkillResources:
    """Snapshot auxiliary files without editing or interpreting their contents.

    Args:
        path: Skill directory or its SKILL.md file.
    """
    root = Path(path).expanduser()
    if root.is_file():
        root = root.parent
    files = {}
    for source in sorted(root.rglob("*")):
        if source.is_symlink():
            raise ValueError("Skill resource files and directories must not be symbolic links")
        if source.is_file() and source != root / "SKILL.md":
            files[source.relative_to(root).as_posix()] = source.read_bytes()
    return SkillResources(files=files)


def write_skill(bundle: SkillBundle, resources: SkillResources, directory: str | Path) -> Path:
    """Export bundle and auxiliary bytes with their original relative layout.

    Args:
        bundle: Current task-time or source content.
        resources: Unchanged auxiliary files belonging to the Skill.
        directory: Destination Skill directory.
    """
    files = SkillResources(files=resources.files).files
    files.update({name: content.encode("utf-8") for name, content in SkillBundle(files=bundle.files).files.items()})
    return _write_files(files, directory)


def export_skill(bundle: SkillBundle, resources: SkillResources, directory: str | Path, *, overwrite: bool) -> Path:
    """Stage a complete export and restore the old directory if replacement fails.

    Args:
        bundle: Complete managed source content.
        resources: Complete auxiliary files to include.
        directory: Destination application directory.
        overwrite: Allow replacing a nonempty directory and removing its stale files.

    Returns:
        Absolute export destination.
    """
    raw = Path(directory).expanduser().absolute()
    if raw.is_symlink():
        raise ValueError("Export destination must not be a symbolic link")
    target = raw.resolve()
    if target == Path.home() or target == Path.cwd() or target in Path.cwd().parents or target.parent == target:
        raise ValueError("Export destination must be a dedicated application Skill directory")
    if target.exists() and (not target.is_dir() or (not overwrite and any(target.iterdir()))):
        raise SkillConflictError("Destination exists and is not an empty directory; explicitly enable overwrite")
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(mkdtemp(prefix=f".{target.name}-export-", dir=target.parent))
    backup = target.parent / f".{target.name}-backup-{uuid4().hex}"
    try:
        write_skill(bundle, resources, stage)
        if target.exists():
            if not overwrite and any(target.iterdir()):
                raise SkillConflictError("Destination changed during export")
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


def _write_files(files: dict[str, bytes], directory: str | Path) -> Path:
    """Replace each declared file atomically without following target symlinks."""
    root = Path(directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    for name in files:
        target = root / name
        if target.is_symlink() or any(
            parent.is_symlink() for parent in target.parents if parent != root and root in parent.parents
        ):
            raise ValueError("Skill destination files and directories must not be symbolic links")
    for name, content in files.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with NamedTemporaryFile(mode="wb", dir=target.parent, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(content)
            os.replace(temporary, target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return root
