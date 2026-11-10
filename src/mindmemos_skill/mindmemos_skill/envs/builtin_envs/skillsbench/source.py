"""Validate pinned downloaded tasks and bind actual source files to rollout recovery."""

from pathlib import Path

from ....common.input_files import file_fingerprints
from .typing import SkillsBenchRewardData


def task_identity(source: SkillsBenchRewardData) -> dict[str, str]:
    """Verify the exact source tree before allowing execution or cached-result reuse.

    Args:
        source: Private original package and checksum values captured by the loader.

    Returns:
        Actual absolute file paths mapped to their verified digests for base rollout recovery.

    Raises:
        ValueError: Files changed, paths are unsafe, or required task files are absent.
    """
    if source.task_dir.expanduser().is_symlink():
        raise ValueError("SkillsBench task directories must not be symbolic links")
    root = source.task_dir.expanduser().resolve()
    files = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("SkillsBench source files must not be symbolic links")
        if path.is_file():
            files[path.relative_to(root).as_posix()] = path
    if set(files) != set(source.checksums):
        raise ValueError("SkillsBench task files differ from the downloaded snapshot")
    required = {"instruction.md", "task.toml", "environment/Dockerfile", "tests/test.sh"}
    if not required <= files.keys():
        raise ValueError("SkillsBench requires a Harbor-format task package")
    identity = file_fingerprints(files.values())
    if any(identity[str(path)] != source.checksums[name] for name, path in files.items()):
        raise ValueError("SkillsBench task content changed since download")
    return identity


def regular_file(path: Path, root: Path) -> Path:
    """Resolve a native result file without following agent-created links.

    Args:
        path: Expected native artifact path.
        root: Private native output root containing that file.

    Returns:
        Validated original path.

    Raises:
        ValueError: The file escapes its root or includes a symbolic link.
        FileNotFoundError: The expected artifact is absent.
    """
    if root.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Harbor artifact escapes the private output directory")
    for part in (path, *path.parents):
        if part == root:
            break
        if part.is_symlink():
            raise ValueError("Harbor result paths must not contain symbolic links")
    if not path.is_file():
        raise FileNotFoundError(f"Harbor did not produce {path.name}")
    return path
