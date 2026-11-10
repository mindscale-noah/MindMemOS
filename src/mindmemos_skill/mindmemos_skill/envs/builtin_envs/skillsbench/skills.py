"""Prepare isolated Harbor task copies with explicit Skills instead of implicit bundled inputs."""

import shutil
from pathlib import Path

from ....skill import Skill
from ....skill.io import read_skill


def prepare_task(source: Path, destination: Path, skills: list[Skill], *, oracle: bool) -> list[Path]:
    """Copy a task and select the complete Skills visible to a native model agent.

    Args:
        source: Verified original task package.
        destination: Fresh private copy used by Harbor.
        skills: Already assembled complete source Skills; an empty list is a no-Skill run.
        oracle: Preserve the official package for the reference solution sanity check.

    Returns:
        Exported Skill directories for Harbor's native injection mechanism.

    Raises:
        ValueError: Oracle receives candidate Skills or exported names collide.
    """
    if oracle and skills:
        raise ValueError("Oracle sanity checks do not evaluate supplied Skills")
    shutil.copytree(source, destination)
    if oracle:
        return []
    directory = destination / "environment" / "skills"
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    names = [skill.resolve_export_directory(directory).name for skill in skills]
    if len(set(names)) != len(names):
        raise ValueError("SkillsBench input Skills have colliding export names")
    exported = [skill.export(directory) for skill in skills]
    for path in exported:
        read_skill(path)
    return exported
