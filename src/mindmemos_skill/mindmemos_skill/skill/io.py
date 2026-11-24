"""Read and atomically export complete Skill source files."""

from pathlib import Path

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
        directory: Parent directory for the named Skill child.
        overwrite: Replace a nonempty destination when true.

    Returns:
        Absolute path of the exported Skill child directory.

    Raises:
        ValueError: Destination is unsafe or already populated without overwrite.
    """
    from mindmemos_sdk.skills.files import export_skill_files

    source = SkillContent.model_validate(source.model_dump())
    return export_skill_files(
        source.content,
        source.resources,
        directory,
        name=source.name,
        standard_format=source.standard_format,
        overwrite=overwrite,
    )
