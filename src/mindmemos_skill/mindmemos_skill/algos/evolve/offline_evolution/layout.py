"""Input directory discovery and atomic text writes."""

from pathlib import Path


def skill_names(skills: Path) -> set[str]:
    """Names that a skill source has after `RunLayout.snapshot`.

    Args:
        skills: Directory of skill directories, or one skill directory containing SKILL.md.

    Returns:
        The source's own name for a single skill, otherwise its subdirectories holding SKILL.md.
    """
    if (skills / "SKILL.md").is_file():
        return {skills.name}
    return {child.name for child in skills.iterdir() if (child / "SKILL.md").is_file()} if skills.is_dir() else set()


def write_text(path: Path, text: str) -> None:
    """Replace a text file atomically through a temporary sibling.

    Args:
        path: Destination file whose parent directories are created if missing.
        text: Complete UTF-8 text to commit.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
