"""Standard skill format checks applied to changed or new skills."""

import re
from pathlib import Path

import yaml

FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*(\n|\Z)", re.DOTALL)
"""Leading YAML block of SKILL.md."""

RESOURCE_REFERENCE = re.compile(r"(?<![\w/.-])(?:\./)?((?:scripts|references|assets)/[\w.\-/]*[\w])")
"""Bundled-resource paths mentioned in skill Markdown."""

UNSAFE_CODE = re.compile(
    r"(\bimport\s+(socket|requests|httpx|urllib3)\b|\bfrom\s+(socket|requests|httpx|urllib)\b|urllib\.request|"
    r"http\.client|shutil\.rmtree|\brm\s+-rf\b|\bcurl\s|\bwget\s)"
)
"""Network access and recursive deletion in common scripting languages; a safety net, not a complete list."""

DESCRIPTION_LIMIT = 1024
"""Maximum description length in characters accepted by skill loaders."""


def check_skill(skill: Path, baseline: Path | None) -> list[str]:
    """List format problems of a changed or new skill directory.

    Args:
        skill: Candidate skill directory.
        baseline: The same skill before changes, or None for a new skill. Resources and files
            unchanged from the baseline are not re-checked; binary files are skipped.

    Returns:
        Human-readable problems; empty when the skill passes.
    """
    main = skill / "SKILL.md"
    if not main.is_file():
        return [f"{skill.name}: SKILL.md missing"]
    meta = _frontmatter(main)
    if meta is None:
        return [f"{skill.name}: SKILL.md has no valid YAML frontmatter"]
    problems: list[str] = []
    name, description = str(meta.get("name") or ""), str(meta.get("description") or "")
    if not name or not description:
        problems.append(f"{skill.name}: frontmatter needs non-empty name and description")
    if len(description) > DESCRIPTION_LIMIT:
        problems.append(f"{skill.name}: description longer than {DESCRIPTION_LIMIT} characters")
    if "<" in description or ">" in description:
        problems.append(f"{skill.name}: description contains angle brackets")
    baseline_meta = _frontmatter(baseline / "SKILL.md") if baseline else None
    # Enforce name == directory only for new skills or skills that already followed the rule.
    if name != skill.name and (baseline is None or baseline_meta is None or baseline_meta.get("name") == baseline.name):
        problems.append(f"{skill.name}: frontmatter name {name!r} differs from directory name")
    known = _references(baseline) if baseline else set()
    for reference in sorted(_references(skill) - known):
        if not (skill / reference).exists():
            problems.append(f"{skill.name}: newly referenced {reference} does not exist")
    for script in sorted(p for p in skill.rglob("*") if p.is_file() and p.suffix != ".md"):
        relative = script.relative_to(skill).as_posix()
        previous = baseline / relative if baseline else None
        data = script.read_bytes()
        if (previous and previous.is_file() and previous.read_bytes() == data) or b"\0" in data[:4096]:
            continue
        unsafe = UNSAFE_CODE.search(data.decode("utf-8", errors="replace"))
        if unsafe:
            problems.append(f"{skill.name}/{relative}: unsafe operation {unsafe.group(0)!r}")
    return problems


def description_of(skill: Path) -> str:
    """Return a skill's frontmatter description.

    Args:
        skill: Skill directory.

    Returns:
        The description, or an empty string when SKILL.md or the field is missing.
    """
    meta = _frontmatter(skill / "SKILL.md")
    return str((meta or {}).get("description") or "")


def _frontmatter(main: Path) -> dict | None:
    """Parse SKILL.md frontmatter, or return None when the file or a valid YAML mapping is missing."""
    if not main.is_file():
        return None
    match = FRONTMATTER.match(main.read_text(encoding="utf-8", errors="replace"))
    if not match:
        return None
    try:
        meta = yaml.safe_load(match.group(1))
    except yaml.YAMLError:
        return None
    return meta if isinstance(meta, dict) else None


def _references(skill: Path) -> set[str]:
    """Collect scripts/, references/ and assets/ paths mentioned in a skill's Markdown files."""
    return {
        ref
        for path in skill.rglob("*.md")
        for ref in RESOURCE_REFERENCE.findall(path.read_text(encoding="utf-8", errors="replace"))
    }
