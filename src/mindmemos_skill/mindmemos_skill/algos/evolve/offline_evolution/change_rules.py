"""Scope and resource checks for cited skill edits, independent of issue labels and code similarity."""

from collections.abc import Mapping
from pathlib import Path, PurePosixPath

from .provenance import quoted
from .typing import Change, FindingRef, RunLayout


def edit_problems(change: Change, layout: RunLayout, accepted: Mapping[str, FindingRef], diff: set[str]) -> list[str]:
    """Verify quotes, skill attribution and bundled resources without dictating an edit's form.

    Existing skills need a cited finding concerning them; new skills need an uncovered finding.
    Descriptions, instructions and code may all be adapted. Their semantic justification is the
    agent's responsibility, not inferred from issue labels, code-line matches or comparison groups.

    Args:
        change: Declared file changes and their supporting references.
        layout: Run containing original and edited Skill resources.
        accepted: Host-verified findings indexed by collection identity.
        diff: Actual changed paths relative to the Skill collection.

    Returns:
        Source-attribution and resource problems that reject the declared edits.
    """
    baseline, edited = layout.input_skills, layout.edited_skills
    cited = [accepted[identity] for identity in change.findings if identity in accepted]
    problems = [
        problem for quote in change.skill_quotes for problem in _quote_problems(quote.path, quote.quote, baseline)
    ]
    files = sorted(path for path in change.files if _safe(path) and path in diff)
    for skill in sorted({PurePosixPath(path).parts[0] for path in files}):
        paths = [path for path in files if PurePosixPath(path).parts[0] == skill]
        if (baseline / skill).is_dir():
            if not (edited / skill / "SKILL.md").is_file():
                problems.append(f"deleting the skill {skill} is not allowed")
            if not any(finding.skill == skill for finding in cited):
                problems.append(f"edits {skill} but cites no finding concerning it")
        elif not any(finding.issue == "uncovered" for finding in cited):
            problems.append(f"adds the skill {skill} without citing an uncovered finding")
        problems += _file_problems(skill, paths, baseline, edited)
    return problems


def _file_problems(skill: str, paths: list[str], baseline: Path, edited: Path) -> list[str]:
    """New resources must be discoverable; binary files must copy original resources."""
    problems = []
    originals: set[bytes] | None = None
    for path in paths:
        new, old = edited / path, baseline / path
        if not new.is_file() or new.name == "SKILL.md":
            continue
        guides = (guide for guide in (edited / skill).rglob("*.md") if guide.is_file() and guide != new)
        if not old.is_file() and not any(
            new.name in guide.read_text(encoding="utf-8", errors="replace") for guide in guides
        ):
            problems.append(f"{path} is new but no Markdown file of {skill} names it where a step uses it")
        if new.suffix == ".md":
            continue
        data = new.read_bytes()
        if b"\0" in data[:4096]:
            if originals is None:
                originals = {original.read_bytes() for original in baseline.rglob("*") if original.is_file()}
            if data not in originals:
                problems.append(f"{path} is binary and does not copy a file of the original skills")
    return problems


def _quote_problems(path: str, quote: str, baseline: Path) -> list[str]:
    """Verify an original-skill passage claimed as a source."""
    if not _safe(path) or not (baseline / path).is_file():
        return [f"skill quote {path!r} is not a file of the input skills"]
    if not quoted(quote, (baseline / path).read_bytes()):
        return [f"skill quote not found in {path}: {quote[:80]!r}"]
    return []


def _safe(path: str) -> bool:
    """Whether a path is relative and stays inside the skill collection."""
    pure = PurePosixPath(path)
    return bool(pure.parts) and not pure.is_absolute() and ".." not in pure.parts
