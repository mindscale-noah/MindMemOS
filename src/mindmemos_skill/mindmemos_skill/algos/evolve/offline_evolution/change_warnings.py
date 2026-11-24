"""Warnings of the change gate: accepted edits that may copy one run, tie guidance to an environment,
or bloat or break a skill.

Warnings never revert an edit. Actionable warnings are returned to the change agent, which revises
the edit or acknowledges the warning with a reason; the description warning only informs, since this
gate does not test routing. Copying is checked on Markdown prose; code adaptations are not judged
by textual similarity. Checks compare characters and file structure, not words or syntax, so they
work for any natural language, programming language and trajectory format.
"""

import difflib
import json
import re
from collections.abc import Callable, Iterator
from pathlib import Path, PurePosixPath

from .skill_check import description_of
from .typing import GateWarning, RunLayout, WarningKind

COPY_CHARS = 40
"""Length of a character run shared with the trajectories that marks added text as copied from a run."""

COPY_STEP = 10
"""Distance between the character runs compared."""

COPY_MIN_ALNUM = 20
"""Letters or digits a run must contain, so punctuation and markup do not count as copied text."""

COPY_EXAMPLES = 3
"""Copied passages quoted per file."""

VERSION_CONDITION = re.compile(r"(?<![\d.])\d+\.\d+\.(?:\d+|x)(?!\d|\.\d)|(?:[<>]=?|==|~=|≥|≤)\s*v?\d+\.\d+(?!\d|\.\d)")
"""Three-part version numbers and version comparisons, which tie guidance to one environment."""

FENCE = re.compile(r"^\s*(```|~~~)")
"""Opening or closing line of a Markdown code block."""

LONG_SKILL_LINES = 500
"""SKILL.md length above which detail belongs in reference files linked from it."""

LARGE_REWRITE_SHARE = 0.5
"""Share of a file's lines removed above which an edit counts as a rewrite."""

LARGE_REWRITE_MIN_LINES = 10
"""Files shorter than this are not checked for rewrites."""

AddWarning = Callable[[WarningKind, str, str, str], None]
"""Records a warning from kind, file, subject and message."""


def collect_warnings(
    layout: RunLayout, files: list[str], corpus: "TrajectoryCorpus | None" = None
) -> list[GateWarning]:
    """Check the accepted files of the candidate against the baseline skills and the trajectories.

    Args:
        layout: Run whose baseline, candidate skills and trajectories are compared.
        files: Accepted skill-relative files.
        corpus: Trajectory text already loaded by the caller; loaded here when None.

    Returns:
        One warning per distinct key, in file order.
    """
    found: dict[str, GateWarning] = {}

    def add(kind: WarningKind, file: str, subject: str, message: str) -> None:
        """Retain one warning for each kind, file and subject."""
        key = f"{kind}:{file}:{subject}"
        found.setdefault(key, GateWarning(key=key, kind=kind, file=file, message=message))

    baseline, candidate = layout.input_skills, layout.candidate_skills
    corpus = corpus or TrajectoryCorpus(layout)
    for skill in sorted({PurePosixPath(p).parts[0] for p in files}):
        for resource, problem in _orphaned_resources(baseline / skill, candidate / skill):
            add("orphaned_resource", f"{skill}/{resource}", resource, f"{skill}/{resource}: {problem}")
        description = description_of(candidate / skill) if (candidate / skill).is_dir() else ""
        if (baseline / skill).is_dir() and description and description != description_of(baseline / skill):
            add(
                "description_changed",
                f"{skill}/SKILL.md",
                "description",
                f"{skill}: description changed; routing not tested",
            )
    for path in files:
        new, old = candidate / path, baseline / path
        new_lines, old_lines = _lines(new), _lines(old)
        if new_lines is None:
            continue
        copied = corpus.copied(" ".join(_prose(new, set(old_lines or []))))
        if copied:
            examples = "; ".join(f"“{c}”" for c in copied)
            add(
                "copied_from_trajectories",
                path,
                "text",
                f"{path}: adds prose found in the trajectories but not in the skills ({examples}); "
                "a run's wording, values or answers do not generalize",
            )
        for line in _prose(new, set(old_lines or [])):
            for version in VERSION_CONDITION.finditer(line):
                add(
                    "version_condition",
                    path,
                    version.group(0),
                    f"{path}: adds version condition “{version.group(0)}”, which ties guidance to one environment",
                )
        if new.name == "SKILL.md" and len(new_lines) > LONG_SKILL_LINES >= len(old_lines or []):
            add(
                "long_skill",
                path,
                "length",
                f"{path}: grows past {LONG_SKILL_LINES} lines; move detail to reference files linked from SKILL.md",
            )
        if old_lines and len(old_lines) >= LARGE_REWRITE_MIN_LINES:
            kept = sum(
                block.size for block in difflib.SequenceMatcher(None, old_lines, new_lines).get_matching_blocks()
            )
            removed_share = 1 - kept / len(old_lines)
            if removed_share > LARGE_REWRITE_SHARE:
                add(
                    "large_rewrite",
                    path,
                    "rewrite",
                    f"{path}: rewrites {removed_share:.0%} of its lines; keep edits to what the evidence supports",
                )
    return list(found.values())


def warning_feedback(warnings: list[GateWarning]) -> str:
    """Instructions for the change agent listing actionable warnings it has not answered.

    Args:
        warnings: Warnings of the latest gate run, with acknowledgements applied.

    Returns:
        Feedback text, or an empty string when nothing needs an answer.
    """
    pending = [w for w in warnings if w.actionable and not w.acknowledged]
    if not pending:
        return ""
    lines = [
        "The change gate accepted your changes but raised warnings. For each one, either revise the edit "
        "(keeping changes.json in sync), or keep it and add "
        '`{"warning": "<key>", "reason": "..."}` to an `acknowledged` list in changes.json.',
        "",
    ]
    return "\n".join(lines + [f"- `{w.key}`: {w.message}" for w in pending])


def _prose(path: Path, old_lines: set[str]) -> Iterator[str]:
    """Added Markdown lines outside code blocks; other files hold no prose."""
    if path.suffix != ".md":
        return
    in_block = False
    for line in _lines(path) or []:
        if FENCE.match(line):
            in_block = not in_block
        elif not in_block and line not in old_lines:
            yield line


def _orphaned_resources(baseline: Path, candidate: Path) -> Iterator[tuple[str, str]]:
    """Files of a skill that its original instructions referenced and the edited ones no longer do.

    Dropping such a file from the instructions removes a capability the skill provided, which
    needs evidence that the file itself caused problems.
    """
    if not baseline.is_dir():
        return
    for path in sorted(p for p in baseline.rglob("*") if p.is_file() and p.name != "SKILL.md"):
        resource = path.relative_to(baseline).as_posix()
        if not _referenced(baseline, resource):
            continue
        if not (candidate / resource).exists():
            yield resource, "removed, although the original instructions used it"
        elif not _referenced(candidate, resource):
            yield resource, "kept, but the instructions no longer refer to it as the original ones did"


def _referenced(skill: Path, resource: str) -> bool:
    """Whether the Markdown files of a skill, other than the resource itself, name it by path, file name or stem."""
    pure = PurePosixPath(resource)
    names = {resource, pure.name} | ({pure.stem} if pure.suffix != ".md" else set())
    pattern = re.compile("|".join(rf"(?<![\w.-]){re.escape(name)}(?![\w-])" for name in sorted(names)))
    documents = (p for p in skill.rglob("*.md") if p.is_file() and p.relative_to(skill).as_posix() != resource)
    return any(pattern.search(p.read_text(encoding="utf-8", errors="replace")) for p in documents)


def _lines(path: Path) -> list[str] | None:
    """Stripped non-empty lines of a text file; an empty list when it does not exist, None when it is binary."""
    if not path.is_file():
        return []
    data = path.read_bytes()
    if b"\0" in data[:4096]:
        return None
    return [line.strip() for line in data.decode("utf-8", errors="replace").splitlines() if line.strip()]


class TrajectoryCorpus:
    """Trajectory and baseline-skill text, loaded once, for finding text copied from the runs."""

    def __init__(self, layout: RunLayout) -> None:
        self.layout = layout
        self._trajectories: str | None = None
        self._skills: str | None = None

    def copied(self, text: str) -> list[str]:
        """Character runs of `text` that occur in the trajectories but in no baseline skill.

        Args:
            text: Added text, lines joined by spaces.

        Returns:
            Up to COPY_EXAMPLES non-overlapping runs.
        """
        text = re.sub(r"\s+", " ", text)
        found: list[str] = []
        start, last = 0, len(text) - COPY_CHARS
        while 0 <= start <= last and len(found) < COPY_EXAMPLES:
            window = text[start : start + COPY_CHARS]
            if (
                sum(c.isalnum() for c in window) >= COPY_MIN_ALNUM
                and self._in_runs(window)
                and not self._in_skills(window)
            ):
                found.append(window.strip())
                start += COPY_CHARS
            elif start < last:
                start = min(start + COPY_STEP, last)  # the final window ends with the text
            else:
                break
        return found

    def _in_runs(self, window: str) -> bool:
        if self._trajectories is None:
            files = sorted(p for p in self.layout.input_trajectories.rglob("*") if p.is_file())
            self._trajectories = "\n".join(p.read_bytes().decode("utf-8", errors="replace") for p in files)
        variants = {window, json.dumps(window, ensure_ascii=False)[1:-1], json.dumps(window)[1:-1]}
        return any(variant in self._trajectories for variant in variants)

    def _in_skills(self, window: str) -> bool:
        if self._skills is None:
            files = sorted(p for p in self.layout.input_skills.rglob("*") if p.is_file())
            self._skills = re.sub(
                r"\s+", " ", "\n".join(p.read_bytes().decode("utf-8", errors="replace") for p in files)
            )
        return window in self._skills
