"""Change gate: accept only skill edits that are claimed, cited and structurally valid.

Rules:
- changed files that no change claims are reverted;
- a change must cite accepted finding ids only and must change every file it claims;
- cited findings bound the skills a change may edit; descriptions, instructions and code may be
  adapted without issue-specific position rules or code-line matching (see change_rules);
- citations of changes claiming the same file jointly authorize its edits; changes sharing a file
  are accepted or rejected together, without lending authority to other files;
- a changed description may not repeat text of the trajectories that the skills do not contain,
  since a description restating one run's task does not describe the tasks a skill serves;
- a skill failing format checks after assembly reverts every change touching it, except for
  problems the baseline skill already had.
Every accepted finding with an issue must be cited by a change, directly or through its case, or listed
under `dropped`; unaddressed ones are reported but revert nothing.
Rejected changes, unclaimed files, unaddressed findings and warnings (see change_warnings) go back to
the change agent, which fixes, claims, answers or acknowledges them; warnings never revert an edit.
"""

import shutil
from collections.abc import Mapping
from pathlib import Path, PurePosixPath

from pydantic import ValidationError

from .change_rules import edit_problems
from .change_warnings import TrajectoryCorpus, collect_warnings
from .constants import CHANGES_FILE
from .layout import write_text
from .skill_check import check_skill, description_of
from .typing import Change, ChangeDecision, ChangeReport, ChangeSet, FindingRef, RunLayout


def load_changes(layout: RunLayout) -> ChangeSet:
    """Read and validate the change agent's claims.

    Args:
        layout: Run whose change workspace holds changes.json.

    Returns:
        The validated claims.

    Raises:
        ValueError: changes.json is missing or invalid; the message is sent back to the agent.
    """
    path = layout.change_workspace / CHANGES_FILE
    try:
        return ChangeSet.model_validate_json(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"{CHANGES_FILE} not found: {error}") from error
    except ValidationError as error:
        raise ValueError(f"{CHANGES_FILE} invalid: {str(error)[:600]}") from error


def changed_files(baseline: Path, edited: Path) -> set[str]:
    """List skill-relative paths added, removed or modified by their exact bytes.

    Args:
        baseline: Original skills directory.
        edited: Edited copy.

    Returns:
        Changed relative posix paths.
    """
    before, after = _files(baseline), _files(edited)
    changed = set(before) ^ set(after)
    for path in set(before) & set(after):
        if before[path].read_bytes() != after[path].read_bytes():
            changed.add(path)
    return changed


def run_change_gate(
    layout: RunLayout, accepted_findings: Mapping[str, FindingRef], editable: set[str], allow_create: bool = True
) -> ChangeReport:
    """Assemble candidate/skills from accepted changes and write candidate/gate.json and gate.md.

    Args:
        layout: Run whose edited skills and changes.json are checked.
        accepted_findings: Findings accepted by the evidence gate, by citable id.
        editable: Existing Skill names whose files may change.
        allow_create: Whether independent new Skills may be generated.

    Returns:
        The gate report; an empty `accepted_files` means no update.

    Raises:
        ValueError: changes.json is missing or invalid.
    """
    baseline, edited = layout.input_skills, layout.edited_skills
    existing = {path.name for path in baseline.iterdir() if path.is_dir()}
    changeset = load_changes(layout)
    diff = changed_files(baseline, edited)
    corpus = TrajectoryCorpus(layout)
    reasons = {
        change.id: _claim_problems(change, accepted_findings, diff, existing, editable, allow_create)
        + _description_problems(change, layout, corpus)
        for change in changeset.changes
    }
    for change in changeset.changes:
        reasons[change.id] += edit_problems(change.model_copy(update={"files": []}), layout, accepted_findings, diff)
    for path in sorted({path for change in changeset.changes for path in change.files}):
        owners = [change for change in changeset.changes if path in change.files]
        combined = owners[0].model_copy(
            update={
                "files": [path],
                "findings": sorted({finding for owner in owners for finding in owner.findings}),
                "skill_quotes": [],
            }
        )
        problems = edit_problems(combined, layout, accepted_findings, diff)
        for owner in owners:
            reasons[owner.id] += problems
    identities = [change.id for change in changeset.changes]
    for identity in {i for i in identities if identities.count(i) > 1}:
        reasons[identity].append("duplicate change id")
    groups = _groups(changeset.changes)
    skill_problems: dict[str, list[str]] = {}
    while True:
        accepted = [group for group in groups if all(not reasons[c.id] for c in group)]
        files = sorted({path for group in accepted for c in group for path in c.files} & diff)
        _assemble(baseline, edited, layout.candidate_skills, files)
        failing = _failing_skills(layout, files)
        if not failing:
            break
        skill_problems.update(failing)
        for group in accepted:
            for skill in {PurePosixPath(p).parts[0] for c in group for p in c.files} & set(failing):
                for change in group:
                    reasons[change.id].append(f"skill {skill} fails checks: " + "; ".join(failing[skill]))
    for group in groups:
        failed = [c.id for c in group if reasons[c.id]]
        for change in group:
            if failed and not reasons[change.id]:
                reasons[change.id].append(f"shares files with rejected change(s) {failed}")
    claimed = {path for change in changeset.changes for path in change.files}
    reasons_kept = {a.warning: a.reason for a in changeset.acknowledged}
    warnings = [
        w.model_copy(update={"acknowledged": reasons_kept.get(w.key)}) for w in collect_warnings(layout, files, corpus)
    ]
    report = ChangeReport(
        changes=[
            ChangeDecision(**c.model_dump(), status="rejected" if reasons[c.id] else "accepted", reasons=reasons[c.id])
            for c in changeset.changes
        ],
        dropped=changeset.dropped,
        reverted_unclaimed=sorted(diff - claimed),
        skill_problems=skill_problems,
        accepted_files=files,
        warnings=warnings,
        unaddressed_findings=_unaddressed(changeset, accepted_findings),
    )
    write_text(layout.candidate / "gate.json", report.model_dump_json(indent=1))
    write_text(layout.candidate / "gate.md", render_change_report(report))
    return report


def rejection_feedback(report: ChangeReport) -> str:
    """Instructions for the change agent listing rejected changes and why.

    Args:
        report: Gate result.

    Returns:
        Feedback text, or an empty string when no change was rejected for its own reasons.
    """
    own = [
        (change, [r for r in change.reasons if not r.startswith("shares files with rejected change")])
        for change in report.changes
        if change.status == "rejected"
    ]
    own = [(change, reasons) for change, reasons in own if reasons]
    if not own:
        return ""
    lines = [
        "The change gate rejected these changes, and changes sharing their files with them. "
        "Fix each one in place (keeping changes.json in sync), or undo it and move it to `dropped`.",
        "",
    ]
    return "\n".join(lines + [f"- {change.id}: {'; '.join(reasons)}" for change, reasons in own])


def open_items_feedback(report: ChangeReport) -> str:
    """Instructions for the change agent listing unclaimed files left out and unaddressed findings.

    Args:
        report: Gate result.

    Returns:
        Feedback text, or an empty string when nothing is open.
    """
    parts = []
    if report.reverted_unclaimed:
        parts.append(
            "These files were changed but no change claims them, so they were left out: "
            + ", ".join(f"`{path}`" for path in report.reverted_unclaimed)
            + ". Claim each in a change, or restore it."
        )
    if report.unaddressed_findings:
        parts.append(
            "These accepted findings are neither cited by a change nor listed under `dropped`: "
            + ", ".join(f"`{finding}`" for finding in report.unaddressed_findings)
            + ". Address each with a change, or add it to `dropped` with the reason it needs no edit."
        )
    return "\n\n".join(parts)


def render_change_report(report: ChangeReport) -> str:
    """Render the gate report as Markdown.

    Args:
        report: Gate result.

    Returns:
        Markdown text.
    """
    lines = ["# Change gate", "", f"Accepted files: {len(report.accepted_files)}", ""]
    for change in report.changes:
        quotes = ", ".join(q.path for q in change.skill_quotes)
        lines.append(
            f"- {change.id} [{change.status}] {', '.join(change.files)} ← findings {', '.join(change.findings)}"
            + (f"; skill quotes {quotes}" if quotes else "")
        )
        lines += [f"  - {reason}" for reason in change.reasons]
    if report.reverted_unclaimed:
        lines += ["", "Reverted (changed but not claimed): " + ", ".join(report.reverted_unclaimed)]
    if report.warnings:
        lines += ["", "Warnings:"]
        for warning in report.warnings:
            answer = f" — kept: {warning.acknowledged}" if warning.acknowledged else ""
            lines.append(f"- [{warning.kind}] {warning.message}{answer}")
    if report.unaddressed_findings:
        lines += ["", "Findings neither cited nor dropped: " + ", ".join(report.unaddressed_findings)]
    if report.dropped:
        lines += ["", "Investigated but not changed:"] + [
            f"- {d.reason} ({', '.join(d.findings)})" for d in report.dropped
        ]
    return "\n".join(lines) + "\n"


def _files(root: Path) -> dict[str, Path]:
    """Map relative posix paths to files under a root, skipping bytecode caches."""
    return {p.relative_to(root).as_posix(): p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts}


def _description_problems(change: Change, layout: RunLayout, corpus: TrajectoryCorpus) -> list[str]:
    """Reject a changed description that repeats trajectory text absent from the skills."""
    problems = []
    for path in change.files:
        pure = PurePosixPath(path)
        if pure.name != "SKILL.md" or len(pure.parts) != 2 or not (layout.edited_skills / path).is_file():
            continue
        skill = pure.parts[0]
        new = description_of(layout.edited_skills / skill)
        old = description_of(layout.input_skills / skill) if (layout.input_skills / skill).is_dir() else ""
        copied = corpus.copied(new) if new and new != old else []
        if copied:
            problems.append(
                f"{skill}: description repeats trajectory text ({'; '.join(copied)}); describe the kinds of tasks, "
                "inputs and stages the skill serves instead of a recorded task"
            )
    return problems


def _unaddressed(changeset: ChangeSet, accepted_findings: Mapping[str, FindingRef]) -> list[str]:
    """Accepted findings with an issue that no change cites, directly or through their case, and no
    `dropped` entry lists."""
    answered = {f for change in changeset.changes for f in change.findings}
    answered |= {f for dropped in changeset.dropped for f in dropped.findings}
    cases = {accepted_findings[f].case for f in answered if f in accepted_findings} - {None}
    return sorted(
        identity
        for identity, ref in accepted_findings.items()
        if ref.issue and identity not in answered and ref.case not in cases
    )


def _claim_problems(
    change: Change,
    accepted_findings: Mapping[str, FindingRef],
    diff: set[str],
    existing: set[str],
    editable: set[str],
    allow_create: bool,
) -> list[str]:
    """Check one claim's paths, protected skills and cited findings."""
    problems = []
    for path in change.files:
        pure = PurePosixPath(path)
        if pure.is_absolute() or ".." in pure.parts or len(pure.parts) < 2 or "\\" in path:
            problems.append(f"file path {path!r} must be relative to the skills directory")
        elif pure.parts[0] in existing and pure.parts[0] not in editable:
            problems.append(f"skill {pure.parts[0]} is read-only")
        elif pure.parts[0] not in existing and not allow_create:
            problems.append(f"creation of skill {pure.parts[0]} is forbidden")
        elif path not in diff:
            problems.append(f"claims {path} but the file is unchanged")
    unknown = [finding for finding in change.findings if finding not in accepted_findings]
    if unknown:
        problems.append(f"cites findings that are not accepted: {unknown[:5]}")
    return problems


def _groups(changes: list[Change]) -> list[list[Change]]:
    """Group changes that share at least one file, transitively."""
    groups: list[tuple[set[str], list[Change]]] = []
    for change in changes:
        files, members = set(change.files), [change]
        for group in [g for g in groups if g[0] & files]:
            groups.remove(group)
            files |= group[0]
            members = group[1] + members
        groups.append((files, members))
    return [members for _, members in groups]


def _assemble(baseline: Path, edited: Path, target: Path, files: list[str]) -> None:
    """Copy the baseline and apply the accepted files from the edited copy."""
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(baseline, target)
    for path in files:
        source, destination = edited / path, target / path
        if source.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        elif destination.exists():
            destination.unlink()
    for directory in sorted((p for p in target.rglob("*") if p.is_dir()), reverse=True):
        if not any(directory.iterdir()):
            directory.rmdir()


def _failing_skills(layout: RunLayout, files: list[str]) -> dict[str, list[str]]:
    """Check every touched candidate skill, ignoring problems its baseline already had."""
    failing = {}
    for skill in sorted({PurePosixPath(p).parts[0] for p in files}):
        candidate, baseline = layout.candidate_skills / skill, layout.input_skills / skill
        if not candidate.exists():
            continue
        before = baseline if baseline.is_dir() else None
        inherited = set(check_skill(before, None)) if before else set()
        problems = [p for p in check_skill(candidate, before) if p not in inherited]
        if problems:
            failing[skill] = problems
    return failing
