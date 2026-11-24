"""Verify one collection's input coverage, finding structure, skill attribution and verbatim quotes.

Comparisons and diagnoses remain agent judgments. The Host does not classify cases as successful,
require full audit tables, or treat optional statistics as independently verified results.
"""

import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Final

from pydantic import ValidationError

from ....infra.execution import ExecutionConfig, execute
from ....infra.execution.constants import PYTHON_IO_ENCODING_ENV
from .config import EvidenceGateConfig
from .constants import COLLECTION_FILE
from .layout import skill_names, write_text
from .provenance import quoted
from .typing import EvidenceManifest, EvidenceReport, Finding, InvalidFinding, RunLayout, ScriptOutput

IGNORED = shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pyc")
"""Generated files excluded when freezing the evidence workspace."""

PYTHON_DONT_WRITE_BYTECODE_ENV: Final[str] = "PYTHONDONTWRITEBYTECODE"
"""Prevent collector imports from creating bytecode files in the frozen workspace."""


async def run_evidence_gate(layout: RunLayout, attempt: int, config: EvidenceGateConfig) -> EvidenceReport:
    """Freeze and rerun the collector, verify its unified findings and preserve the report.

    Args:
        layout: Run containing immutable inputs and the evidence workspace.
        attempt: Gate attempt number used for frozen scripts and reports.
        config: Collector interpreter, timeout and invalid-finding allowance.

    Returns:
        Collection verification report, also saved under the evidence directory.
    """
    frozen = layout.evidence / "frozen" / f"attempt-{attempt}"
    if frozen.exists():
        shutil.rmtree(frozen)
    shutil.copytree(layout.evidence_workspace, frozen, ignore=IGNORED)
    report = EvidenceReport(attempt=attempt, trajectory_files=len(layout.trajectory_files()))
    try:
        manifest = EvidenceManifest.model_validate_json((frozen / COLLECTION_FILE).read_text(encoding="utf-8"))
    except (OSError, ValidationError) as error:
        report.errors.append(f"{COLLECTION_FILE} missing or invalid: {_short(error)}")
    else:
        out = layout.evidence / "outputs" / f"attempt-{attempt}.json"
        succeeded, detail = await _run_collector(layout, frozen, manifest.collector, out, config)
        if not succeeded:
            report.errors.append(f"collector failed: {detail}")
        else:
            try:
                output = ScriptOutput.model_validate_json(out.read_text(encoding="utf-8"))
            except (OSError, ValidationError) as error:
                report.errors.append(f"collector output invalid: {_short(error)}")
            else:
                _verify_output(layout, output, report, config)
    if not report.errors:
        report.status = "accepted"
    write_text(layout.evidence / "reports" / f"attempt-{attempt}.json", report.model_dump_json(indent=1))
    write_text(layout.evidence / "report.json", report.model_dump_json(indent=1))
    write_text(layout.evidence / "report.md", render_evidence_report(report))
    return report


def _verify_output(layout: RunLayout, output: ScriptOutput, report: EvidenceReport, config: EvidenceGateConfig) -> None:
    """Check coverage and located findings; retain summaries and notes as author statements."""
    expected = set(layout.trajectory_files())
    reported = set(output.processed) | {failed.path for failed in output.failed}
    if expected - reported:
        report.errors.append(
            f"{len(expected - reported)} trajectory files neither processed nor reported as failed, "
            f"e.g. {sorted(expected - reported)[:3]}"
        )
    if reported - expected:
        report.errors.append(
            f"{len(reported - expected)} reported paths are not trajectory inputs, "
            f"e.g. {sorted(reported - expected)[:3]}"
        )
    report.processed = len(set(output.processed) & expected)
    report.failed, report.summary, report.notes = output.failed, output.summary, output.notes
    _verify_findings(layout, output, report)
    total = len(output.findings)
    allowed = max(1, math.floor(config.max_invalid_share * total))
    if len(report.invalid_findings) > allowed:
        report.errors.append(
            f"{len(report.invalid_findings)}/{total} findings failed verification (at most {allowed} may fail)"
        )


def _verify_findings(layout: RunLayout, output: ScriptOutput, report: EvidenceReport) -> None:
    """Exclude malformed, unattributed or untraceable entries and make duplicate ids unique."""
    contents: dict[str, bytes | None] = {}
    ids: set[str] = set()
    findings: list[Finding] = []
    for index, raw in enumerate(output.findings):
        try:
            findings.append(Finding.model_validate(raw))
        except ValidationError as error:
            entry = raw if isinstance(raw, dict) else {}
            report.invalid_findings.append(
                InvalidFinding(
                    id=str(entry.get("id") or f"#{index + 1}"),
                    path=str(entry.get("path") or ""),
                    quote=str(entry.get("quote") or "")[:200],
                    problem=f"malformed finding: {_short(error)[:200]}",
                )
            )
    original = {finding.id for finding in findings}
    skills = skill_names(layout.input_skills)
    for finding in findings:
        if finding.id in ids:
            suffix = 2
            while f"{finding.id}-{suffix}" in ids | original:
                suffix += 1
            finding = finding.model_copy(update={"id": f"{finding.id}-{suffix}"})
            report.renamed_ids += 1
        problem = None
        if not finding.path.startswith(("trajectories/", "skills/")):
            problem = "path must start with trajectories/ or skills/"
        else:
            if finding.path not in contents:
                path = _inside(layout.inputs, finding.path)
                contents[finding.path] = path.read_bytes() if path is not None and path.is_file() else None
            data = contents[finding.path]
            if data is None:
                problem = "cited file does not exist"
            elif not quoted(finding.quote, data):
                problem = "quote not found verbatim in the cited file"
            elif finding.skill and finding.skill not in skills:
                problem = f"skill {finding.skill!r} is not one of the input skills"
        ids.add(finding.id)
        if problem:
            report.invalid_findings.append(
                InvalidFinding(
                    id=finding.id,
                    path=finding.path,
                    quote=finding.quote[:200],
                    problem=problem,
                )
            )
        else:
            report.findings.append(finding)


async def _run_collector(
    layout: RunLayout, frozen: Path, relative: str, out: Path, config: EvidenceGateConfig
) -> tuple[bool, str]:
    """Execute the frozen collector on a fresh copy of the inputs."""
    script = _inside(frozen, relative)
    if script is None or not script.is_file():
        return False, f"script {relative} not found in the workspace"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory(prefix="inputs-") as scratch:
        inputs = Path(scratch) / "inputs"
        shutil.copytree(layout.inputs, inputs)
        env = {**os.environ, PYTHON_DONT_WRITE_BYTECODE_ENV: "1", PYTHON_IO_ENCODING_ENV: "utf-8"}
        try:
            result = await execute(
                [config.python, str(script), "--inputs", str(inputs), "--out", str(out)],
                frozen,
                ExecutionConfig(timeout_s=config.script_timeout_s, environment=env),
            )
        except subprocess.TimeoutExpired:
            return False, f"timed out after {config.script_timeout_s:.0f}s"
    output = result.stdout + result.stderr
    return result.returncode == 0, output.strip()[-600:] if result.returncode else ""


def rejection_feedback(report: EvidenceReport) -> str:
    """Explain structural and provenance errors without grading their meaning.

    Args:
        report: Latest collection verification result.

    Returns:
        Concrete rejection feedback, or an empty string for an accepted collection.
    """
    if report.status == "accepted":
        return ""
    return "\n".join(
        report.errors + [f"finding {finding.id}: {finding.problem}" for finding in report.invalid_findings[:5]]
    )


def render_evidence_report(report: EvidenceReport, limit: int = 30) -> str:
    """Render a compact view, keeping complete evidence in report.json.

    Args:
        report: Collection verification result and author-supplied observations.
        limit: Maximum findings listed in the Markdown view.

    Returns:
        Markdown separating verified structure from unverified author statements.
    """
    lines = [
        f"# Evidence report (attempt {report.attempt}) — {report.status}",
        "",
        f"Trajectory files: {report.trajectory_files}; processed: {report.processed}; failed: {len(report.failed)}.",
        f"Valid findings: {len(report.findings)}; invalid findings: {len(report.invalid_findings)}.",
        "Acceptance verifies structure, quotes and skill attribution; "
        "it does not establish causal effects or task correctness.",
        "Only findings from an accepted report may be cited.",
        "",
    ]
    lines += [f"- error: {error}" for error in report.errors]
    if report.renamed_ids:
        lines += [f"{report.renamed_ids} repeated finding ids renamed with a suffix.", ""]
    if report.failed:
        lines += ["Unparsed inputs:"] + [f"- {failed.path}: {failed.reason}" for failed in report.failed] + [""]
    if report.summary:
        lines += ["Author summary (unverified):", json.dumps(report.summary, ensure_ascii=False)[:2000], ""]
    notes = [report.notes] if isinstance(report.notes, str) else report.notes
    if any(notes):
        lines += ["Author notes (unverified):"] + [f"- {note}" for note in notes if note] + [""]
    listed = sorted(report.findings, key=lambda finding: finding.issue is None)
    for finding in listed[:limit]:
        tags = " ".join(
            f"{name}={value}"
            for name, value in (("issue", finding.issue), ("skill", finding.skill), ("case", finding.case))
            if value
        )
        lines.append(f"- `{finding.id}` `{finding.path}` {tags} — “{finding.quote[:240]}” {finding.note[:320]}")
    if len(listed) > limit:
        lines.append(f"- ... {len(listed) - limit} more in report.json")
    return "\n".join(lines) + "\n"


def _inside(root: Path, relative: str) -> Path | None:
    """Resolve a referenced file only when it remains inside the allowed input tree."""
    path = (root / relative).resolve()
    return path if path.is_relative_to(root.resolve()) else None


def _short(error: Exception) -> str:
    """Compact an exception for machine feedback."""
    return str(error).replace("\n", " ")[:1200]
