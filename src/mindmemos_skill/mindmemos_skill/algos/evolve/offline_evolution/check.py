"""Preflight stage deliverables in a temporary run, without changing real reports or state."""

import argparse
import asyncio
import json
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from ....common.json import read_json
from .change_gate import open_items_feedback, run_change_gate
from .change_gate import rejection_feedback as change_feedback
from .change_warnings import warning_feedback
from .evidence_gate import rejection_feedback, run_evidence_gate
from .layout import skill_names
from .typing import PreflightContext, RunLayout


async def preflight(stage: Literal["evidence", "change"], work_dir: Path) -> tuple[dict[str, object], bool]:
    """Run the real gate on copies, leaving state, reports and candidates untouched.

    Args:
        stage: Evidence collection or Skill-change gate to preview.
        work_dir: Existing execution directory containing the public preflight context.

    Returns:
        JSON-compatible feedback and whether the deliverables pass the preview.

    Raises:
        ValueError: State, inputs or deliverables are invalid or inconsistent.
        OSError: Required run files cannot be read.
    """
    source = RunLayout(root=work_dir)
    context = PreflightContext.model_validate(read_json(source.preflight_path))
    if source.inputs_digest() != context.inputs_digest:
        raise ValueError("inputs/ changed since the run was created")
    with tempfile.TemporaryDirectory(prefix="offline-evolution-check-") as scratch:
        layout = RunLayout(root=Path(scratch))
        shutil.copytree(source.inputs, layout.inputs)
        if stage == "evidence":
            shutil.copytree(source.evidence_workspace, layout.evidence_workspace)
            evidence = await run_evidence_gate(layout, 1, context.gate)
            feedback = rejection_feedback(evidence)
            return {
                "status": evidence.status,
                "valid_findings": len(evidence.findings),
                "invalid_findings": len(evidence.invalid_findings),
                "errors": evidence.errors,
                "feedback": feedback,
            }, not feedback
        shutil.copytree(source.change_workspace, layout.change_workspace)
        changes = run_change_gate(
            layout,
            context.accepted_findings,
            set(context.options.allowed_ids(skill_names(layout.input_skills))),
            context.options.allow_create_skills,
        )
        feedback = "\n\n".join(
            part
            for part in (change_feedback(changes), open_items_feedback(changes), warning_feedback(changes.warnings))
            if part
        )
        return {
            "accepted_files": changes.accepted_files,
            "rejected": [c.id for c in changes.changes if c.status == "rejected"],
            "unaddressed_findings": changes.unaddressed_findings,
            "feedback": feedback,
        }, not feedback


def main(argv: Sequence[str] | None = None) -> int:
    """Print JSON feedback for an internal preflight command.

    Args:
        argv: Command arguments; None reads the child process command line.

    Returns:
        Exit code 0 when ready, 1 for gate feedback or 2 for invalid run inputs.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("evidence", "change"))
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result, ready = asyncio.run(preflight(args.stage, args.run))
    except (OSError, ValueError) as error:
        print(json.dumps({"feedback": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False))
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
