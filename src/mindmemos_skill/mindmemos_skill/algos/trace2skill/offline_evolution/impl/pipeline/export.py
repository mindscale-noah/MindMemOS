"""Finalize each Skill independently and write final outputs."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from mindmemos_skill.algos.trace2skill.offline_evolution.impl.pipeline.artifacts import (
    scope_artifact_dir,
    scope_baseline_skills,
)
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.pipeline.editor.skill_validation import (
    validate_skill_tree,
)
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.pipeline.report import (
    build_final_report,
    render_final_report_md,
)
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.pipeline.types import PipelinePaths
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.utils.files import write_text_file


def _copy_tree(source: Path, target: Path) -> None:
    def ignore_generated(_directory: str, names: list[str]) -> set[str]:
        return {name for name in names if name in {"__pycache__", ".DS_Store"} or name.endswith(".pyc")}

    shutil.copytree(source, target, ignore=ignore_generated)


def _skill_dirs(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        return {}
    return {item.name: item for item in root.iterdir() if item.is_dir() and not item.name.startswith(".")}


def _finalize_scope(*, original: Path, evolved: Path, target: Path) -> list[dict[str, Any]]:
    original_skills = _skill_dirs(original)
    evolved_skills = _skill_dirs(evolved)
    validation = validate_skill_tree(original, evolved)
    errors_by_name = {
        str(skill.get("skill_name") or ""): list(skill.get("errors") or [])
        for skill in validation.get("skills", []) or []
    }
    target.mkdir(parents=True, exist_ok=False)
    outcomes: list[dict[str, Any]] = []
    for name in sorted(original_skills.keys() | evolved_skills.keys()):
        candidate = evolved_skills.get(name)
        original_skill = original_skills.get(name)
        candidate_errors = errors_by_name.get(name, [])
        if candidate is not None and not candidate_errors:
            _copy_tree(candidate, target / name)
            outcomes.append(
                {
                    "skill_name": name,
                    "finalization": "evolved",
                    "errors": [],
                }
            )
            continue
        if original_skill is not None:
            _copy_tree(original_skill, target / name)
            outcomes.append(
                {
                    "skill_name": name,
                    "finalization": "fallback_original",
                    "errors": candidate_errors,
                }
            )
        else:
            outcomes.append(
                {
                    "skill_name": name,
                    "finalization": "omitted_invalid_new",
                    "errors": candidate_errors,
                }
            )
    return outcomes


def _reset_final_outputs(paths: PipelinePaths, run_dir: Path) -> None:
    """Wipe prior final skills/report artifacts; deletion stays inside the run directory."""
    from mindmemos_skill.algos.trace2skill.offline_evolution.impl.pipeline.prepare.shared import (
        _remove_tree_under_run_dir,
    )

    for raw in (
        paths.final_skills_dir,
        paths.final_report_md,
    ):
        path = Path(raw)
        if path.exists():
            _remove_tree_under_run_dir(path, run_dir)


def export_final_skills(
    paths: PipelinePaths,
    run_dir: Path,
    *,
    scope_ids: list[str],
) -> dict[str, dict[str, Any]]:
    """Check Skill disclosure structure/resources and export scope libraries.

    Returns the per-scope export outcomes. The final report is NOT written
    here: it is a run-level closing artifact the runner renders after the
    finalize stage is marked done, so the report can truthfully cover every
    stage including finalize itself.
    """
    _reset_final_outputs(paths, run_dir)
    export_root = Path(paths.final_skills_dir)
    export_root.mkdir(parents=True, exist_ok=False)
    exports: dict[str, dict[str, Any]] = {}
    for scope_id in sorted(scope_ids):
        target = export_root / scope_id
        skills = _finalize_scope(
            original=scope_baseline_skills(paths.skills_baseline_dir, scope_id),
            evolved=scope_artifact_dir(paths.workspace_dir, scope_id, "skills"),
            target=target,
        )
        exports[scope_id] = {
            "scope_id": scope_id,
            "final_skills_path": str(target),
            "skills": skills,
        }
    return exports


def write_final_report(
    paths: PipelinePaths,
    *,
    scope_ids: list[str],
    scope_outcomes: list[dict[str, Any]],
    exports: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Render and write final_report.md; returns the report's summary counts."""
    report = build_final_report(paths, scope_ids, scope_outcomes, exports)
    write_text_file(paths.final_report_md, render_final_report_md(report))
    return report.get("summary", {})


__all__ = ["export_final_skills", "write_final_report"]
