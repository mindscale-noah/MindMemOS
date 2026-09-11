"""Build the diff-, finalization-, and usage-based final report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mindmemos_skill.algos.trace2skill.offline_evolution.impl.pipeline.artifacts import scope_baseline_skills
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.pipeline.editor.diff import scope_tree_diff
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.pipeline.types import PipelinePaths
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.pipeline.usage import collect_llm_usage


def _portable_path(value: Any, *, run_dir: str) -> str:
    text = str(value or "")
    if not text:
        return ""
    path = Path(text)
    try:
        return path.resolve().relative_to(Path(run_dir).resolve()).as_posix()
    except ValueError:
        return path.name


def _pipeline_timing(state_path: str | Path) -> dict[str, Any]:
    """Stage timings from state.json.

    The report is a run-level closing artifact written by the runner AFTER
    the finalize stage is marked done, so every stage -- finalize included
    -- appears with its real status and duration.
    """
    path = Path(state_path)
    if not path.is_file():
        return {"total_stage_elapsed_sec": 0.0, "complete": False, "by_stage": {}}
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"total_stage_elapsed_sec": 0.0, "complete": False, "by_stage": {}}
    stages = state.get("stages") if isinstance(state, dict) else {}
    by_stage: dict[str, dict[str, Any]] = {}
    if isinstance(stages, dict):
        for stage_id, value in stages.items():
            if not isinstance(value, dict):
                continue
            item: dict[str, Any] = {"status": str(value.get("status") or "unknown")}
            duration = value.get("duration_sec")
            if isinstance(duration, (int, float)) and not isinstance(duration, bool):
                item["elapsed_sec"] = round(float(duration), 3)
            by_stage[str(stage_id)] = item
    return {
        "total_stage_elapsed_sec": round(sum(float(item.get("elapsed_sec") or 0.0) for item in by_stage.values()), 3),
        "complete": bool(by_stage) and all(item.get("status") == "done" for item in by_stage.values()),
        "by_stage": by_stage,
    }


def build_final_report(
    paths: PipelinePaths,
    scope_ids: list[str],
    scope_outcomes: list[dict[str, Any]],
    exports: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Report what was actually exported after per-Skill finalization."""
    outcomes = {str(item.get("scope_id") or ""): item for item in scope_outcomes if isinstance(item, dict)}
    summary = {
        "scopes": 0,
        "skills_total": 0,
        "skills_created": 0,
        "skills_updated": 0,
        "skills_unchanged": 0,
        "skills_fallback_original": 0,
        "skills_omitted_invalid": 0,
        "skills_with_errors": 0,
        "scopes_reverted": 0,
        "validation_retries": 0,
    }
    scope_reports: list[dict[str, Any]] = []
    for scope_id in sorted(scope_ids):
        export = exports.get(scope_id, {})
        final_path = Path(str(export.get("final_skills_path") or ""))
        original = scope_baseline_skills(paths.skills_baseline_dir, scope_id)
        filesystem_diff = scope_tree_diff(original, final_path)
        diff_by_name = {item["skill_name"]: item for item in filesystem_diff["skills"]}
        finalization = {
            str(item.get("skill_name") or ""): item for item in export.get("skills", []) or [] if isinstance(item, dict)
        }
        outcome = outcomes.get(scope_id, {})
        scope_status = str(outcome.get("status") or "unknown")
        if scope_status == "reverted":
            summary["scopes_reverted"] += 1
        validation_retries = int(outcome.get("validation_retries") or 0)
        summary["validation_retries"] += validation_retries
        skills: list[dict[str, Any]] = []
        for skill_name in sorted(diff_by_name.keys() | finalization.keys()):
            diff = diff_by_name.get(
                skill_name,
                {
                    "skill_name": skill_name,
                    "status": "error",
                    "changed_files": [],
                    "summary": {
                        "n_changed_files": 0,
                        "lines_added": 0,
                        "lines_removed": 0,
                    },
                },
            )
            finalized = finalization.get(skill_name, {})
            disposition = str(finalized.get("finalization") or "evolved")
            status = str(diff.get("status") or "error")
            errors = list(finalized.get("errors") or [])
            if disposition == "fallback_original":
                summary["skills_fallback_original"] += 1
                summary["skills_with_errors"] += 1
            elif disposition.startswith("omitted_invalid"):
                status = "error"
                summary["skills_omitted_invalid"] += 1
                summary["skills_with_errors"] += 1
            elif status == "created":
                summary["skills_created"] += 1
            elif status == "updated":
                summary["skills_updated"] += 1
            elif status == "unchanged":
                summary["skills_unchanged"] += 1
            else:
                summary["skills_with_errors"] += 1
            summary["skills_total"] += 1
            export_root = _portable_path(final_path, run_dir=paths.run_dir)
            skills.append(
                {
                    "skill_name": skill_name,
                    "status": status,
                    "finalization": disposition,
                    "errors": errors,
                    "export_path": (
                        f"{export_root}/{skill_name}"
                        if export_root and not disposition.startswith("omitted_invalid")
                        else ""
                    ),
                    "diff": diff,
                }
            )
        scope_reports.append(
            {
                "scope_id": scope_id,
                "status": scope_status,
                "error": str(outcome.get("error") or ""),
                "attempts": int(outcome.get("attempts") or 0),
                "validation_retries": validation_retries,
                "final_skills_path": _portable_path(final_path, run_dir=paths.run_dir),
                "diff": filesystem_diff,
                "skills": skills,
            }
        )
    summary["scopes"] = len(scope_reports)
    return {
        "artifacts": {
            "extractions_dir": _portable_path(paths.extractions_dir, run_dir=paths.run_dir),
            "final_skills_dir": _portable_path(paths.final_skills_dir, run_dir=paths.run_dir),
            "final_report_md": _portable_path(paths.final_report_md, run_dir=paths.run_dir),
        },
        "summary": summary,
        "pipeline_timing": _pipeline_timing(paths.state_json),
        "llm_usage": collect_llm_usage(paths.workspace_dir),
        "scopes": scope_reports,
    }


def _cell(value: Any) -> str:
    return str(value or "").replace("\n", " ").replace("|", "\\|")


def render_final_report_md(report: dict[str, Any]) -> str:
    artifacts = report.get("artifacts", {}) or {}
    summary = report.get("summary", {}) or {}
    usage = report.get("llm_usage", {}) or {}
    timing = report.get("pipeline_timing", {}) or {}
    lines = [
        "# Pipeline Final Report",
        "",
        f"- Final skills: `{artifacts.get('final_skills_dir', '')}`",
        "",
        "## Summary",
        "",
    ]
    for key in (
        "scopes",
        "skills_total",
        "skills_created",
        "skills_updated",
        "skills_unchanged",
        "skills_fallback_original",
        "skills_omitted_invalid",
        "skills_with_errors",
        "scopes_reverted",
        "validation_retries",
    ):
        lines.append(f"- {key}: {summary.get(key, 0)}")
    lines.extend(
        [
            "",
            "## LLM Usage",
            "",
            f"- logical_calls: {usage.get('logical_calls', 0)}",
            f"- attempts: {usage.get('attempts', 0)}",
            f"- successful_calls: {usage.get('successful_calls', 0)}",
            f"- failed_calls: {usage.get('failed_calls', 0)}",
            f"- input_tokens: {usage.get('input_tokens', 0)}",
            f"- cache_creation_input_tokens: {usage.get('cache_creation_input_tokens', 0)}",
            f"- cache_read_input_tokens: {usage.get('cache_read_input_tokens', 0)}",
            f"- output_tokens: {usage.get('output_tokens', 0)}",
            f"- accounted_tokens: {usage.get('accounted_tokens', 0)}",
            f"- recorded_cost_usd: {usage.get('recorded_cost_usd', 0)}",
            f"- attempts_without_usage: {usage.get('attempts_without_usage', 0)}",
            f"- attempts_without_cost: {usage.get('attempts_without_cost', 0)}",
            f"- tokens_complete: {str(bool(usage.get('tokens_complete', True))).lower()}",
            f"- cost_complete: {str(bool(usage.get('cost_complete', True))).lower()}",
            f"- usage_complete: {str(bool(usage.get('usage_complete', True))).lower()}",
            f"- failures: `{json.dumps(usage.get('failures', {}), sort_keys=True)}`",
            "",
            "| Stage | Calls | Attempts | Success | Failed | Input | Cache Create | Cache Read | Output | Accounted | Cost (USD) | Missing Usage |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for stage, item in (usage.get("by_stage", {}) or {}).items():
        lines.append(
            "| "
            + " | ".join(
                [
                    _cell(stage),
                    str(item.get("logical_calls", 0)),
                    str(item.get("attempts", 0)),
                    str(item.get("successful_calls", 0)),
                    str(item.get("failed_calls", 0)),
                    str(item.get("input_tokens", 0)),
                    str(item.get("cache_creation_input_tokens", 0)),
                    str(item.get("cache_read_input_tokens", 0)),
                    str(item.get("output_tokens", 0)),
                    str(item.get("accounted_tokens", 0)),
                    str(item.get("recorded_cost_usd", 0)),
                    str(item.get("attempts_without_usage", 0)),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Pipeline Timing",
            "",
            f"- total_stage_elapsed_sec: {timing.get('total_stage_elapsed_sec', 0)}",
            f"- complete: {str(bool(timing.get('complete', False))).lower()}",
            "",
            "| Stage | Status | Elapsed (sec) |",
            "| --- | --- | ---: |",
        ]
    )
    for stage, item in (timing.get("by_stage", {}) or {}).items():
        lines.append(
            "| "
            + " | ".join(
                [
                    _cell(stage),
                    _cell(item.get("status")),
                    str(item.get("elapsed_sec", "")),
                ]
            )
            + " |"
        )
    for scope in report.get("scopes", []) or []:
        lines.extend(
            [
                "",
                f"## Scope: `{_cell(scope.get('scope_id'))}`",
                "",
                f"- Status: `{_cell(scope.get('status'))}`",
                f"- Attempts: {scope.get('attempts', 0)}",
                f"- Validation retries: {scope.get('validation_retries', 0)}",
                f"- Final skills path: `{scope.get('final_skills_path', '')}`",
            ]
        )
        if scope.get("error"):
            lines.append(f"- Error: {_cell(scope.get('error'))}")
        lines.extend(
            [
                "",
                "| Skill | Status | Finalization | Files Changed | Lines + / - | Export Path |",
                "| --- | --- | --- | ---: | ---: | --- |",
            ]
        )
        for skill in scope.get("skills", []) or []:
            diff_summary = skill.get("diff", {}).get("summary", {})
            lines.append(
                "| "
                + " | ".join(
                    [
                        _cell(skill.get("skill_name")),
                        _cell(skill.get("status")),
                        _cell(skill.get("finalization")),
                        str(diff_summary.get("n_changed_files", 0)),
                        f"{diff_summary.get('lines_added', 0)} / {diff_summary.get('lines_removed', 0)}",
                        f"`{_cell(skill.get('export_path'))}`",
                    ]
                )
                + " |"
            )
            for error in skill.get("errors", []) or []:
                lines.append(f"  - `{_cell(skill.get('skill_name'))}`: {_cell(error)}")
    lines.append("")
    return "\n".join(lines)


__all__ = ["build_final_report", "render_final_report_md"]
