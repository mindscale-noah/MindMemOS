"""Aggregate saved task outcomes without executing or resuming any tasks."""

import csv
import json
from pathlib import Path
from statistics import mean, pstdev

from mindmemos_skill.common.checkpoint import read_boundary
from mindmemos_skill.common.json import read_json, write_json
from mindmemos_skill.envs.typing import RolloutResult
from pydantic import JsonValue

from .typing import BENCHMARKS, ArmResult, BenchmarkReport, EnvTestSummary, SuiteReport


def _row(benchmark: BenchmarkReport, arm: ArmResult) -> dict[str, JsonValue]:
    """Retain planned denominators and withhold scores for incomplete conditions."""
    if len(set(arm.task_ids)) != len(arm.task_ids):
        raise ValueError("Planned task IDs must be unique")
    seen = [case.task_id for case in arm.cases]
    if seen != arm.task_ids[: len(seen)]:
        raise ValueError("Saved cases must match the planned task order")
    completed = [
        case for case in arm.cases if case.error is None and case.score is not None and case.success is not None
    ]
    planned = len(arm.task_ids)
    complete = len(completed) == planned and planned > 0
    return {
        "benchmark": benchmark.benchmark,
        "arm": arm.arm,
        "repeat": arm.repeat,
        "planned": planned,
        "completed": len(completed),
        "errors": sum(case.error is not None for case in arm.cases),
        "coverage": len(completed) / planned if planned else 0,
        "incomplete": not complete or benchmark.error is not None,
        "mean_score": mean(case.score for case in completed) if complete else None,
        "success_rate": mean(case.success for case in completed) if complete else None,
        "wall_seconds": arm.wall_seconds
        if arm.wall_seconds is not None
        else (
            (max(case.finished_at for case in arm.cases) - min(case.started_at for case in arm.cases)).total_seconds()
            if arm.cases and all(case.started_at is not None and case.finished_at is not None for case in arm.cases)
            else None
        ),
        "task_seconds": sum(case.duration_s for case in arm.cases)
        if arm.cases and all(case.duration_s is not None for case in arm.cases)
        else None,
        "input_tokens": sum(case.usage["input_tokens"] for case in arm.cases)
        if complete and all(case.usage is not None for case in arm.cases)
        else None,
        "output_tokens": sum(case.usage["output_tokens"] for case in arm.cases)
        if complete and all(case.usage is not None for case in arm.cases)
        else None,
        "total_tokens": sum(case.usage["total_tokens"] for case in arm.cases)
        if complete and all(case.usage is not None for case in arm.cases)
        else None,
        "seed": arm.seed,
        "metrics": {
            key: mean(case.metrics[key] for case in completed)
            for key in sorted({key for case in completed for key in case.metrics})
            if complete and all(isinstance(case.metrics.get(key), (int, float)) for case in completed)
        },
    }


def write_report(report: SuiteReport, output_dir: Path) -> None:
    """Write JSON, CSV, and Markdown views from saved task outcomes.

    Args:
        report: Requested benchmark scope and original result references.
        output_dir: Existing run directory where reports are written.
    """
    rows = [_row(benchmark, arm) for benchmark in report.benchmarks for arm in benchmark.arms]
    incomplete = any(benchmark.error or not benchmark.arms for benchmark in report.benchmarks) or any(
        row["incomplete"] for row in rows
    )
    groups = []
    for benchmark in report.benchmarks:
        for arm_name in dict.fromkeys(arm.arm for arm in benchmark.arms):
            selected = [row for row in rows if row["benchmark"] == benchmark.benchmark and row["arm"] == arm_name]
            scores = [row["mean_score"] for row in selected]
            groups.append(
                {
                    "benchmark": benchmark.benchmark,
                    "arm": arm_name,
                    "repeat_mean_score": mean(scores)
                    if scores and all(score is not None for score in scores)
                    else None,
                    "repeat_score_std": pstdev(scores)
                    if scores and all(score is not None for score in scores)
                    else None,
                }
            )
    gains = []
    for benchmark in report.benchmarks:
        for trained in [arm for arm in benchmark.arms if arm.arm == "trained_skill"]:
            trained_row = _row(benchmark, trained)
            for baseline in [
                arm for arm in benchmark.arms if arm.arm != "trained_skill" and arm.repeat == trained.repeat
            ]:
                baseline_row = _row(benchmark, baseline)
                if (
                    trained.task_ids == baseline.task_ids
                    and trained.seed == baseline.seed
                    and not trained_row["incomplete"]
                    and not baseline_row["incomplete"]
                ):
                    gains.append(
                        {
                            "benchmark": benchmark.benchmark,
                            "repeat": trained.repeat,
                            "baseline": baseline.arm,
                            "score_gain": trained_row["mean_score"] - baseline_row["mean_score"],
                            "success_gain": trained_row["success_rate"] - baseline_row["success_rate"],
                        }
                    )
    write_json(
        output_dir / "summary.json",
        {
            "smoke": report.smoke,
            "incomplete": bool(incomplete),
            "benchmarks": report.benchmarks,
            "rows": rows,
            "repeats": groups,
            "gains": gains,
        },
    )
    columns = [
        "benchmark",
        "arm",
        "repeat",
        "planned",
        "completed",
        "errors",
        "coverage",
        "incomplete",
        "mean_score",
        "success_rate",
        "wall_seconds",
        "task_seconds",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "seed",
        "metrics",
    ]
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({**row, "metrics": json.dumps(row["metrics"], ensure_ascii=False)} for row in rows)
    lines = [
        "# Skill evaluation report",
        "",
        f"Mode: {'smoke' if report.smoke else 'full fixed split'}",
        f"Status: {'incomplete' if incomplete else 'complete'}",
        "",
        "Scores use the full planned task set. Incomplete conditions have no aggregate accuracy.",
        "",
        "| Benchmark | Arm | Repeat | Planned | Completed | Errors | Coverage | Score | Success |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                str(row[key]) if row[key] is not None else "missing"
                for key in (
                    "benchmark",
                    "arm",
                    "repeat",
                    "planned",
                    "completed",
                    "errors",
                    "coverage",
                    "mean_score",
                    "success_rate",
                )
            )
            + " |"
        )
    lines.extend(["", "## Repeated tests", ""])
    for group in groups:
        lines.append(
            f"- {group['benchmark']} / {group['arm']}: mean={group['repeat_mean_score']}, "
            f"population standard deviation={group['repeat_score_std']}"
        )
    lines.extend(["", "## Matched-condition gains", ""])
    for gain in gains:
        lines.append(
            f"- {gain['benchmark']} / repeat {gain['repeat']} / trained vs {gain['baseline']}: "
            f"score gain={gain['score_gain']}, success gain={gain['success_gain']}"
        )
    lines.extend(["", "## Training and failures", ""])
    for benchmark in report.benchmarks:
        lines.append(
            f"- {benchmark.benchmark}: pipeline seconds={benchmark.training_seconds}; "
            f"optimizer usage={benchmark.optimizer_usage}; error={benchmark.error}"
        )
    lines.extend(
        [
            "",
            "Missing token usage is retained as null. Training pipeline time includes its validation and first test.",
            "SearchQA score is EM; OfficeQA uses its native binary scorer. No cross-benchmark total is computed.",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_report(run_dir: Path) -> SuiteReport:
    """Reload original benchmark records for report-only aggregation.

    Args:
        run_dir: Existing run containing config.json and benchmark results.json files.

    Returns:
        Suite including explicit unfinished benchmarks when their results are absent.

    Raises:
        ValueError: Saved options or benchmark results are invalid.
        OSError: Run configuration cannot be read.
    """
    from .config import RunOptions

    directory = run_dir.expanduser().resolve()
    saved_options = read_json(directory / "config.json")["options"]
    legacy_suite = isinstance(saved_options, dict) and saved_options.get("benchmark") == "all"
    if legacy_suite:
        # Old suite outputs can still be aggregated without running any benchmarks.
        saved_options.pop("benchmark")
    options = RunOptions.model_validate(saved_options)
    benchmarks = BENCHMARKS if legacy_suite else (options.benchmark,)
    reports = []
    for benchmark in benchmarks:
        path = directory / benchmark / "results.json"
        if path.is_file():
            report = BenchmarkReport.model_validate(read_json(path))
            if report.benchmark != benchmark:
                raise ValueError("Benchmark results belong to a different benchmark")
            for arm in report.arms:
                for case in arm.cases:
                    if case.error is not None:
                        continue
                    path = case.result_path.resolve()
                    try:
                        if not path.is_relative_to(directory):
                            raise ValueError("Result reference escapes its run directory")
                        raw = (
                            read_boundary(path).payload["result"] if path.name == "completed.json" else read_json(path)
                        )
                        result = RolloutResult.model_validate(raw)
                        if (case.score, case.success, case.duration_s) != (
                            result.reward.score,
                            result.reward.success,
                            result.duration_s,
                        ):
                            raise ValueError("Saved task row disagrees with the original rollout")
                    except (ValueError, OSError) as exc:
                        case.error = f"Invalid original result: {exc}"
                        case.score = None
                        case.success = None
            expected = {(arm, repeat) for arm in options.arms for repeat in range(1, options.test_repeats + 1)}
            if {(arm.arm, arm.repeat) for arm in report.arms} != expected:
                report.error = report.error or "Requested arms or repetitions did not finish"
            reports.append(report)
        else:
            reports.append(BenchmarkReport(benchmark=benchmark, error="Benchmark did not finish"))
    return SuiteReport(output_dir=directory, smoke=options.smoke, benchmarks=reports)


def write_test_report(summary: EnvTestSummary, directory: Path) -> None:
    """Write the shared six-category human-readable evaluation summary.

    Args:
        summary: Counts, time, token coverage, and trajectory location.
        directory: Result directory containing summary.json.
    """
    missing = summary.usage_missing
    usage = summary.reported_usage
    elapsed = f"{summary.wall_seconds:.3f}" if summary.wall_seconds is not None else "unknown"
    successful = summary.successful
    rate = f"{summary.success_rate:.2%}" if summary.success_rate is not None else "N/A"
    tokens = str(summary.total_tokens) if not missing else f"unknown; reported {usage['total_tokens']}"
    (directory / "report.md").write_text(
        f"Environment: {summary.environment}\n\nMode: {summary.mode}\n\n"
        "| Metric | Value |\n| --- | --- |\n"
        f"| Tasks attempted / planned | {summary.attempted} / {summary.planned} |\n"
        f"| Tasks not started | {summary.not_started} |\n"
        f"| Elapsed seconds | {elapsed} |\n"
        f"| Tokens | {tokens} |\n"
        f"| Tasks with unknown token usage | {missing} |\n"
        f"| Execution errors | {summary.errors} |\n"
        f"| Successful / completed | {successful} / {summary.completed} |\n"
        f"| Success rate (completed tasks) | {rate} |\n\n"
        f"Trajectories: [{summary.trajectory_file.name}]({summary.trajectory_file.relative_to(directory)})\n",
        encoding="utf-8",
    )
