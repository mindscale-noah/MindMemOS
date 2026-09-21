"""Recover a later failed portion of a STATE-Bench online experiment.

This protocol deliberately keeps the original run immutable.  It reuses the
completed baseline evaluation and the existing Evolution project state, repairs
the first incomplete round, and writes a separate repair artifact tree.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from . import runner as statebench_runner
from .config_runner import DEFAULT_CONFIG, _format, _load_config, _load_keys, _resolve
from .schedule import FeedbackEvoSchedule, build_schedule

DEFAULT_FROM_ROUND = 9
DEFAULT_MAX_SCATTERED_FAILURES = 2


class RepairError(RuntimeError):
    """Raised when a repair cannot preserve the online experiment protocol."""


def _schedule(config: dict[str, Any], seed: int) -> FeedbackEvoSchedule:
    experiment = config["experiment"]
    training = config["training"]
    state_bench_dir = _resolve(str(experiment["state_bench_dir"]))
    split_path = state_bench_dir / "state_bench" / "domains" / str(experiment["domain"]) / "splits" / "train_test.json"
    if not split_path.exists():
        raise RepairError(f"split file not found: {split_path}")
    splits = json.loads(split_path.read_text(encoding="utf-8"))
    return build_schedule(
        list(splits.get("splits", {}).get("train", [])),
        list(splits.get("splits", {}).get("test", [])),
        rounds=int(training["rounds"]),
        train_per_round=int(training["tasks_per_round"]),
        seed=seed,
        eval_count=int(config["evaluation"]["test_task_count"]),
    )


def _context(
    config: dict[str, Any],
    seed: int,
    *,
    from_round: int | None,
    source_output_dir: Path | None,
    output_dir: Path | None,
    separate_output: bool,
    require_credentials: bool,
) -> dict[str, Any]:
    experiment = config["experiment"]
    training = config["training"]
    evaluation = config["evaluation"]
    isolation = config["isolation"]
    repair = config.get("repair") or {}
    round_index = int(from_round if from_round is not None else repair.get("from_round", DEFAULT_FROM_ROUND))
    rounds = int(training["rounds"])
    if round_index < 1 or round_index > rounds:
        raise RepairError(f"from_round must be between 1 and {rounds}")

    experiment_id = _format(str(experiment["group_id_template"]), seed)
    source_root = (
        _resolve(source_output_dir)
        if source_output_dir is not None
        else _resolve(str(experiment["output_dir"])) / f"seed{seed}"
    ).resolve()
    configured_in_place = str(repair.get("output_mode", "in-place")) == "in-place"
    in_place = output_dir is None and configured_in_place and not separate_output
    output_template = str(
        repair.get("output_dir_template") or str(experiment["output_dir"]) + "/seed{seed}-repair-round{from_round:02d}"
    )
    repair_output = (
        source_root
        if in_place
        else _resolve(output_dir)
        if output_dir is not None
        else _resolve(output_template.format(seed=seed, from_round=round_index))
    ).resolve()
    if not in_place and (repair_output == source_root or source_root in repair_output.parents):
        raise RepairError("repair output must be separate from, and outside, the source output")

    keys = _load_keys(_resolve(str(isolation["api_keys_file"])))
    branches: dict[str, dict[str, str]] = {}
    for branch in ("baseline", "evolution"):
        branch_config = isolation[branch]
        project_id = _format(str(branch_config["project_id_template"]), seed)
        key_id = _format(str(branch_config["key_id_template"]), seed)
        entry = keys.get(key_id)
        if not entry:
            if require_credentials or branch == "evolution":
                raise RepairError(f"key_id {key_id!r} is missing from api_keys.yaml")
            branches[branch] = {
                "project_id": project_id,
                "key_id": key_id,
                "api_key": "",
                "credential_status": "missing",
            }
            continue
        if entry.get("project_id") != project_id:
            raise RepairError(f"{key_id} points to {entry.get('project_id')!r}, expected {project_id!r}")
        branches[branch] = {
            "project_id": project_id,
            "key_id": key_id,
            "api_key": str(entry["api_key"]),
            "credential_status": "available",
        }

    return {
        "config": config,
        "seed": seed,
        "from_round": round_index,
        "rounds": rounds,
        "experiment_id": experiment_id,
        "domain": str(experiment["domain"]),
        "api_base": str(experiment["api_base"]),
        "agent_model_name": str(experiment["agent_model_name"]),
        "state_bench_dir": _resolve(str(experiment["state_bench_dir"])),
        "source_root": source_root,
        "output_root": repair_output,
        "in_place": in_place,
        "branches": branches,
        "train_workers": int(training["train_workers"]),
        "eval_workers": int(evaluation["eval_workers"]),
        "no_score": bool(training.get("no_score", False)),
        "checkpoint_runs": int(evaluation["checkpoint_runs"]),
        "final_runs": int(evaluation["final_runs"]),
        "max_scattered_failures": int(repair.get("max_scattered_failures", DEFAULT_MAX_SCATTERED_FAILURES)),
    }


def _metadata(ctx: dict[str, Any]) -> dict[str, Any]:
    return {
        "protocol": "online-controlled-repair",
        "seed": ctx["seed"],
        "domain": ctx["domain"],
        "experiment_id": ctx["experiment_id"],
        "from_round": ctx["from_round"],
        "source_output_dir": str(ctx["source_root"]),
        "output_dir": str(ctx["output_root"]),
        "overwrite_source": ctx["in_place"],
        "max_scattered_failures": ctx["max_scattered_failures"],
        "baseline_reused": True,
        "branches": {
            branch: {
                "project_id": values["project_id"],
                "key_id": values["key_id"],
                "credential_status": values["credential_status"],
            }
            for branch, values in ctx["branches"].items()
        },
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    statebench_runner._write_json(path, payload)


def _load_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return statebench_runner._load(path)
    except Exception as exc:
        raise RepairError(f"invalid repair artifact {path}: {exc}") from exc


def _prepare_manifest(ctx: dict[str, Any], *, resume: bool) -> Path:
    root = ctx["output_root"]
    manifest_path = root / "manifest.json"
    if ctx["in_place"]:
        if root != ctx["source_root"]:
            raise RepairError("in-place repair must use the configured source output directory")
        if not manifest_path.exists():
            raise RepairError(f"source manifest not found: {manifest_path}")
        manifest = statebench_runner._load(manifest_path)
        previous = manifest.get("repair") or {}
        if previous.get("status") in {"running", "failed", "complete"} and not resume:
            raise RepairError("source output already has a repair record; pass --resume")
        repair_metadata = {
            **_metadata(ctx),
            "status": "running",
            "previous_status": previous.get("status"),
        }
        manifest["repair"] = repair_metadata
        _write_json(manifest_path, manifest)
        return manifest_path
    if manifest_path.exists():
        manifest = statebench_runner._load(manifest_path)
        expected = _metadata(ctx)
        for key in ("experiment_id", "seed", "domain", "from_round", "source_output_dir"):
            if manifest.get(key) != expected[key]:
                raise RepairError(f"repair manifest mismatch for {key}: {manifest.get(key)!r}")
        if not resume:
            raise RepairError(f"repair output already exists; pass --resume: {root}")
        return manifest_path
    if root.exists() and any(root.iterdir()):
        raise RepairError(f"repair output is non-empty without a manifest: {root}")
    _write_json(
        manifest_path,
        {
            **_metadata(ctx),
            "status": "running",
            "git_commit": statebench_runner._git_commit(),
            "trajectory_mapping_version": statebench_runner.TRAJECTORY_MAPPING_VERSION,
            "source_manifest": str(ctx["source_root"] / "manifest.json"),
        },
    )
    return manifest_path


def _update_manifest(path: Path, **updates: Any) -> None:
    manifest = statebench_runner._load(path)
    if updates.pop("in_place", False):
        repair = manifest.get("repair") or {}
        repair.update(updates)
        manifest["repair"] = repair
    else:
        manifest.update(updates)
    _write_json(path, manifest)


def _clear_in_place_targets(ctx: dict[str, Any], *, resume: bool) -> None:
    """Remove stale failed-tail artifacts before the first in-place retry.

    The old final evaluation was produced before the repaired Round 9/10
    state, so retaining its 35 successful files would mix two different
    memory/configuration states.  Only the Evolution tail and final eval are
    removed; Baseline and all earlier rounds remain untouched.
    """
    if not ctx["in_place"] or resume:
        return
    evolution = ctx["source_root"] / "evolution"
    reports = evolution / "reports"
    stale_rounds = {ctx["from_round"], ctx["from_round"] + 1}
    rounds_jsonl = reports / "rounds.jsonl"
    if rounds_jsonl.exists():
        kept: list[str] = []
        for line in rounds_jsonl.read_text(encoding="utf-8").splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                kept.append(line)
                continue
            if int(value.get("round_index", -1)) not in stale_rounds:
                kept.append(line)
        rounds_jsonl.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    for round_index in (ctx["from_round"], ctx["from_round"] + 1):
        for path in (
            reports / f"feedback_round{round_index:02d}.json",
            reports / f"evolution_round{round_index:02d}.json",
            reports / f"round_{round_index:02d}.json",
        ):
            if path.exists():
                path.unlink()
    final_checkpoint = ctx["rounds"]
    for path in (evolution / "eval" / f"checkpoint{final_checkpoint:02d}",):
        if path.is_dir():
            shutil.rmtree(path)
    for snapshot_root in (evolution / "eval_state_before", evolution / "eval_state_after"):
        for snapshot_dir in (snapshot_root / "memory_snapshots", snapshot_root / "config_snapshots"):
            path = snapshot_dir / f"checkpoint{final_checkpoint:02d}.json"
            if path.exists():
                path.unlink()
    for path in (
        evolution / "logs" / f"train_round{ctx['from_round'] + 1:02d}.log",
        evolution / "logs" / f"train_round{ctx['from_round'] + 1:02d}.failures.json",
        evolution / "logs" / f"eval_checkpoint{final_checkpoint:02d}.log",
        evolution / "logs" / f"eval_checkpoint{final_checkpoint:02d}.failures.json",
        ctx["source_root"] / "comparison.json",
        ctx["source_root"] / "paired_results.jsonl",
        ctx["source_root"] / "reports" / "summary.json",
    ):
        if path.exists():
            path.unlink()


def _validate_source(
    ctx: dict[str, Any],
    schedule: FeedbackEvoSchedule,
    *,
    allow_repair_state: bool = False,
) -> None:
    source = ctx["source_root"]
    manifest_path = source / "manifest.json"
    if not manifest_path.exists():
        raise RepairError(f"source manifest not found: {manifest_path}")
    manifest = statebench_runner._load(manifest_path)
    if manifest.get("experiment_id") != ctx["experiment_id"]:
        raise RepairError("source manifest experiment_id does not match the configured experiment")
    if int(manifest.get("seed", -1)) != ctx["seed"] or manifest.get("domain") != ctx["domain"]:
        raise RepairError("source manifest seed/domain does not match the repair request")

    baseline_eval = source / "baseline" / "eval" / f"checkpoint{schedule.rounds[-1].round_index:02d}"
    baseline_final = statebench_runner._summarize_eval(baseline_eval, schedule.eval_task_ids, ctx["final_runs"])
    if not baseline_final.get("complete"):
        raise RepairError("source Baseline final evaluation is incomplete")

    evolution_final = statebench_runner._summarize_eval(
        source / "evolution" / "eval" / f"checkpoint{schedule.rounds[-1].round_index:02d}",
        schedule.eval_task_ids,
        ctx["final_runs"],
    )
    if evolution_final.get("complete") and not allow_repair_state:
        raise RepairError("source Evolution final evaluation is already complete; refusing a duplicate repair")

    source_evolution = source / "evolution"
    if ctx["from_round"] > 1:
        previous = source_evolution / "reports" / f"round_{ctx['from_round'] - 1:02d}.json"
        if not previous.exists():
            raise RepairError(f"source Evolution report missing: {previous}")
    repair_round = source_evolution / "reports" / f"round_{ctx['from_round']:02d}.json"
    if not repair_round.exists() and not allow_repair_state:
        raise RepairError(f"source repair round report missing: {repair_round}")
    train_dir = source_evolution / "train" / f"round{ctx['from_round']:02d}"
    plan = schedule.rounds[ctx["from_round"] - 1]
    if not statebench_runner._batch_complete(train_dir, list(plan.train_task_ids), 1):
        raise RepairError("source repair round does not contain all training trajectories")

    # A non-zero successful count would mean that replaying the same
    # conversations could create duplicate feedback events in the existing
    # project.
    feedback_path = source_evolution / "reports" / f"feedback_round{ctx['from_round']:02d}.json"
    feedback = _load_if_exists(feedback_path)
    if feedback and int(feedback.get("collected_event_count") or 0) != 0 and not allow_repair_state:
        raise RepairError(
            f"source Round {ctx['from_round']} already has successful feedback events; refusing unsafe replay"
        )


def _feedback_complete(feedback: dict[str, Any], expected: int) -> bool:
    return (
        int(feedback.get("expected_event_count") or 0) == expected
        and int(feedback.get("collected_event_count") or 0) == expected
        and not feedback.get("failures")
    )


def _repair_round(
    *,
    ctx: dict[str, Any],
    schedule: FeedbackEvoSchedule,
    round_index: int,
    train_dir: Path,
    runtime: dict[str, Any],
    source_train_dir: Path | None = None,
) -> dict[str, Any]:
    branch_root = ctx["output_root"] / "evolution"
    reports = branch_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    plan = schedule.rounds[round_index - 1]
    expected = len(plan.train_task_ids)
    user_prefix = f"{ctx['experiment_id']}::evolution"
    api_key = ctx["branches"]["evolution"]["api_key"]

    report_path = reports / f"round_{round_index:02d}.json"
    existing = _load_if_exists(report_path)
    if existing and existing.get("repair_status") == "complete":
        return existing

    feedback_path = reports / f"feedback_round{round_index:02d}.json"
    feedback = _load_if_exists(feedback_path)
    if not feedback or not _feedback_complete(feedback, expected):
        feedback = statebench_runner._collect_round(
            train_dir,
            plan.train_task_ids,
            ctx["api_base"],
            api_key,
            feedback_path,
            user_prefix,
            retry_failed=True,
        )
    if not _feedback_complete(feedback, expected):
        raise RepairError(
            f"Round {round_index} feedback remains incomplete: "
            f"{feedback.get('collected_event_count', 0)}/{expected} events"
        )

    evolution_path = reports / f"evolution_round{round_index:02d}.json"
    evolution = _load_if_exists(evolution_path)
    repair_key = f"{ctx['experiment_id']}:repair:round-{round_index:02d}"
    if (
        not evolution
        or evolution.get("status") != "ok"
        or (
            int(evolution.get("selected_event_count") or 0) != expected
            or int(evolution.get("consumed_event_count") or 0) != expected
        )
    ):
        evolution = statebench_runner._evolve(
            ctx["api_base"],
            api_key,
            event_selection="unconsumed",
            idempotency_key=repair_key,
        )
        _write_json(evolution_path, evolution)
    if (
        evolution.get("status") != "ok"
        or int(evolution.get("selected_event_count") or 0) != expected
        or int(evolution.get("consumed_event_count") or 0) != expected
    ):
        raise RepairError(f"Round {round_index} evolution did not consume exactly {expected} events")

    report = statebench_runner._round_report(
        plan,
        train_dir,
        feedback,
        evolution,
        runtime=runtime,
    )
    report.update(
        {
            "branch": "evolution",
            "project_id": ctx["branches"]["evolution"]["project_id"],
            "collect_enabled": True,
            "evolve_enabled": True,
            "user_id_prefix": user_prefix,
            "repair_status": "complete",
            "source_train_dir": str(source_train_dir) if source_train_dir else None,
            "repair_idempotency_key": repair_key,
        }
    )
    statebench_runner._write_round(reports, report)
    return report


def _failure_count(runtime: dict[str, Any]) -> int:
    return len(runtime.get("failures") or [])


def _baseline_summary(ctx: dict[str, Any], schedule: FeedbackEvoSchedule) -> dict[str, Any]:
    source = ctx["source_root"] / "baseline"
    final_checkpoint = schedule.rounds[-1].round_index
    final = statebench_runner._summarize_eval(
        source / "eval" / f"checkpoint{final_checkpoint:02d}",
        schedule.eval_task_ids,
        ctx["final_runs"],
    )
    if not final.get("complete"):
        raise RepairError("source Baseline final evaluation is incomplete")
    return {
        "branch": "baseline",
        "project_id": ctx["branches"]["baseline"]["project_id"],
        "reused_from": str(source),
        "checkpoints": {str(final_checkpoint): final},
        "rounds": [],
        "failures": [],
    }


def _evolution_rounds(ctx: dict[str, Any], repair_reports: Path, repaired_last_round: int) -> list[dict[str, Any]]:
    """Combine immutable completed rounds with the repaired tail.

    The final comparison must include the original Evolution costs and signal
    counts, not just the two reports generated by the repair.  Reports for the
    repaired range are read from the repair tree; earlier reports remain
    immutable references to the source run.
    """
    source_reports = ctx["source_root"] / "evolution" / "reports"
    rounds: list[dict[str, Any]] = []
    for round_index in range(1, ctx["from_round"]):
        path = source_reports / f"round_{round_index:02d}.json"
        report = _load_if_exists(path)
        if report is None:
            raise RepairError(f"source Evolution report missing: {path}")
        report = dict(report)
        report["report_source"] = str(path)
        rounds.append(report)
    for round_index in range(ctx["from_round"], repaired_last_round + 1):
        path = repair_reports / f"round_{round_index:02d}.json"
        report = _load_if_exists(path)
        if report is None:
            raise RepairError(f"repair Evolution report missing: {path}")
        rounds.append(report)
    return rounds


def run_repair(ctx: dict[str, Any], *, resume: bool = False) -> int:
    schedule = _schedule(ctx["config"], ctx["seed"])
    manifest_path = _prepare_manifest(ctx, resume=resume)
    try:
        _validate_source(ctx, schedule, allow_repair_state=ctx["in_place"] and resume)
        source_evolution = ctx["source_root"] / "evolution"
        source_train_dir = source_evolution / "train" / f"round{ctx['from_round']:02d}"
        source_report_path = source_evolution / "reports" / f"round_{ctx['from_round']:02d}.json"
        source_report = _load_if_exists(source_report_path) or {}
        start_runtime = source_report.get("runtime_counts") or {}
        _clear_in_place_targets(ctx, resume=resume)
        repair_evolution = ctx["output_root"] / "evolution"
        repair_reports = repair_evolution / "reports"
        repair_reports.mkdir(parents=True, exist_ok=True)
        user_prefix = f"{ctx['experiment_id']}::evolution"
        api_key = ctx["branches"]["evolution"]["api_key"]

        _repair_round(
            ctx=ctx,
            schedule=schedule,
            round_index=ctx["from_round"],
            train_dir=source_train_dir,
            runtime=start_runtime,
            source_train_dir=source_train_dir,
        )
        statebench_runner._capture_checkpoint_state(
            project_id=ctx["branches"]["evolution"]["project_id"],
            output_root=repair_evolution,
            checkpoint=ctx["from_round"],
            api_base=ctx["api_base"],
            api_key=api_key,
        )

        round_index = ctx["from_round"] + 1
        if round_index > ctx["rounds"]:
            raise RepairError("from_round must leave at least one training round to repair")
        plan = schedule.rounds[round_index - 1]
        train_dir = repair_evolution / "train" / f"round{round_index:02d}"
        runtime = statebench_runner._run_batch(
            ctx["state_bench_dir"],
            domain=ctx["domain"],
            task_ids=list(plan.train_task_ids),
            output_dir=train_dir,
            log_path=repair_evolution / "logs" / f"train_round{round_index:02d}.log",
            agent_class="MindMemOSAgent",
            agent_model_name=ctx["agent_model_name"],
            reasoning_level=None,
            num_workers=ctx["train_workers"],
            no_score=ctx["no_score"],
            role="train",
            api_base=ctx["api_base"],
            api_key=api_key,
            user_id_prefix=user_prefix,
        )
        if not statebench_runner._batch_complete(train_dir, list(plan.train_task_ids), 1):
            raise RepairError(
                f"repair Round {round_index} still has missing trajectories; "
                f"see {repair_evolution / 'logs' / f'train_round{round_index:02d}.failures.json'}"
            )
        if _failure_count(runtime) > ctx["max_scattered_failures"]:
            raise RepairError(
                f"repair Round {round_index} has {_failure_count(runtime)} runtime failures; "
                f"maximum accepted is {ctx['max_scattered_failures']}"
            )
        _repair_round(
            ctx=ctx,
            schedule=schedule,
            round_index=round_index,
            train_dir=train_dir,
            runtime=runtime,
        )
        statebench_runner._capture_checkpoint_state(
            project_id=ctx["branches"]["evolution"]["project_id"],
            output_root=repair_evolution,
            checkpoint=round_index,
            api_base=ctx["api_base"],
            api_key=api_key,
        )

        baseline = _baseline_summary(ctx, schedule)
        final_checkpoint = schedule.rounds[-1].round_index
        evaluation = statebench_runner._run_eval(
            state_bench_dir=ctx["state_bench_dir"],
            domain=ctx["domain"],
            task_ids=schedule.eval_task_ids,
            output_root=repair_evolution,
            checkpoint=final_checkpoint,
            num_runs=ctx["final_runs"],
            agent_class="MindMemOSAgent",
            agent_model_name=ctx["agent_model_name"],
            reasoning_level=None,
            num_workers=ctx["eval_workers"],
            api_base=ctx["api_base"],
            api_key=api_key,
            user_id_prefix=user_prefix,
            audit_memory=True,
        )
        evolution = {
            "branch": "evolution",
            "project_id": ctx["branches"]["evolution"]["project_id"],
            "checkpoints": {str(final_checkpoint): evaluation},
            "rounds": _evolution_rounds(ctx, repair_reports, round_index),
            "failures": evaluation.get("failures", []),
        }
        comparison = statebench_runner._paired_comparison(baseline, evolution)
        paired = statebench_runner._write_paired_results(
            ctx["output_root"] / "paired_results.jsonl",
            baseline,
            evolution,
            domain=ctx["domain"],
            seed=ctx["seed"],
        )
        comparison.update(paired)
        comparison["repair"] = {
            "source_output_dir": str(ctx["source_root"]),
            "from_round": ctx["from_round"],
            "baseline_reused": True,
            "evolution_project_id": ctx["branches"]["evolution"]["project_id"],
        }
        _write_json(ctx["output_root"] / "comparison.json", comparison)
        _write_json(
            ctx["output_root"] / "reports" / "summary.json",
            {
                "protocol": "online-controlled-repair",
                "domain": ctx["domain"],
                "seed": ctx["seed"],
                "source_output_dir": str(ctx["source_root"]),
                "baseline": baseline,
                "evolution": evolution,
                "comparison": comparison,
            },
        )
        _update_manifest(
            manifest_path,
            in_place=ctx["in_place"],
            status="complete" if comparison.get("pairing_complete") and evaluation.get("complete") else "incomplete",
            pairing_complete=bool(comparison.get("pairing_complete")),
            final_evaluation_complete=bool(evaluation.get("complete")),
        )
        return 0 if comparison.get("pairing_complete") and evaluation.get("complete") else 1
    except Exception as exc:
        _update_manifest(
            manifest_path,
            in_place=ctx["in_place"],
            status="failed",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        print(f"STATE-Bench repair failed: {exc}", file=sys.stderr)
        return 1


def run(
    config_path: Path,
    seed: int,
    *,
    source_output_dir: Path | None = None,
    output_dir: Path | None = None,
    from_round: int | None = None,
    separate_output: bool = False,
    resume: bool = False,
    dry_run: bool = False,
) -> int:
    config_path = _resolve(config_path)
    config = _load_config(config_path)
    ctx = _context(
        config,
        seed,
        from_round=from_round,
        source_output_dir=source_output_dir,
        output_dir=output_dir,
        separate_output=separate_output,
        require_credentials=not dry_run,
    )
    metadata = _metadata(ctx)
    metadata["config"] = str(config_path)
    metadata["expected_repair_rounds"] = [ctx["from_round"], ctx["from_round"] + 1]
    if dry_run:
        print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    return run_repair(ctx, resume=resume)


def add_repair_args(parser: argparse.ArgumentParser) -> None:
    """Add centralized-config arguments for the repair protocol."""
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--from-round", type=int, default=None)
    parser.add_argument("--source-output-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--separate-output",
        action="store_true",
        help="Write a separate repair tree instead of replacing the source tail.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate repair scope and print redacted metadata without starting tasks.",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_repair_args(parser)
    args = parser.parse_args(argv)
    return run(
        args.config,
        args.seed,
        source_output_dir=args.source_output_dir,
        output_dir=args.output_dir,
        from_round=args.from_round,
        separate_output=args.separate_output,
        resume=args.resume,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
