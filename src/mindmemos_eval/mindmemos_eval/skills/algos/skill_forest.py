"""Run Skill-Forest controls, training, and result reporting."""

import asyncio
import csv
import json
import subprocess
import sys
from collections.abc import Sequence
from datetime import datetime
from importlib.util import find_spec
from pathlib import Path
from statistics import mean, pstdev
from time import perf_counter
from uuid import uuid4

import pyarrow.parquet as parquet
import yaml
from mindmemos_skill.api.typing import FileSkillInfo, RegisterFromLocalRequest, TrainingRequest, TrainingSplit
from mindmemos_skill.common.checkpoint import fingerprint, read_boundary
from mindmemos_skill.common.json import read_json, write_json
from mindmemos_skill.common.secrets import redact_credentials
from mindmemos_skill.common.skill_runtime import load_runtime_skill
from mindmemos_skill.common.skill_versions import snapshot_from_version
from mindmemos_skill.config.config import ImplementationConfig, ServiceConfig
from mindmemos_skill.envs import Environment
from mindmemos_skill.envs.builtin_envs.alfworld_bounded_history.config import ALFWorldConfig
from mindmemos_skill.envs.builtin_envs.alfworld_bounded_history.simulator import Simulator
from mindmemos_skill.envs.builtin_envs.alfworld_bounded_history.typing import ALFWorldTask
from mindmemos_skill.envs.builtin_envs.spreadsheetbench.recalculation import _find_soffice
from mindmemos_skill.envs.builtin_envs.spreadsheetbench.tool_call_mode.environment import (
    SpreadsheetBenchToolCallingEnvironment,
)
from mindmemos_skill.envs.data import Data, DataSample
from mindmemos_skill.envs.typing import RolloutResult
from mindmemos_skill.pipelines.builtin_pipelines.training.typing import HeldOutResultIndex
from mindmemos_skill.skill import Skill, SkillSet, SkillSnapshot
from pydantic import JsonValue

from ..config import ROOT, EvalConfig, RunOptions
from ..rollout import case_from_result, run_tasks
from ..service import open_skill_service
from ..typing import (
    BENCHMARKS,
    ArmResult,
    BenchmarkName,
    BenchmarkReport,
    CaseResult,
    DatasetInputs,
    SuiteReport,
)


async def run(config: EvalConfig, options: RunOptions) -> SuiteReport:
    """Validate the selected benchmark and run its requested conditions.

    Args:
        config: Validated shared models, native forest, and environment settings.
        options: Selected benchmark and execution options.

    Returns:
        Results with explicit failures; reports are written automatically.

    Raises:
        ValueError: Output directory is nonempty or any selected preflight check fails.
    """
    root = options.output_dir or ROOT / "output/mindmemos_eval/skill" / (
        datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8]
    )
    root = root.expanduser().resolve()
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise ValueError(f"Output directory must be empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    errors = []
    for label, model in (("target", config.models.target), ("optimizer", config.models.optimizer)):
        if label == "optimizer" and "trained_skill" not in options.arms:
            continue
        try:
            model.require_connection()
        except ValueError as exc:
            errors.append(f"models.{label} in config/mindmemos_eval/skill_train_skill_forest/dev.yaml: {exc}")
    benchmark = options.benchmark
    directory = root / benchmark
    print(f"Checking {benchmark} inputs and runtime", flush=True)
    try:
        inputs = await asyncio.to_thread(load_dataset, benchmark, options.data_root, directory, config)
        if options.smoke:
            inputs.splits = {name: samples[: 4 if name == "train" else 2] for name, samples in inputs.splits.items()}
        await _check_runtime(inputs)
    except Exception as exc:
        errors.append(f"{benchmark}: {type(exc).__name__}: {exc}")
    if errors:
        write_json(root / "diagnostics.json", {"errors": errors})
        raise ValueError("Evaluation preflight failed:\n" + "\n".join(errors))

    write_json(
        root / "config.json",
        {
            "config": redact_credentials(config.model_dump(mode="json")),
            "options": options,
            "target_model": config.models.target.model,
            "optimizer_model": config.models.optimizer.model,
            "actual_forest": redact_credentials(config.forest(options.smoke).model_dump(mode="json")),
            "inputs": {
                benchmark: {
                    "source": str(inputs.source),
                    "split_file": str(inputs.split_file),
                    "initial_skill": str(inputs.initial_skill),
                    "environment": redact_credentials(inputs.environment.model_dump(mode="json")),
                    "splits": {name: [sample.task.id for sample in samples] for name, samples in inputs.splits.items()},
                }
            },
        },
    )
    report = BenchmarkReport(benchmark=benchmark)
    results = SuiteReport(output_dir=root, smoke=options.smoke, benchmarks=[report])
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "initial_skill.json", Skill.from_path(inputs.initial_skill).snapshot())
    print(f"Running {benchmark}: {', '.join(options.arms)}", flush=True)
    try:
        await _run_skill_forest(config, options, inputs, directory, report)
    except Exception as exc:
        report.error = f"{type(exc).__name__}: {exc}"
    finally:
        write_json(directory / "results.json", report)
        write_report(results, root)
    return results


def rebuild_report(run_dir: Path) -> SuiteReport:
    """Rebuild the Skill-Forest report from saved rollout results.

    Args:
        run_dir: Directory containing the original evaluation outputs.

    Returns:
        The results used to regenerate the report.
    """
    report = read_report(run_dir)
    write_report(report, report.output_dir)
    return report


async def _check_runtime(inputs: DatasetInputs) -> None:
    """Check the real selected runtime without calling the target model."""
    spec = inputs.environment
    if spec.class_name == "alfworld_bounded_history":
        if find_spec("alfworld") is None:
            raise RuntimeError("ALFWorld requires mindmemos-skill[alfworld]")
        sample = inputs.splits["test"][0]
        if not isinstance(sample.task, ALFWorldTask) or not isinstance(spec.config, ALFWorldConfig):
            raise TypeError("Expected native ALFWorld task and configuration")
        simulator = await Simulator.create(spec.config, sample.task.gamefile)
        try:
            await simulator.reset()
        finally:
            await simulator.close()
    elif spec.class_name == "spreadsheetbench_tool_calling":
        env = Environment.build(spec.class_name, spec.config)
        if not isinstance(env, SpreadsheetBenchToolCallingEnvironment):
            raise TypeError("Expected native SpreadsheetBench tool-calling environment")
        await env.initialize()
        if env.config.transactional_recalculation:
            _find_soffice()


def _seed(inputs: DatasetInputs, config: EvalConfig) -> int | None:
    """Record simulator seed for ALFWorld and provider sampling seed otherwise."""
    if isinstance(inputs.environment.config, ALFWorldConfig):
        return inputs.environment.config.seed
    return config.models.target.seed


async def _run_skill_forest(
    config: EvalConfig,
    options: RunOptions,
    inputs: DatasetInputs,
    directory: Path,
    report: BenchmarkReport,
) -> None:
    """Train once and evaluate only the requested control conditions."""
    env = Environment.build(inputs.environment.class_name, inputs.environment.config)
    initial_skill = Skill.from_path(inputs.initial_skill)
    test = inputs.splits["test"]
    ids = [sample.task.id for sample in test]
    for arm in options.arms:
        for repeat in range(1, options.test_repeats + 1):
            report.arms.append(ArmResult(arm=arm, repeat=repeat, task_ids=ids, seed=_seed(inputs, config)))
    for arm in report.arms:
        if arm.arm == "trained_skill":
            continue
        skills = None if arm.arm == "no_skill" else SkillSet({"selected": initial_skill})
        start = perf_counter()
        arm.cases = await run_tasks(
            env,
            test,
            skills,
            directory / arm.arm / f"repeat-{arm.repeat:03d}",
            config.evaluation_concurrency,
        )
        arm.wall_seconds = perf_counter() - start
        write_json(directory / "results.json", report)
    if "trained_skill" not in options.arms:
        return

    project_id = f"eval-{uuid4().hex}"
    service_config = ServiceConfig(
        home=directory / "state",
        training_pipelines={
            "skill_forest": ImplementationConfig(
                class_name="training",
                config={
                    "trainer": {"class_name": "skill_forest", "config": config.forest(options.smoke)},
                    "evaluation_concurrency": config.evaluation_concurrency,
                },
            )
        },
    )
    async with open_skill_service(service_config, project_id) as service:
        registered = await service.register_from_local(
            RegisterFromLocalRequest(
                project_id=project_id,
                skill=FileSkillInfo(path=str(inputs.initial_skill)),
            )
        )
        report.initial_version_id = registered.skill_version_id
        initial = await service.get_skill_version(registered.skill_version_id, project_id=project_id)
        if snapshot_from_version(initial) != initial_skill.snapshot():
            raise ValueError("Registered initial Skill differs from the frozen control Skill")
        write_json(directory / "initial_skill.json", initial)
        start = perf_counter()
        result = await service.run_training(
            TrainingRequest(
                project_id=project_id,
                algo="skill_forest",
                env=inputs.environment,
                init_skill_version_id=registered.skill_version_id,
                work_dir=str(directory / "training"),
                tasks=inputs.splits["train"] + inputs.splits["val"] + test,
                split=TrainingSplit(
                    train=[sample.task.id for sample in inputs.splits["train"]],
                    val=[sample.task.id for sample in inputs.splits["val"]],
                    test=ids,
                ),
            )
        )
        report.training_seconds = perf_counter() - start
        published = await service.get_skill_version(result.version_id, project_id=project_id)
        if published.skill_id != initial.skill_id or result.base_version_id != initial.version_id:
            raise ValueError("Trained publication does not descend from the registered initial version")
        report.trained_version_id = published.version_id
        write_json(directory / "trained_skill.json", published)
        snapshot = snapshot_from_version(published)
        skills = SkillSet({"selected": load_runtime_skill(snapshot)})
        for arm in report.arms:
            if arm.arm != "trained_skill":
                continue
            if arm.repeat == 1:
                arm.cases = read_pipeline_test(directory / "training/test", ids, snapshot, env, test)
            else:
                start = perf_counter()
                arm.cases = await run_tasks(
                    env,
                    test,
                    skills,
                    directory / "trained_skill" / f"repeat-{arm.repeat:03d}",
                    config.evaluation_concurrency,
                )
                arm.wall_seconds = perf_counter() - start
            write_json(directory / "results.json", report)


def _livemath(root: Path, resources: Path, run_dir: Path) -> tuple[Data, dict[str, JsonValue]]:
    """Read monthly records directly from the official download."""
    return Data.build("livemath", {"data_path": root / "data"}), {}


def _alfworld(root: Path, resources: Path, run_dir: Path) -> tuple[Data, dict[str, JsonValue]]:
    """Resolve upstream simulator paths without modifying its downloaded YAML."""
    simulator = yaml.safe_load(
        (root / "configs/base_config.yaml").read_text(encoding="utf-8").replace("$ALFWORLD_DATA", str(root))
    )
    return Data.build(
        "alfworld_bounded_history", {"data_path": resources / "data_splits/games.json", "game_root": root}
    ), {"simulator": simulator}


def _spreadsheetbench(root: Path, resources: Path, run_dir: Path) -> tuple[Data, dict[str, JsonValue]]:
    """Use Verified 400 metadata and private workbook references in place."""
    release = root / "spreadsheetbench_verified_400"
    return Data.build("spreadsheetbench", {"data_path": release / "dataset.json", "data_root": release}), {}


def _officeqa(root: Path, resources: Path, run_dir: Path) -> tuple[Data, dict[str, JsonValue]]:
    """Use the official CSV and nested parsed document corpus."""
    return Data.build("officeqa", {"data_path": root / "officeqa_full.csv", "docs_root": root}), {}


def _searchqa(root: Path, resources: Path, run_dir: Path) -> tuple[Data, dict[str, JsonValue]]:
    """Select pinned HF records using their native keys without changing input text."""
    splits = read_json(resources / "data_splits/splits.json")
    expected_ids = [identifier for ids in splits.values() for identifier in ids]
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("SearchQA splits contain duplicate task IDs")
    wanted = set(expected_ids)
    records = {}
    paths = sorted((root / "data").glob("*.parquet"))
    if not paths:
        raise ValueError(f"No SearchQA parquet files found under {root / 'data'}")
    for path in paths:
        for batch in parquet.ParquetFile(path).iter_batches(columns=["key", "question", "context", "answers"]):
            for row in batch.to_pylist():
                identifier = row["key"]
                if identifier not in wanted:
                    continue
                if identifier in records:
                    raise ValueError(f"Duplicate SearchQA source key: {identifier}")
                records[identifier] = {
                    "id": identifier,
                    "question": row["question"],
                    "context": row["context"],
                    "answers": row["answers"],
                }
    missing = wanted - records.keys()
    if missing:
        raise ValueError(f"SearchQA source is missing {len(missing)} split IDs: {sorted(missing)[:5]}")
    destination = run_dir / "prepared/searchqa/items.json"
    write_json(destination, [records[identifier] for identifier in expected_ids])
    return Data.build("searchqa", {"data_path": destination}), {}


def load_dataset(benchmark: BenchmarkName, data_root: Path, run_dir: Path, config: EvalConfig) -> DatasetInputs:
    """Load, verify, and split one official benchmark without rewriting downloads.

    Args:
        benchmark: One of the five explicitly supported benchmarks.
        data_root: Download parent directory.
        run_dir: Benchmark output directory for necessary prepared data.
        config: Shared target model and native environment overrides.

    Returns:
        Complete ordered train/val/test pools and native environment options.

    Raises:
        ValueError: Receipt, membership, references, or source records are invalid.
        OSError: Required data or resources are missing.
    """
    root = (data_root / benchmark).expanduser().resolve()
    resources = ROOT / f"resources/mindmemos_skill_eval/skill_forest-{benchmark}"
    loaders = {
        "livemath": _livemath,
        "alfworld": _alfworld,
        "spreadsheetbench": _spreadsheetbench,
        "searchqa": _searchqa,
        "officeqa": _officeqa,
    }
    data, prepared = loaders[benchmark](root, resources, run_dir)
    verification = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/download_datasets.py"),
            "download",
            "skill",
            "--name",
            benchmark,
            "--data-root",
            str(data_root.expanduser().resolve()),
            "--verify-only",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if verification.returncode:
        raise ValueError(
            f"Download receipt verification failed: {verification.stdout.strip()} {verification.stderr.strip()}"
        )
    split_file = resources / "data_splits/splits.json"
    splits = data.split(data.load(), split_file=split_file)
    if set(splits) != {"train", "val", "test"} or len(splits["train"]) < 2 or not splits["test"]:
        raise ValueError("Fixed splits require train, val, test and at least two train samples and one test sample")
    initial_skill = resources / "initial_skill"
    Skill.from_path(initial_skill)
    return DatasetInputs(
        splits=splits,
        environment=config.environment(benchmark, prepared),
        initial_skill=initial_skill,
        split_file=split_file,
        source=root,
    )


def read_pipeline_test(
    directory: Path,
    expected_task_ids: list[str],
    snapshot: SkillSnapshot,
    env: Environment,
    samples: Sequence[DataSample],
) -> list[CaseResult]:
    """Verify pipeline task order, raw outcomes, and the exact frozen Skill.

    Args:
        directory: Training pipeline test directory.
        expected_task_ids: Requested test IDs in fixed manifest order.
        snapshot: Exact published Skill snapshot used for trained repetitions.
        env: Native environment whose configuration must match the pipeline test.
        samples: Original ordered test inputs used to verify checkpoint identity.

    Returns:
        Outcomes adopted from the first pipeline test without calling any models.

    Raises:
        ValueError: Task order, completion, metadata, or frozen version disagrees.
        OSError: Required pipeline output is missing.
    """
    inputs = read_boundary(directory / "inputs.json")
    if read_json(directory.parent / "selected_skill.json") != snapshot.model_dump(mode="json"):
        raise ValueError("Pipeline test frozen snapshot disagrees with published version")
    identity = {
        "skill": snapshot.model_dump(mode="json"),
        "environment": f"{type(env).__module__}.{type(env).__qualname__}",
        "config": env.config.model_dump(mode="json"),
        "samples": [sample.model_dump(mode="json") for sample in samples],
    }
    if inputs.identity != fingerprint(identity):
        raise ValueError("Pipeline test inputs disagree with requested environment, tasks, or frozen version")
    index = HeldOutResultIndex.model_validate(read_json(directory / "results.json"))
    if [task.task_id for task in index.tasks] != expected_task_ids:
        raise ValueError("Pipeline test index disagrees with requested task order")
    cases = []
    for task in index.tasks:
        path = (directory / task.result_path).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError("Pipeline result path escapes its test directory")
        completed = read_boundary(path)
        if completed.identity != inputs.identity:
            raise ValueError("Pipeline completion belongs to different evaluation inputs")
        result = RolloutResult.model_validate(completed.payload["result"])
        case = case_from_result(task.task_id, result, path)
        if (case.score, case.success, case.duration_s, case.usage or {}) != (
            task.score,
            task.success,
            task.duration_s,
            task.usage,
        ):
            raise ValueError(f"Pipeline index disagrees with completed task {task.task_id}")
        cases.append(case)
    summary = read_json(directory / "summary.json")
    if summary != {
        "count": len(cases),
        "mean_score": sum(case.score for case in cases) / len(cases),
        "success_rate": sum(case.success for case in cases) / len(cases),
    }:
        raise ValueError("Pipeline summary disagrees with completed outcomes")
    return cases


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
