"""Run fresh in-process Skill-Forest evaluations with frozen control conditions."""

import asyncio
from datetime import datetime
from importlib.util import find_spec
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from mindmemos_skill.api import create_service
from mindmemos_skill.api.typing import FileSkillInfo, RegisterFromLocalRequest, TrainingRequest, TrainingSplit
from mindmemos_skill.common.json import write_json
from mindmemos_skill.common.secrets import redact_credentials
from mindmemos_skill.common.skill_runtime import load_runtime_skill
from mindmemos_skill.common.skill_versions import snapshot_from_version
from mindmemos_skill.config.config import (
    FastAPIConfig,
    HttpAuthConfig,
    ImplementationConfig,
    PersistenceConfig,
    ServiceConfig,
    SkillConfig,
)
from mindmemos_skill.envs import Environment
from mindmemos_skill.envs.builtin_envs.alfworld_bounded_history.config import ALFWorldConfig
from mindmemos_skill.envs.builtin_envs.alfworld_bounded_history.simulator import Simulator
from mindmemos_skill.envs.builtin_envs.alfworld_bounded_history.typing import ALFWorldTask
from mindmemos_skill.envs.builtin_envs.spreadsheetbench.recalculation import _find_soffice
from mindmemos_skill.envs.builtin_envs.spreadsheetbench.tool_call_mode.environment import (
    SpreadsheetBenchToolCallingEnvironment,
)
from mindmemos_skill.skill import Skill, SkillSet

from .config import ROOT, EvalConfig, RunOptions
from .data import load_dataset
from .evaluation import evaluate, read_pipeline_test
from .report import write_report
from .typing import ArmResult, BenchmarkName, BenchmarkReport, DatasetInputs, SuiteReport


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
        arm.cases = await evaluate(
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
    state = directory / "state"
    service_config = ServiceConfig(
        home=state,
        persistence=PersistenceConfig(
            database=ImplementationConfig(
                class_name="sqlite",
                config={"url": f"sqlite+aiosqlite:///{state / 'skills.db'}"},
            )
        ),
        repository_pipeline=ImplementationConfig(class_name="repository"),
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
    async with await create_service(
        SkillConfig(
            service=service_config,
            fastapi=FastAPIConfig(auth=HttpAuthConfig(mode="none", default_project_id=project_id)),
        )
    ) as service:
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
                arm.cases = await evaluate(
                    env,
                    test,
                    skills,
                    directory / "trained_skill" / f"repeat-{arm.repeat:03d}",
                    config.evaluation_concurrency,
                )
                arm.wall_seconds = perf_counter() - start
            write_json(directory / "results.json", report)


async def run(config: EvalConfig, options: RunOptions) -> SuiteReport:
    """Validate the whole selected suite before executing fresh benchmark runs.

    Args:
        config: Validated shared models, native forest, and environment settings.
        options: Supported command options including scope and requested conditions.

    Returns:
        Suite results with explicit failures; reports are written automatically.

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
            errors.append(f"models.{label} in config/mindmemos_eval/dev.yaml: {exc}")
    datasets: dict[BenchmarkName, DatasetInputs] = {}
    for benchmark in options.selected_benchmarks():
        print(f"Checking {benchmark} inputs and runtime", flush=True)
        try:
            inputs = await asyncio.to_thread(load_dataset, benchmark, options.data_root, root / benchmark, config)
            if options.smoke:
                inputs.splits = {
                    name: samples[: 4 if name == "train" else 2] for name, samples in inputs.splits.items()
                }
            await _check_runtime(inputs)
            datasets[benchmark] = inputs
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
                for benchmark, inputs in datasets.items()
            },
        },
    )
    suite = SuiteReport(
        output_dir=root,
        smoke=options.smoke,
        benchmarks=[BenchmarkReport(benchmark=name) for name in options.selected_benchmarks()],
    )
    for report in suite.benchmarks:
        directory = root / report.benchmark
        directory.mkdir(parents=True, exist_ok=True)
        write_json(
            directory / "initial_skill.json", Skill.from_path(datasets[report.benchmark].initial_skill).snapshot()
        )
        print(f"Running {report.benchmark}: {', '.join(options.arms)}", flush=True)
        try:
            await _run_skill_forest(config, options, datasets[report.benchmark], directory, report)
        except Exception as exc:
            report.error = f"{type(exc).__name__}: {exc}"
        finally:
            write_json(directory / "results.json", report)
            write_report(suite, root)
    return suite
