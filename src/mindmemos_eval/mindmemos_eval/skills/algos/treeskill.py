"""Train TreeSkill and export held-out evaluation results."""

import asyncio
import re
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from time import perf_counter

from mindmemos_skill.algos.train.treeskill import TreeSkillTrainerConfig
from mindmemos_skill.api.typing import RegisterRequest, StructuredSkillInfo, TrainingRequest, TrainingSplit
from mindmemos_skill.common.checkpoint import fingerprint, read_boundary
from mindmemos_skill.common.json import append_jsonl, read_json, write_json
from mindmemos_skill.common.secrets import redact_credentials
from mindmemos_skill.config.config import ImplementationConfig, ServiceConfig, TaskManagerConfig
from mindmemos_skill.config.loader import read_skill_config
from mindmemos_skill.envs import Environment
from mindmemos_skill.envs.config import EnvironmentConfig
from mindmemos_skill.envs.data import Data, DataSample
from mindmemos_skill.envs.data.typing import SplitManifest
from mindmemos_skill.pipelines.builtin_pipelines.training.config import TrainingPipelineConfig
from mindmemos_skill.skill import Skill

from ..config import ROOT, TreeSkillEvalConfig
from ..rollout import case_from_result
from ..service import open_skill_service
from ..typing import CaseResult, EnvTestSummary, TaskTrajectory


async def run(config: TreeSkillEvalConfig) -> EnvTestSummary:
    """Register the initial Skill and evaluate the trained Skill through a local service.

    Args:
        config: Local training profile and evaluation inputs.

    Returns:
        Held-out summary after the local task terminates.
    """
    env, pools, initial, source = prepare(config)
    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f")
    name = f"{initial.source.name}-mindmemos-eval-{stamp}"
    directory = (config.output_dir or ROOT / "output/mindmemos_eval/treeskill" / stamp).expanduser().resolve()
    if directory.exists() and any(directory.iterdir()):
        raise ValueError(f"Output directory must be empty: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    content = re.sub(r"(?m)^name:.*$", f"name: {name}", initial.source.content, count=1)
    renamed = initial.with_content(content, name=name)
    snapshot = renamed.snapshot()
    project_id = "mindmemos-eval"
    resources = ServiceConfig(
        home=directory / "state",
        agents=source.agents,
        task_manager=TaskManagerConfig(mode="local"),
        training_pipelines={config.profile: source.training_pipelines[config.profile]},
    )
    write_json(directory / "config.json", redact_credentials(config.model_dump(mode="json")))
    write_json(directory / "split.json", {key: [s.task.id for s in values] for key, values in pools.items()})
    start = perf_counter()
    async with open_skill_service(resources, project_id) as service:
        registered = await service.register_skill(
            RegisterRequest(
                project_id=project_id,
                alias=name,
                skill=StructuredSkillInfo(
                    content=snapshot.content,
                    resources=snapshot.resources,
                    name=name,
                    description=snapshot.description,
                    metadata={
                        "evaluation": "mindmemos-eval",
                        "environment": config.environment.class_name,
                    },
                ),
            )
        )
        write_json(directory / "registration.json", registered)
        accepted = await service.submit_training(
            TrainingRequest(
                project_id=project_id,
                init_skill_version_id=registered.skill_version_id,
                work_dir=str(directory / "training"),
                tasks=[sample for phase in ("train", "val", "test") for sample in pools[phase]],
                split=TrainingSplit(
                    train=[s.task.id for s in pools["train"]],
                    val=[s.task.id for s in pools["val"]],
                    test=[s.task.id for s in pools["test"]],
                ),
                algo=config.profile,
                env=EnvironmentConfig(class_name=config.environment.class_name, config=env.config),
            )
        )
        write_json(directory / "task.json", accepted)
        print(f"Registered {name}; local task {accepted.task_id}", flush=True)
        after = 0
        while True:
            state = await service.task_status(project_id, accepted.task_id)
            if state is None:
                raise RuntimeError(f"Local training task disappeared: {accepted.task_id}")
            logs = await service.task_progress(project_id, accepted.task_id, after=after)
            for event in logs:
                await append_jsonl(directory / "service_progress.jsonl", event)
                after = max(after, event.sequence)
            write_json(directory / "task.json", state)
            if state.status in {"succeeded", "failed", "cancelled"}:
                break
            await asyncio.sleep(config.poll_interval)
    write_json(directory / "runtime.json", {"pipeline_wall_seconds": perf_counter() - start})
    timing = directory / "training/test/report.json"
    seconds = read_json(timing)["wall_seconds"] if timing.exists() else None
    summary = await export_test(directory, pools["test"], env, seconds, config.environment.class_name)
    if state.status != "succeeded":
        raise RuntimeError(f"Local task {accepted.task_id} ended as {state.status}; see {directory / 'task.json'}")
    return summary


def prepare(config: TreeSkillEvalConfig) -> tuple[Environment, dict[str, list[DataSample]], Skill, ServiceConfig]:
    """Validate local inputs and assemble the training and test splits.

    Args:
        config: Evaluation inputs, environment settings, and training profile selection.

    Returns:
        Native environment, ordered samples by split, initial Skill, and validated service config.

    Raises:
        ValueError: The profile, dataset, split, or initial Skill is invalid.
    """
    source = read_skill_config(config.service_config).service
    profile = source.training_pipelines.get(config.profile)
    if (
        profile is None
        or not isinstance(profile.config, TrainingPipelineConfig)
        or not isinstance(profile.config.trainer.config, TreeSkillTrainerConfig)
    ):
        raise ValueError(f"Missing TreeSkill service profile: {config.profile}")

    environment = config.environment
    options = environment.config
    data = config.data
    if environment.class_name == "spreadsheetbench_text_action":
        if config.data_root is None:
            raise ValueError("SpreadsheetBench requires data_root")
        algorithm = profile.config.trainer.config
        workers = {
            algorithm.rollout_concurrency,
            algorithm.analysis_concurrency,
            algorithm.localization_concurrency,
            profile.config.evaluation_concurrency,
        }
        if workers != {config.concurrency}:
            raise ValueError(
                "Service rollout, analysis, localization and evaluation concurrency must match concurrency"
            )
        policy = (algorithm.failure_optimizer or algorithm.optimizer).model_copy(
            update={"temperature": 0.7, "stop": ["Observation:"]}
        )
        options = {"agent": policy.model_dump(mode="json")}
        options.update(environment.config)
        data = ImplementationConfig(
            class_name="spreadsheetbench",
            config={"data_path": config.data_root / "dataset.json", "data_root": config.data_root},
        )

    env = Environment.build(environment.class_name, options)
    samples = _load_samples(data, config.samples_file, env)
    split = SplitManifest.model_validate(read_json(config.split_file.expanduser())).root
    index = {sample.task.id: sample for sample in samples}
    if set(split) - {"train", "val", "test"}:
        raise ValueError("TreeSkill supports only train, val and test splits")
    if any(identifier not in index for ids in split.values() for identifier in ids):
        raise ValueError("Split contains task IDs absent from the dataset")
    pools = {phase: [index[key] for key in split.get(phase, [])] for phase in ("train", "val", "test")}
    if not pools["train"] or not pools["test"]:
        raise ValueError("TreeSkill evaluation requires nonempty train and test splits")
    return env, pools, Skill.from_path(config.initial_skill.expanduser()), source


async def export_test(
    directory: Path, samples: Sequence[DataSample], env: Environment, wall_seconds: float | None, environment_name: str
) -> EnvTestSummary:
    """Export service test results using the required counts, token totals, and JSONL trajectory format.

    Args:
        directory: Evaluation root containing the server's training/test artifacts.
        samples: Exact ordered held-out tasks submitted to the server.
        env: Native scoring environment used to restore typed results.
        wall_seconds: Test-only elapsed time, or None if the service did not finish the phase.
        environment_name: Registered environment actually used for this run.

    Returns:
        Held-out summary; missing or failed tasks are explicitly recorded.
    """
    cases = []
    attempted = 0
    inputs_path = directory / "training/test/inputs.json"
    inputs = read_boundary(inputs_path) if inputs_path.exists() else None
    if inputs is not None:
        identity = {
            "skill": read_json(directory / "training/selected_skill.json"),
            "environment": f"{type(env).__module__}.{type(env).__qualname__}",
            "config": env.config.model_dump(mode="json"),
            "samples": [sample.model_dump(mode="json") for sample in samples],
        }
        if inputs.identity != fingerprint(identity):
            raise ValueError("Test checkpoint differs from the requested tasks, environment, or frozen Skill")
    index_path = directory / "training/test/results.json"
    if index_path.exists() and [row["task_id"] for row in read_json(index_path)["tasks"]] != [
        sample.task.id for sample in samples
    ]:
        raise ValueError("Test index differs from the requested task order")
    trace = directory / "trajectories.jsonl"
    # Export to a temporary file so a repeated report does not duplicate trajectories.
    temporary = directory / "trajectories.jsonl.tmp"
    temporary.unlink(missing_ok=True)
    for position, sample in enumerate(samples):
        path = directory / f"training/test/task-{position:06d}/completed.json"
        result = None
        attempts = path.parent / "attempts"
        ran = path.exists() or (attempts.exists() and any(attempts.iterdir()))
        attempted += int(ran)
        if path.exists():
            saved = read_boundary(path)
            if inputs is None or inputs.identity != saved.identity:
                raise ValueError(f"Test checkpoint does not match evaluation inputs: {path}")
            result = env.result_type.model_validate(saved.payload["result"])
            case = case_from_result(sample.task.id, result, path)
        else:
            case = CaseResult(
                task_id=sample.task.id,
                error="Service execution failed before publishing a result"
                if ran
                else "Not started: service stopped before this task",
                result_path=path,
            )
        cases.append(case)
        await append_jsonl(temporary, TaskTrajectory(position=position, task=sample.task, outcome=case, result=result))
    temporary.replace(trace)
    completed = [case for case in cases if case.error is None]
    known = [case.usage for case in cases if case.usage is not None]
    usage = {key: sum(item[key] for item in known) for key in ("input_tokens", "output_tokens", "total_tokens")}
    missing = attempted - len(known)
    summary = EnvTestSummary(
        environment=environment_name,
        mode="treeskill",
        planned=len(samples),
        attempted=attempted,
        not_started=len(samples) - attempted,
        completed=len(completed),
        errors=attempted - len(completed),
        successful=sum(case.success for case in completed),
        success_rate=sum(case.success for case in completed) / len(completed) if completed else None,
        wall_seconds=wall_seconds,
        input_tokens=usage["input_tokens"] if not missing else None,
        output_tokens=usage["output_tokens"] if not missing else None,
        total_tokens=usage["total_tokens"] if not missing else None,
        reported_usage=usage,
        usage_missing=missing,
        trajectory_file=trace,
    )
    write_json(directory / "summary.json", summary)
    write_json(directory / "results.json", cases)
    _write_report(summary, directory)
    return summary


def _load_samples(data: ImplementationConfig | None, samples_file: Path | None, env: Environment) -> list[DataSample]:
    """Load native TreeSkill samples before assigning training and held-out splits."""
    if (data is None) == (samples_file is None):
        raise ValueError("Provide exactly one of data or samples_file")
    if data is not None:
        loader = Data.build(data.class_name, data.config)
        samples = loader.load()
    else:
        rows = read_json(samples_file.expanduser())
        if not isinstance(rows, list) or any(
            not isinstance(row, dict) or set(row) != {"task", "reward_data"} for row in rows
        ):
            raise ValueError("samples_file must be an array of objects with task and reward_data fields")
        samples = [
            DataSample(
                task=env.task_type.model_validate(row["task"]),
                reward_data=env.reward_data_type.model_validate(row["reward_data"]),
            )
            for row in rows
        ]
    index = {}
    for sample in samples:
        if not isinstance(sample.task, env.task_type) or not isinstance(sample.reward_data, env.reward_data_type):
            raise ValueError("Dataset task or reward_data types do not match the selected environment")
        if sample.task.id in index:
            raise ValueError(f"Duplicate task ID: {sample.task.id}")
        index[sample.task.id] = sample
    if not samples:
        raise ValueError("No evaluation tasks selected")
    return samples


def _write_report(summary: EnvTestSummary, directory: Path) -> None:
    """Write TreeSkill held-out scores, execution coverage, and usage.

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
        f"# TreeSkill held-out evaluation\n\nEnvironment: {summary.environment}\n\nMode: {summary.mode}\n\n"
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
