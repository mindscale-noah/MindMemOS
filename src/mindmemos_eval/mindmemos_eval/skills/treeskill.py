"""Align TreeSkill to recorded SpreadsheetBench inputs and submit through the HTTP SDK."""

import asyncio
import re
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from time import perf_counter

import yaml
from mindmemos_sdk.config import ConfigManager
from mindmemos_sdk.skills import AsyncSkillClient
from mindmemos_sdk.skills.config import EnvironmentConfig as SDKEnvironmentConfig
from mindmemos_sdk.skills.typing import (
    RegisterRequest,
    StructuredSkillInfo,
    TaskProgressRequest,
    TaskShowRequest,
    TrainingRequest,
    TrainingSplitRequest,
)
from mindmemos_sdk.transport import AsyncHttpTransport
from mindmemos_skill.agents import AgentCatalog
from mindmemos_skill.algos.train.treeskill import TreeSkillTrainerConfig
from mindmemos_skill.common.checkpoint import fingerprint, read_boundary
from mindmemos_skill.common.json import append_jsonl, read_json, write_json
from mindmemos_skill.common.secrets import redact_credentials
from mindmemos_skill.envs import Environment
from mindmemos_skill.envs.data import Data, DataSample
from mindmemos_skill.envs.data.typing import SplitManifest
from mindmemos_skill.skill import Skill

from .config import ROOT, EnvTestConfig, TreeSkillEvalConfig
from .evaluation import _case
from .report import write_test_report
from .testing import _load_samples
from .typing import CaseResult, EnvTestSummary, TaskTrajectory


def _protocol(value):
    """Ignore only connection credentials when comparing recorded execution settings."""
    if isinstance(value, dict):
        return {key: _protocol(child) for key, child in value.items() if key not in {"api_key", "base_url"}}
    if isinstance(value, list):
        return [_protocol(child) for child in value]
    return value


def _reference_algorithm(reference: Path, optimizer_model: str) -> dict:
    """Restore recorded models and the owner-confirmed historical optimizer model."""
    algorithm = read_json(reference / "training/train/config.json")["algorithm"]
    manifest = read_json(reference / "manifest.json")["protocol"]
    router = read_json(reference / "training/train/final_skill.json")["config"]["router"]
    for role in ("optimizer", "failure_optimizer"):
        if algorithm[role]["model"] == "[REDACTED]":
            algorithm[role]["model"] = optimizer_model if role == "optimizer" else manifest["model"]
    if algorithm["router"]["model"] == "[REDACTED]":
        algorithm["router"]["model"] = router["model"]
    return algorithm


def _portable(value):
    """Compare dataset-relative workbook locations across local checkouts."""
    if isinstance(value, dict):
        return {key: _portable(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_portable(child) for child in value]
    if isinstance(value, str) and "/spreadsheetbench_verified_400/" in value:
        return value.split("/spreadsheetbench_verified_400/", 1)[1]
    return value


def prepare(config: TreeSkillEvalConfig) -> tuple[Environment, dict[str, list[DataSample]], Skill]:
    """Validate the reference protocol, dataset content, and service profile without model calls.

    Args:
        config: Recorded reference, local input paths, and HTTP profile selection.

    Returns:
        Native environment, ordered training/test samples, and frozen initial Skill.

    Raises:
        ValueError: Reference results or any execution input differ from the strict protocol.
    """
    document = yaml.safe_load(config.service_config.read_text())
    pipeline = document["service"]["training_pipelines"].get(config.profile)
    if pipeline is None or pipeline["config"]["trainer"]["class_name"] != "treeskill":
        raise ValueError(f"Missing TreeSkill service profile: {config.profile}")
    pipeline = pipeline["config"]
    if config.environment.class_name != "spreadsheetbench_text_action":
        env = Environment.build(config.environment.class_name, config.environment.config)
        samples = _load_samples(
            EnvTestConfig(
                environment=config.environment.model_dump(), data=config.data, samples_file=config.samples_file
            ),
            env,
        )
        split = SplitManifest.model_validate(read_json(config.split_file)).root
        index = {sample.task.id: sample for sample in samples}
        if any(identifier not in index for ids in split.values() for identifier in ids):
            raise ValueError("Split contains task IDs absent from the dataset")
        pools = {phase: [index[key] for key in split.get(phase, [])] for phase in ("train", "val", "test")}
        if not pools["train"] or not pools["test"]:
            raise ValueError("TreeSkill evaluation requires nonempty train and test splits")
        print(f"Environment {config.environment.class_name}: no reference score guarantee.", flush=True)
        return env, pools, Skill.from_path(config.initial_skill)
    if config.reference_run is None or config.data_root is None:
        raise ValueError("text_action alignment requires reference_run and data_root")
    reference = config.reference_run.expanduser().resolve()
    summary = read_json(reference / "training/test/summary.json")
    if summary != {"count": 200, "success_rate": 0.61, "mean_score": 0.61}:
        raise ValueError("Reference must be the completed 122/200 SpreadsheetBench test")
    split = read_json(config.split_file)
    if (
        split != read_json(reference / "split.json")
        or len(split["train"]) != 200
        or len(split["test"]) != 200
        or split["val"]
    ):
        raise ValueError("TreeSkill requires the exact reference 200/200 split and order")
    algorithm = TreeSkillTrainerConfig.model_validate(
        pipeline["trainer"]["config"], context=AgentCatalog(agents=document["service"].get("agents", {}))
    )
    expected = TreeSkillTrainerConfig.model_validate(_reference_algorithm(reference, config.reference_optimizer_model))
    if _protocol(algorithm.model_dump(mode="json")) != _protocol(expected.model_dump(mode="json")):
        raise ValueError("Service TreeSkill profile differs from the reference trainer/model parameters")
    if pipeline["evaluation_concurrency"] != 32:
        raise ValueError("Reference evaluation concurrency must be 32")
    requests = {}
    initial = Skill.from_path(config.initial_skill)
    for phase, pattern in (
        ("train", "training/train/rollouts/train/**/audit/request.json"),
        ("test", "training/test/**/audit/request.json"),
    ):
        for path in sorted(reference.glob(pattern)):
            request = read_json(path)
            identifier = request["task"]["data"]["id"]
            if (phase, identifier) in requests:
                raise ValueError(f"Ambiguous reference attempts for {phase}/{identifier}")
            requests[phase, identifier] = request
    first = requests["train", split["train"][0]]
    snapshot = initial.snapshot().model_dump(mode="json")
    if any(snapshot[key] != first["skills"][0][key] for key in ("content", "resources", "name", "description")):
        raise ValueError("Initial Skill differs from the reference full initial Skill")
    env_options = dict(first["config"])
    policy = (algorithm.failure_optimizer or algorithm.optimizer).model_copy(
        update={"temperature": 0.7, "stop": ["Observation:"]}
    )
    env_options["agent"] = policy.model_dump(mode="json")
    env_options.update(config.environment.config)
    env = Environment.build(config.environment.class_name, env_options)
    loader = Data.build(
        "spreadsheetbench", {"data_path": config.data_root / "dataset.json", "data_root": config.data_root}
    )
    pools = loader.split(loader.load(), split_file=config.split_file)
    for phase in ("train", "test"):
        for sample in pools[phase]:
            request = requests.get((phase, sample.task.id))
            if request is None:
                raise ValueError(f"Missing reference audit input for {phase}/{sample.task.id}")
            if _protocol(request["config"]) != _protocol(env.config.model_dump(mode="json")):
                raise ValueError(f"Environment protocol differs for {phase}/{sample.task.id}")
            if _portable(request["task"]["data"]) != _portable(sample.task.model_dump(mode="json")) or _portable(
                request["reward_data"]["data"]
            ) != _portable(sample.reward_data.model_dump(mode="json")):
                raise ValueError(f"Dataset metadata differs for {phase}/{sample.task.id}")
            recorded = {_portable(path): value for path, value in request["input_identity"].items()}
            current = {
                _portable(path): value for path, value in env.input_identity(sample.task, sample.reward_data).items()
            }
            if recorded != current:
                raise ValueError(f"Workbook content differs for {phase}/{sample.task.id}")
    return env, pools, initial


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
            case = _case(sample.task.id, result, path)
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
    write_test_report(summary, directory)
    return summary


async def run(config: TreeSkillEvalConfig) -> EnvTestSummary:
    """Register a named Skill and execute training/test as a persisted UI-visible service task.

    Args:
        config: Strict reference protocol and SDK/service paths on a shared local filesystem.

    Returns:
        Held-out summary after the HTTP task terminates.
    """
    env, pools, initial = prepare(config)
    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f")
    name = f"{initial.source.name}-mindmemos-eval-{stamp}"
    directory = (config.output_dir or ROOT / "output/mindmemos_eval/treeskill" / stamp).expanduser().resolve()
    if directory.exists() and any(directory.iterdir()):
        raise ValueError(f"Output directory must be empty: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    content = re.sub(r"(?m)^name:.*$", f"name: {name}", initial.source.content, count=1)
    renamed = initial.with_content(content, name=name)
    snapshot = renamed.snapshot()
    sdk = ConfigManager(config.sdk_home).load()
    url, api_key = sdk.resolve_skill_connection()
    write_json(directory / "config.json", redact_credentials(config.model_dump(mode="json")))
    write_json(directory / "split.json", {key: [s.task.id for s in values] for key, values in pools.items()})
    start = perf_counter()
    async with AsyncHttpTransport(
        base_url=url, api_key=api_key, timeout_seconds=sdk.network.timeout_seconds, max_retries=0, require_api_key=False
    ) as transport:
        client = AsyncSkillClient(transport)
        registered = await client.register(
            RegisterRequest(
                alias=name,
                skill=StructuredSkillInfo(
                    content=snapshot.content,
                    resources=snapshot.resources,
                    name=name,
                    description=snapshot.description,
                    metadata={
                        "evaluation": "mindmemos-eval",
                        "reference_run": str(config.reference_run),
                        "environment": config.environment.class_name,
                    },
                ),
            )
        )
        write_json(directory / "registration.json", registered)
        accepted = await client.train(
            TrainingRequest(
                init_skill_version_id=registered["skill_version_id"],
                work_dir=str(directory / "training"),
                tasks=[sample.model_dump(mode="json") for sample in pools["train"] + pools["val"] + pools["test"]],
                split=TrainingSplitRequest(
                    train=[s.task.id for s in pools["train"]],
                    val=[s.task.id for s in pools["val"]],
                    test=[s.task.id for s in pools["test"]],
                ),
                algo=config.profile,
                env=SDKEnvironmentConfig(
                    class_name=config.environment.class_name, config=env.config.model_dump(mode="json")
                ),
            )
        )
        write_json(directory / "task.json", accepted)
        print(f"Registered {name}; service task {accepted['task_id']}", flush=True)
        after = 0
        while True:
            logs = await client.task_progress(TaskProgressRequest(task_id=accepted["task_id"], after=after))
            for event in logs:
                await append_jsonl(directory / "service_progress.jsonl", event)
                after = max(after, event["sequence"])
            state = await client.show_task(TaskShowRequest(task_id=accepted["task_id"]))
            write_json(directory / "task.json", state)
            if state["status"] in {"succeeded", "failed", "cancelled"}:
                break
            await asyncio.sleep(config.poll_interval)
    write_json(directory / "runtime.json", {"pipeline_wall_seconds": perf_counter() - start})
    timing = directory / "training/test/report.json"
    seconds = read_json(timing)["wall_seconds"] if timing.exists() else None
    summary = await export_test(directory, pools["test"], env, seconds, config.environment.class_name)
    if state["status"] != "succeeded":
        raise RuntimeError(
            f"Service task {accepted['task_id']} ended as {state['status']}; see {directory / 'task.json'}"
        )
    if config.environment.class_name != "spreadsheetbench_text_action":
        return summary
    actual = read_json(directory / "training/train/config.json")["algorithm"]
    expected = read_json(config.reference_run / "training/train/config.json")["algorithm"]
    if _protocol(actual) != _protocol(expected):
        raise ValueError("Running service trainer differed from the reference; results are not aligned")
    return summary
