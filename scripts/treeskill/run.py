"""Compare initial and evolved TreeSkill on identical held-out SpreadsheetBench tasks."""

import argparse
import asyncio
import os
import signal
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import httpx
import yaml
from filelock import FileLock
from mindmemos_sdk.config import ConfigManager
from mindmemos_sdk.config.constants import MINDMEMOS_HOME_ENV
from mindmemos_sdk.skills.async_client import AsyncSkillClient
from mindmemos_sdk.skills.config import EnvironmentConfig
from mindmemos_sdk.skills.typing import (
    ListRequest,
    RegisterRequest,
    ShowRequest,
    StructuredSkillInfo,
    TaskProgressRequest,
    TaskShowRequest,
    TrainingRequest,
    TrainingSplitRequest,
)
from mindmemos_sdk.transport import AsyncHttpTransport
from mindmemos_skill.agents import OpenAIChatAgent
from mindmemos_skill.algos.train.treeskill.config import TreeSkillTrainerConfig
from mindmemos_skill.common.checkpoint import fingerprint
from mindmemos_skill.common.input_files import model_file_fingerprints
from mindmemos_skill.common.json import append_jsonl, read_json, write_json
from mindmemos_skill.common.skill_runtime import load_runtime_skill
from mindmemos_skill.envs import Environment
from mindmemos_skill.envs.data import Data
from mindmemos_skill.skill import Skill, SkillSet, SkillSnapshot

from scripts.treeskill.report import build_comparison


def _event(root: Path, event: str, **details) -> None:
    """Append one progress event and show its non-sensitive fields."""
    row = {"time": datetime.now(UTC).isoformat(), "event": event, **details}
    append_jsonl(root / "events.jsonl", row)
    print(row, flush=True)


def _check_recalculation(directory: Path, timeout: float = 120) -> dict:
    """Bound the entire UNO probe, including child cleanup that can hang on macOS."""
    directory.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-c",
        "import sys; from pathlib import Path; "
        "from mindmemos_skill.envs.builtin_envs.spreadsheetbench.recalculation "
        "import preflight_recalculation_runtime; "
        "preflight_recalculation_runtime(Path(sys.argv[1]))",
        str(directory),
    ]
    log = directory / "probe.log"
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=stream, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
            raise RuntimeError(f"LibreOffice/UNO preflight exceeded {timeout}s; inspect {log}") from None
    if code:
        raise RuntimeError(f"LibreOffice/UNO preflight failed; inspect {log}")
    return read_json(directory / "status.json")


async def _evaluate(*, env, skill, samples, root: Path, phase: str, concurrency: int) -> dict:
    """Gate on one real rollout, then collect independent results with explicit failures."""
    directory = root / phase
    items, failures = [], []
    started = perf_counter()
    semaphore = asyncio.Semaphore(concurrency)
    skills = SkillSet({"selected": skill}, max_concurrency=1)

    def save():
        count = len(items)
        summary = {
            "expected": len(samples),
            "completed": count,
            "failed": len(failures),
            "skipped": len(samples) - count - len(failures),
            "complete": count == len(samples),
            "score_sum": sum(item["score"] for item in items),
            "mean_score": sum(item["score"] for item in items) / count if count else None,
            "success_count": sum(item["success"] for item in items),
            "success_rate": sum(item["success"] for item in items) / count if count else None,
            "duration_s": perf_counter() - started,
            "usage": {
                key: sum(item["usage"].get(key, 0) for item in items)
                for key in ("input_tokens", "output_tokens", "total_tokens")
            },
            "items": items,
            "failures": failures,
        }
        write_json(directory / "summary.json", summary)
        return summary

    async def execute(sample):
        async with semaphore:
            task_root = directory / sample.task.id
            result_path = task_root / "audit" / "result.json"
            error_path = task_root / "error.json"
            if error_path.exists() and not result_path.exists():
                failures.append(read_json(error_path))
                save()
                return False
            _event(root, "rollout_started", phase=phase, task_id=sample.task.id, reused=result_path.exists())
            try:
                result = await env.rollout(
                    sample.task,
                    task_root,
                    reward_data=sample.reward_data,
                    skill_set=skills,
                )
                if not any(item.get("role") == "assistant" for item in result.trajectory):
                    raise RuntimeError("Rollout contains no real assistant response")
            except Exception as exc:
                failure = {"task_id": sample.task.id, "error_type": type(exc).__name__, "error": str(exc)}
                failures.append(failure)
                write_json(task_root / "error.json", failure)
                _event(root, "rollout_failed", phase=phase, task_id=sample.task.id, error_type=type(exc).__name__)
                save()
                return False
            row = {
                "task_id": sample.task.id,
                "score": result.reward.score,
                "success": result.reward.success,
                "duration_s": result.duration_s,
                "usage": result.usage.model_dump(mode="json"),
                "result_path": str(result_path),
            }
            items.append(row)
            _event(root, "rollout_completed", phase=phase, task_id=sample.task.id, score=row["score"])
            save()
            return True

    if not await execute(samples[0]):
        raise RuntimeError(f"First {phase} rollout failed; batch was not started")
    await asyncio.gather(*(execute(sample) for sample in samples[1:]))
    return save()


async def _train(client: AsyncSkillClient, *, root: Path, skill: Skill, samples, env_config) -> Skill:
    """Submit only training data to the existing service and preserve progress events."""
    accepted_path = root / "training_task.json"
    if accepted_path.exists():
        accepted = read_json(accepted_path)
    else:
        registration_path = root / "registration.json"
        if registration_path.exists():
            registered = read_json(registration_path)
        else:
            source = skill.snapshot()
            registered = await client.register(
                RegisterRequest(
                    alias=f"treeskill-{fingerprint(str(root))[:20]}",
                    skill=StructuredSkillInfo(
                        content=source.content,
                        resources=source.resources,
                        name=source.name,
                        description=source.description,
                        standard_format=source.standard_format,
                    ),
                )
            )
            write_json(registration_path, registered)
        accepted = await client.train(
            TrainingRequest(
                init_skill_version_id=registered["skill_version_id"],
                work_dir=str(root / "training"),
                tasks=[sample.model_dump(mode="json") for sample in samples],
                split=TrainingSplitRequest(train=[sample.task.id for sample in samples]),
                algo="treeskill",
                env=EnvironmentConfig(class_name="spreadsheetbench_text_action", config=env_config),
            )
        )
        write_json(accepted_path, accepted)
    task_id = accepted["task_id"]
    _event(root, "training_submitted", task_id=task_id)
    after = 0
    while True:
        progress = await client.task_progress(TaskProgressRequest(task_id=task_id, after=after))
        for event in progress:
            append_jsonl(root / "service_progress.jsonl", event)
            after = max(after, event["sequence"])
            _event(root, "training_progress", stage=event["stage"], message=event["message"])
        state = await client.show_task(TaskShowRequest(task_id=task_id))
        write_json(root / "training_status.json", state)
        build_comparison(root)
        if state["status"] == "succeeded":
            detail = await client.show(ShowRequest(skill_version_id=state["result"]["version_id"]))
            version = detail["version"]
            snapshot = SkillSnapshot.model_validate({key: version[key] for key in SkillSnapshot.model_fields})
            if snapshot.implementation != "treeskill":
                raise ValueError("Service returned a Skill without TreeSkill routing")
            write_json(root / "evolved_skill.json", snapshot)
            final = load_runtime_skill(snapshot)
            final.export(root / "evolved_skill", overwrite=True)
            return final
        if state["status"] in {"failed", "cancelled"}:
            raise RuntimeError(f"Training task {task_id} ended as {state['status']}: {state.get('error_type')}")
        await asyncio.sleep(30)


async def _run(args) -> None:
    """Validate inputs, probe models, and execute the three ordered phases."""
    root = args.output.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.environ[MINDMEMOS_HOME_ENV] = str(args.sdk_home.resolve())
    document = yaml.safe_load(args.service_config.read_text(encoding="utf-8"))
    pipeline = document["service"]["training_pipelines"]["treeskill"]["config"]
    algorithm = TreeSkillTrainerConfig.model_validate(pipeline["trainer"]["config"])
    if algorithm.samples_per_task != 1 or algorithm.max_trajectories < args.train_count:
        raise ValueError("This comparison requires one trajectory per task and a sufficient evidence budget")
    for connection in (algorithm.optimizer, algorithm.router, algorithm.failure_optimizer or algorithm.optimizer):
        connection.require_connection()
    data_root = args.data_root.expanduser().resolve()
    data = Data.build("spreadsheetbench", {"data_path": data_root / "dataset.json", "data_root": data_root})
    split = data.split(data.load(), split_file=args.split_file.resolve())
    train, test = split["train"][: args.train_count], split["test"][: args.test_count]
    if split["val"] or len(train) != args.train_count or len(test) != args.test_count:
        raise ValueError("Expected an empty validation split and exactly the requested train/test counts")
    skill = Skill.from_path(args.initial_skill.resolve())
    env_config = {
        "agent": algorithm.optimizer.model_dump(mode="json"),
        "max_turns": args.max_turns,
        "transactional_recalculation": True,
    }
    sdk = ConfigManager(args.sdk_home).load()
    url, key = sdk.resolve_skill_connection()
    protocol = {
        "train_count": len(train),
        "test_count": len(test),
        "samples_per_task": 1,
        "test_repeats": 1,
        "rollout_concurrency": algorithm.rollout_concurrency,
        "analysis_concurrency": algorithm.analysis_concurrency,
        "localization_concurrency": algorithm.localization_concurrency,
        "test_concurrency": args.concurrency,
        "fusion_concurrency": 1,
        "experiment_concurrency": 1,
        "model": algorithm.optimizer.model,
        "router_model": algorithm.router.model,
        "reasoning_effort": algorithm.optimizer.reasoning_effort,
        "max_turns": args.max_turns,
        "max_completion_tokens": algorithm.optimizer.max_completion_tokens,
        "model_seed": algorithm.optimizer.seed,
        "split_selection": "ordered_prefix",
        "service_url": url,
        "service_config": str(args.service_config.resolve()),
        "data_root": str(data_root),
        "initial_skill_hash": skill.snapshot().skill_hash,
        "selection": "one_round_without_test_selection",
    }
    identity = fingerprint(
        {
            "protocol": protocol,
            "algorithm": algorithm.model_dump(mode="json"),
            "input_identity": {
                sample.task.id: model_file_fingerprints(sample.task, sample.reward_data) for sample in [*train, *test]
            },
            "train": [sample.model_dump(mode="json") for sample in train],
            "test": [sample.model_dump(mode="json") for sample in test],
        }
    )
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest["identity"] != identity:
            raise ValueError("Run inputs changed; use a new output directory")
    else:
        if any(path.name != ".run.lock" for path in root.iterdir()):
            raise ValueError("Nonempty output has no comparison manifest")
        manifest = {
            "identity": identity,
            "status": "prepared",
            "protocol": protocol,
            "train_ids": [sample.task.id for sample in train],
            "test_ids": [sample.task.id for sample in test],
        }
        write_json(manifest_path, manifest)
    write_json(root / "split.json", {"train": manifest["train_ids"], "val": [], "test": manifest["test_ids"]})
    skill.export(root / "initial_skill", overwrite=True)
    _event(root, "parameters", **protocol)
    try:
        env = await asyncio.to_thread(Environment.build, "spreadsheetbench_text_action", env_config)
        runtime = await asyncio.to_thread(_check_recalculation, root / "preflight" / "recalculation")
    except Exception as exc:
        manifest.update(status="preflight_failed", error_type=type(exc).__name__)
        write_json(manifest_path, manifest)
        _event(root, "runtime_preflight_failed", error_type=type(exc).__name__)
        build_comparison(root)
        raise
    write_json(
        root / "preflight" / "runtime.json",
        {"python": env.python_runtime.model_dump(mode="json"), "recalculation": runtime},
    )
    if args.preflight_only:
        _event(root, "offline_preflight_passed")
        return
    try:
        async with httpx.AsyncClient(timeout=10) as health:
            response = await health.get(url.rstrip("/") + "/health")
            response.raise_for_status()
        async with AsyncHttpTransport(base_url=url, api_key=key, max_retries=0) as transport:
            client = AsyncSkillClient(transport)
            await client.list(ListRequest(limit=1))
            connections = {
                fingerprint(c.model_dump(mode="json")): c
                for c in (
                    algorithm.optimizer,
                    algorithm.router,
                    algorithm.failure_optimizer or algorithm.optimizer,
                )
            }
            for index, connection in enumerate(connections.values()):
                probe_config = connection.model_copy(update={"timeout_s": 60, "max_retries": 0})
                agent = OpenAIChatAgent(probe_config)
                probe = await asyncio.to_thread(
                    agent.run,
                    system="You are a connection check. Follow the user's instruction.",
                    user="Reply with OK only.",
                )
                if not isinstance(probe.output, str) or not probe.output.strip():
                    raise RuntimeError("Model probe returned an empty response")
                write_json(
                    root / "preflight" / f"model-{index}.json",
                    {
                        "model": connection.model,
                        "response": probe.output,
                        "usage": probe.usage,
                    },
                )
            _event(root, "online_preflight_passed")
            if args.probe_only:
                return
            for phase in ("baseline", "training", "evolved"):
                manifest["status"] = phase
                write_json(manifest_path, manifest)
                _event(root, "phase_started", phase=phase)
                if phase == "training":
                    final = await _train(client, root=root, skill=skill, samples=train, env_config=env_config)
                else:
                    await _evaluate(
                        env=env,
                        skill=skill if phase == "baseline" else final,
                        samples=test,
                        root=root,
                        phase=phase,
                        concurrency=args.concurrency,
                    )
                build_comparison(root)
            comparison = build_comparison(root)
            manifest["status"] = (
                "completed"
                if all(comparison["phases"][phase]["complete"] for phase in ("baseline", "evolved"))
                and comparison["phases"]["training"]["status"] == "completed"
                else "partial"
            )
    except BaseException as exc:
        manifest.update(status="failed", error_type=type(exc).__name__)
        _event(root, "experiment_failed", error_type=type(exc).__name__)
        raise
    finally:
        write_json(manifest_path, manifest)
        build_comparison(root)


def main() -> None:
    """Parse the comparison protocol and serialize ownership of its output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service-config", type=Path, default=Path("config/mindmemos_skill/dev.yaml"))
    parser.add_argument("--sdk-home", type=Path, default=Path(".mindmemos"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=30)
    parser.add_argument("--test-count", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument(
        "--split-file",
        type=Path,
        default=Path("resources/mindmemos_skill_eval/treeskill-spreadsheetbench/data_splits/splits.json"),
    )
    parser.add_argument(
        "--initial-skill",
        type=Path,
        default=Path("resources/mindmemos_skill_eval/treeskill-spreadsheetbench/initial_skill/SKILL.md"),
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--preflight-only", action="store_true")
    group.add_argument("--probe-only", action="store_true")
    args = parser.parse_args()
    if min(args.train_count, args.test_count, args.concurrency, args.max_turns) < 1:
        parser.error("counts, concurrency, and turn limit must be positive")
    args.output.mkdir(parents=True, exist_ok=True, mode=0o700)
    with FileLock(str(args.output / ".run.lock"), timeout=0):
        asyncio.run(_run(args))


if __name__ == "__main__":
    main()
