"""One-attempt rollouts, separately recoverable extraction, and patch export."""

import asyncio
import json
import os
from collections.abc import Callable
from contextlib import AsyncExitStack

from openai import AsyncOpenAI

from mindmemos_eval.backend import build_mindmemos_backend
from mindmemos_eval.llm import LLMClient, LLMConfig

from .agents import HierarchicalAgent, task_text
from .config import ExperimentConfig
from .constants import (
    CHILD_API_KEY_ENV,
    CHILD_BASE_URL_ENV,
    EXPERIENCE_MEMORY_KEY_ENV,
    MEMORY_BASE_URL_ENV,
    PARENT_API_KEY_ENV,
    PARENT_BASE_URL_ENV,
    PLAN_MEMORY_KEY_ENV,
)
from .data import append_jsonl, load_split, write_json
from .environment import TaskContainer
from .trajectory_view import PARENT_VIEW_FORMAT, build_parent_memory_view, parent_view_messages
from .typing import Task, Trajectory


def _required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


async def _memories(config: ExperimentConfig, stack: AsyncExitStack) -> dict:
    keys = {"parent": _required(PLAN_MEMORY_KEY_ENV), "child": _required(EXPERIENCE_MEMORY_KEY_ENV)}
    if keys["parent"] == keys["child"]:
        raise ValueError("The two memory roles must use different project API keys")
    clients = {}
    for role, project in (("parent", config.memory.plan_project_id), ("child", config.memory.experience_project_id)):
        backend = build_mindmemos_backend(
            connection_mode="http",
            project_id=project,
            base_url=_required(MEMORY_BASE_URL_ENV),
            api_key=keys[role],
            timeout_seconds=config.memory.timeout_seconds,
            max_retries=0,
        )
        clients[role] = (await stack.enter_async_context(backend)).memory
    return clients


async def _models(config: ExperimentConfig, stack: AsyncExitStack) -> dict:
    clients = {}
    for role, settings, key_env, url_env in (
        ("parent", config.parent, PARENT_API_KEY_ENV, PARENT_BASE_URL_ENV),
        ("child", config.child, CHILD_API_KEY_ENV, CHILD_BASE_URL_ENV),
    ):
        provider = await stack.enter_async_context(
            AsyncOpenAI(
                api_key=_required(key_env),
                base_url=_required(url_env),
                timeout=settings.timeout_seconds,
                max_retries=0,
            )
        )
        clients[role] = LLMClient(
            LLMConfig(
                model=settings.model,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
                timeout=settings.timeout_seconds,
                max_retries=0,
                extra={"extra_body": {"thinking": {"type": settings.thinking}}} if settings.thinking else {},
            ),
            client=provider,
        )
    return clients


def _training_ready(config: ExperimentConfig, tasks: list[Task]) -> None:
    totals = {"parent": 0, "child": 0}
    for task in tasks:
        directory = config.output_dir / "train" / task.instance_id
        for role in totals:
            receipt = json.loads((directory / f"extraction-{role}.json").read_text(encoding="utf-8"))
            if receipt["state"] != "completed":
                raise ValueError(f"Extraction incomplete: {task.instance_id}/{role}")
            totals[role] += receipt["memory_count"]
    if not all(totals.values()):
        raise ValueError("Each role must have extracted memories before memory-augmented testing")


async def run_rollouts(
    config: ExperimentConfig,
    phase: str,
    on_result: Callable[[], None] | None = None,
) -> None:
    """Run each task once, then persist traces and official-format patch predictions.

    Args:
        config: Frozen experiment settings.
        phase: Train or baseline without retrieval, or test with role-specific retrieval.
        on_result: Optional synchronous callback after each persisted task result.
    """
    if phase not in {"train", "test", "baseline"}:
        raise ValueError("Unknown rollout phase")
    split = load_split(config)
    if phase == "test":
        _training_ready(config, split.train)
    tasks = split.train if phase == "train" else split.test
    images = json.loads(config.image_map_path.read_text(encoding="utf-8"))
    if any(not isinstance(images.get(task.instance_id), str) or not images[task.instance_id] for task in tasks):
        raise ValueError("Image map must contain a prepared local image for every task in this phase")
    directory = config.output_dir / phase
    directory.mkdir(exist_ok=True)
    # An interrupted attempt remains visible and is never automatically rerun.
    for task in tasks:
        attempt = directory / task.instance_id
        if attempt.exists() and not (attempt / "result.json").exists():
            raise RuntimeError(f"Interrupted attempt requires manual inspection; refusing rerollout: {attempt}")
    async with AsyncExitStack() as stack:
        clients = await _models(config, stack)
        memories = await _memories(config, stack) if phase == "test" else {}

        async def run_task(task: Task) -> None:
            attempt = directory / task.instance_id
            if attempt.exists():
                return
            print(f"START {phase} {task.instance_id}", flush=True)
            attempt.mkdir()
            write_json(attempt / "attempt.json", {"instance_id": task.instance_id, "rollout": 1})
            result = {"instance_id": task.instance_id, "state": "failed", "finished": False}
            try:
                async with TaskContainer(task, images[task.instance_id], config.docker) as container:
                    write_json(
                        attempt / "environment.json",
                        {
                            "image": images[task.instance_id],
                            "image_id": container.image_id,
                            "base_commit": task.base_commit,
                        },
                    )
                    agent = HierarchicalAgent(
                        config, task, container, clients, memories, attempt, retrieve=phase == "test"
                    )
                    trace = await agent.run()
                    write_json(attempt / "parent-memory-input.json", build_parent_memory_view(attempt))
                    patch = await container.patch()
                    (attempt / "model.patch").write_text(patch, encoding="utf-8")
                    result.update(state="completed", finished=trace.finished, child_invocations=agent.children)
            except Exception as exc:
                # Do not persist provider exception text, which may contain endpoint credentials.
                result["error_type"] = type(exc).__name__
                write_json(attempt / "result.json", result)
                if on_result:
                    on_result()
                raise
            write_json(attempt / "result.json", result)
            print(f"DONE {phase} {task.instance_id}", flush=True)
            if on_result:
                on_result()

        queue = asyncio.Queue()
        for task in tasks:
            queue.put_nowait(task)
        failures = []

        async def worker() -> None:
            while not failures:
                try:
                    task = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                try:
                    await run_task(task)
                except Exception as exc:
                    failures.append(exc)
                finally:
                    queue.task_done()

        await asyncio.gather(*(worker() for _ in range(config.task_concurrency)))
        if failures:
            raise failures[0]
    records = [json.loads((directory / task.instance_id / "result.json").read_text()) for task in tasks]
    predictions = []
    for task, result in zip(tasks, records, strict=True):
        if result["state"] == "completed":
            predictions.append(
                {
                    "instance_id": task.instance_id,
                    "model_name_or_path": f"hierarchical:{config.parent.model}+{config.child.model}",
                    "model_patch": (directory / task.instance_id / "model.patch").read_text(encoding="utf-8"),
                }
            )
    prediction_path = directory / "predictions.jsonl"
    prediction_path.write_text("", encoding="utf-8")
    for prediction in predictions:
        append_jsonl(prediction_path, prediction)
    write_json(
        directory / "summary.json",
        {
            "requested": len(tasks),
            "completed": len(predictions),
            "failed": sum(record["state"] == "failed" for record in records),
            "finished": sum(record["finished"] for record in records),
            "officially_graded": False,
        },
    )


async def extract_memories(config: ExperimentConfig) -> None:
    """Extract two role-specific task memories from each completed training rollout.

    Args:
        config: Frozen experiment settings pointing to existing local trajectories.

    Raises:
        RuntimeError: An earlier write has an uncertain outcome and needs reconciliation.
    """
    split = load_split(config)
    async with AsyncExitStack() as stack:
        memories = await _memories(config, stack)
        for task in split.train:
            directory = config.output_dir / "train" / task.instance_id
            result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
            if result["state"] != "completed":
                raise RuntimeError(f"Cannot extract a failed rollout: {task.instance_id}")
            for role in ("parent", "child"):
                receipt_path = directory / f"extraction-{role}.json"
                if receipt_path.exists():
                    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                    if receipt["state"] == "completed":
                        continue
                    raise RuntimeError(f"Uncertain memory write; reconcile before retrying: {receipt_path}")
                paths = [directory / "parent.json"] if role == "parent" else sorted(directory.glob("child-*.json"))
                if not paths:
                    raise RuntimeError(f"No {role} trajectories for {task.instance_id}")
                traces = [Trajectory.model_validate_json(path.read_text(encoding="utf-8")) for path in paths]
                messages = []
                for trace in traces:
                    messages.append(
                        {
                            "role": "user",
                            "content": json.dumps(
                                {
                                    "trajectory_id": trace.trajectory_id,
                                    "task": trace.task,
                                    "finished": trace.finished,
                                    "error": trace.error,
                                    "note": "No official grading verdict is available.",
                                },
                                ensure_ascii=False,
                            ),
                            "agent": trace.agent,
                        }
                    )
                    for message in trace.messages:
                        messages.append(
                            {
                                "role": message["role"],
                                "agent": message.get("agent", trace.agent),
                                "content": json.dumps(message, ensure_ascii=False),
                            }
                        )
                extract_type = "plan" if role == "parent" else "experience"
                metadata = {
                    "extract_type": extract_type,
                    "instance_id": task.instance_id,
                    "phase": "train",
                    "role": role,
                }
                if role == "parent":
                    view = build_parent_memory_view(directory)
                    write_json(directory / "parent-memory-input.json", view)
                    messages = parent_view_messages(view)
                    metadata["trajectory_view"] = PARENT_VIEW_FORMAT
                write_json(receipt_path, {"state": "pending", "extract_type": extract_type})
                response = await memories[role].add(
                    messages,
                    user_id=config.memory.user_id,
                    agent_id="main" if role == "parent" else "coding",
                    mode="sync",
                    session_id=task.instance_id,
                    task_id=task.instance_id,
                    task=task_text(task),
                    metadata=metadata,
                )
                if response.code != "ok":
                    raise RuntimeError(f"Extraction did not return ok: {receipt_path}")
                write_json(
                    receipt_path,
                    {
                        "state": "completed",
                        "extract_type": extract_type,
                        "memory_count": len(response.memories),
                        "response": response.model_dump(mode="json"),
                    },
                )
    write_json(
        config.output_dir / "extraction-summary.json",
        {
            "tasks": 50,
            "role_receipts": 100,
            "officially_graded": False,
        },
    )
