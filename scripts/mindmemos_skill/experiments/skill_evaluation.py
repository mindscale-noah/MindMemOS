#!/usr/bin/env python3
"""Script-side evaluation of a supplied Skill, or no Skill, on one test split."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, TextIO

from mindmemos_skill.agents.base import Agent
from mindmemos_skill.agents.react import ReactAgent
from mindmemos_skill.algos.evolve.skill_grpo_with_replay_buffer.config import RolloutConfig
from mindmemos_skill.algos.evolve.skill_grpo_with_replay_buffer.contracts import RolloutOutcome
from mindmemos_skill.algos.evolve.skill_grpo_with_replay_buffer.rollout import (
    FixedGroupPlan,
    FixedGroupRolloutStrategy,
    MappingAgentResolver,
    RegistryEnvFactory,
    RolloutScheduler,
)
from mindmemos_skill.datasets import (
    ALFWorldPathSplitDataset,
    LiveMathIdSplitDataset,
    SpreadsheetBenchIdSplitDataset,
    TaskDataset,
)
from mindmemos_skill.llm import LLMCallSink, LLMClient, get_router, llm_run_context
from mindmemos_skill.management import frontmatter_value
from mindmemos_skill.typing import Skill, Task, Trajectory, compute_skill_content_hash


@dataclass(frozen=True, slots=True)
class TestEvaluationConfig:
    benchmark: str
    env_ref: str | None = None
    rollouts: int = 1
    max_concurrent_rollouts: int = 16
    rollout_timeout: float | None = None
    rollout_retries: int = 1
    seed: int = 0
    env_options: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.benchmark not in {"alfworld", "livemath", "spreadsheetbench"}:
            raise ValueError(f"unsupported benchmark: {self.benchmark!r}")
        if self.rollouts < 1:
            raise ValueError("test rollouts must be at least 1")
        if self.max_concurrent_rollouts < 1:
            raise ValueError("max concurrent rollouts must be at least 1")
        if self.rollout_retries < 1:
            raise ValueError("rollout retries must be at least 1")


class EvaluationProgress:
    """Small dependency-free progress bar for script-side test rollouts."""

    def __init__(self, total: int, *, stream: TextIO | None = None, width: int = 24) -> None:
        self.total = total
        self.stream = stream or sys.stderr
        self.width = width
        self.completed = 0
        self.correct = 0
        self.errors = 0
        self.reward_sum = 0.0
        self._is_tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self._non_tty_step = max(1, (total + 19) // 20)
        self._last_rendered = -1

    def start(self) -> None:
        self._render(force=True)

    async def on_outcome(self, outcome: RolloutOutcome) -> None:
        reward = reward_of(outcome)
        self.completed += 1
        self.correct += int(reward > 0)
        self.errors += int(outcome.trajectory is None)
        self.reward_sum += reward
        self._render(force=self.completed == self.total)

    def close(self) -> None:
        if self._is_tty:
            self.stream.write("\n")
            self.stream.flush()
        elif self._last_rendered != self.completed:
            self._render(force=True)

    def _render(self, *, force: bool) -> None:
        if not force and not self._is_tty and self.completed - self._last_rendered < self._non_tty_step:
            return
        ratio = self.completed / self.total if self.total else 1.0
        filled = min(self.width, round(self.width * ratio))
        bar = "#" * filled + "-" * (self.width - filled)
        mean_reward = self.reward_sum / self.completed if self.completed else 0.0
        line = (
            f"test [{bar}] {self.completed}/{self.total} "
            f"correct={self.correct} errors={self.errors} mean_reward={mean_reward:.4f}"
        )
        self.stream.write(f"\r{line}" if self._is_tty else f"{line}\n")
        self.stream.flush()
        self._last_rendered = self.completed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("alfworld", "livemath", "spreadsheetbench"), required=True)
    parser.add_argument("--env-ref", help="registered Env override; defaults to --benchmark")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path)
    skill_group = parser.add_mutually_exclusive_group(required=True)
    skill_group.add_argument("--skill", type=Path, help="SKILL.md file or directory containing it")
    skill_group.add_argument("--no-skill", action="store_true", help="run the Agent without injecting a Skill")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)

    parser.add_argument("--target-model", default="openai/gpt-5.4-mini")
    parser.add_argument("--api-base", default=os.getenv("OPENAI_ENDPOINT") or os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--request-timeout", type=float, default=1800.0)
    parser.add_argument("--model-retries", type=int, default=3)
    parser.add_argument("--max-completion-tokens", type=int, default=16384)

    parser.add_argument("--test-rollouts", type=int, default=1)
    parser.add_argument("--test-limit", type=int)
    parser.add_argument("--max-concurrent-rollouts", type=int, default=16)
    parser.add_argument("--rollout-timeout", type=float)
    parser.add_argument("--rollout-retries", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-initial-components",
        type=int,
        help="override virtual_components eager selection; use 0 for fully model-directed loading",
    )

    parser.add_argument("--max-turns", type=int, required=True)
    parser.add_argument("--env-seed", type=int, default=42)
    parser.add_argument("--shell-timeout", type=int, default=120)
    parser.add_argument("--use-theorem", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-sketch", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--shuffle-choices", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def build_dataset(
    benchmark: str,
    *,
    data_root: Path,
    split_dir: Path | None,
    seed: int,
    shuffle_choices: bool,
) -> TaskDataset:
    if benchmark == "alfworld":
        return ALFWorldPathSplitDataset(
            split_dir=split_dir or Path("data/ALFWorld/alfworld_path_split"),
            alfworld_data=data_root,
        )
    if benchmark == "livemath":
        return LiveMathIdSplitDataset(
            data_path=data_root,
            split_dir=split_dir or Path("data/LiveMath/livemathematicianbench_id_split"),
            seed=seed,
            shuffle_choices=shuffle_choices,
        )
    if benchmark == "spreadsheetbench":
        return SpreadsheetBenchIdSplitDataset(data_root=data_root, split_dir=split_dir)
    raise ValueError(f"unsupported benchmark: {benchmark!r}")


def environment_options(
    benchmark: str,
    *,
    max_turns: int,
    env_seed: int,
    shell_timeout: int,
    use_theorem: bool,
    use_sketch: bool,
) -> dict[str, Any]:
    options: dict[str, Any] = {"max_turns": max_turns}
    if benchmark == "alfworld":
        options["seed"] = env_seed
    if benchmark == "spreadsheetbench":
        options["shell_timeout_seconds"] = shell_timeout
    if benchmark == "livemath":
        options.update({"use_theorem": use_theorem, "use_sketch": use_sketch})
    return options


def limited_test_tasks(dataset: TaskDataset, limit: int | None) -> list[Task]:
    if limit is not None and limit < 1:
        raise ValueError("test-limit must be at least 1")
    tasks = dataset.test_tasks()
    return tasks if limit is None else tasks[:limit]


def _heading_name(content: str) -> str | None:
    match = re.search(r"^#\s+(.+?)\s*$", content, re.MULTILINE)
    return match.group(1).strip() if match else None


def _first_paragraph(content: str) -> str | None:
    for raw in re.split(r"\n\s*\n", content):
        paragraph = " ".join(line.strip() for line in raw.splitlines()).strip()
        if not paragraph or paragraph.startswith(("#", "---", "```", "|", "- ", ">")):
            continue
        return paragraph
    return None


def build_skill(
    path: Path,
    *,
    run_id: str,
    benchmark: str,
    max_initial_components: int | None = None,
) -> Skill:
    requested = path.expanduser()
    source_dir = requested if requested.is_dir() else requested.parent
    source = requested / "SKILL.md" if requested.is_dir() else requested
    if not source.is_file():
        raise FileNotFoundError(f"Skill file does not exist: {source}")
    content = source.read_text(encoding="utf-8")
    blob = {"SKILL.md": content}
    runtime_type = "static"
    runtime_metadata: dict[str, Any] = {}
    metadata_path = source_dir / "runtime_metadata.json"
    if metadata_path.is_file():
        loaded = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"virtual Skill runtime metadata must be a JSON object: {metadata_path}")
        runtime_type = "virtual_components"
        runtime_metadata = loaded
        if max_initial_components is not None:
            runtime_metadata = {**runtime_metadata, "max_initial_components": max_initial_components}
    elif max_initial_components is not None:
        raise ValueError("--max-initial-components requires a Skill directory with runtime_metadata.json")
    name = frontmatter_value(content, "name") or _heading_name(content) or source_dir.name
    description = (
        frontmatter_value(content, "description")
        or _first_paragraph(content)
        or f"Guidance for {benchmark} tasks."
    )
    return Skill(
        skill_id=f"{benchmark}-evaluation-skill",
        version_id=f"{run_id}:evaluation",
        version_label="0.1.0",
        content_hash=compute_skill_content_hash(blob),
        name=name,
        description=description,
        blob=blob,
        runtime_type=runtime_type,
        runtime_schema_version=1,
        runtime_metadata=runtime_metadata,
        created_at=datetime.now(UTC),
    )


def build_client(
    *,
    model: str,
    api_base: str | None,
    request_timeout: float,
    model_retries: int,
    call_sink: LLMCallSink | None = None,
) -> LLMClient:
    endpoint = {
        "model": model,
        "api_key": os.getenv("OPENAI_API_KEY"),
        "api_base": api_base,
        "temperature": None,
        "timeout": request_timeout,
        "num_retries": model_retries,
    }
    router, retries = get_router({"endpoints": [endpoint]}, model, num_retries=model_retries)
    return LLMClient(router, default_model=model, max_attempts=retries + 1, call_sink=call_sink)


def build_agent(
    *,
    client: LLMClient,
    model: str,
    max_turns: int,
    reasoning_effort: str | None,
    max_completion_tokens: int,
) -> ReactAgent:
    return ReactAgent(
        {
            "model": model,
            "max_turns": max_turns,
            "reasoning_effort": reasoning_effort,
            "model_kwargs": {"max_completion_tokens": max_completion_tokens},
        },
        llm=client,
    )


async def evaluate_test_tasks(
    *,
    run_id: str,
    tasks: list[Task],
    skill: Skill | None,
    agent: Agent[Any],
    output_dir: Path,
    config: TestEvaluationConfig,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    rollout_config = RolloutConfig(
        max_concurrent_rollouts=config.max_concurrent_rollouts,
        timeout_seconds=config.rollout_timeout,
        retry={"max_attempts": config.rollout_retries},
        fail_fast=False,
        workspace_root=output_dir / "workspace",
    )
    specs = FixedGroupRolloutStrategy().plan(
        FixedGroupPlan(
            run_id=run_id,
            scope="test",
            phase="test",
            tasks=tasks,
            skills=[skill] if skill is not None else [],
            sequence_start=0,
            group_size=config.rollouts,
            agent_ref="react",
            env_ref=config.env_ref or config.benchmark,
            seed=config.seed,
            env_options=config.env_options or {},
        )
    )
    progress = EvaluationProgress(len(specs))
    scheduler = RolloutScheduler(
        agent_resolver=MappingAgentResolver({"react": agent}),
        env_factory=RegistryEnvFactory(),
        config=rollout_config,
        on_outcome=progress.on_outcome,
    )
    progress.start()
    try:
        with llm_run_context(f"{run_id}:test"):
            outcomes = await scheduler.run(specs)
    finally:
        progress.close()
    return persist_test_artifacts(
        output_dir=output_dir,
        outcomes=outcomes,
        skill=skill,
        rollouts_per_task=config.rollouts,
        create_output_dir=False,
    )


def persist_test_artifacts(
    *,
    output_dir: Path,
    outcomes: Sequence[RolloutOutcome],
    skill: Skill | None,
    rollouts_per_task: int,
    create_output_dir: bool = True,
) -> dict[str, Any]:
    if create_output_dir:
        output_dir.mkdir(parents=True, exist_ok=False)
    summary = summarize(outcomes, rollouts_per_task=rollouts_per_task, skill=skill)
    write_json(output_dir / "summary.json", summary)
    records = [result_record(outcome, skill=skill) for outcome in outcomes]
    write_jsonl(output_dir / "results.jsonl", records)
    write_jsonl(
        output_dir / "skill_usage.jsonl",
        [
            {
                "task_id": record["task_id"],
                "rollout_id": record["rollout_id"],
                "sample_index": record["sample_index"],
                "reward": record["reward"],
                **record["skill_loading"],
            }
            for record in records
        ],
    )
    write_json(output_dir / "skill.json", skill.model_dump(mode="json") if skill is not None else None)
    return summary


def reward_of(outcome: RolloutOutcome) -> float:
    trajectory = outcome.trajectory
    if trajectory is None or trajectory.reward.score is None:
        return 0.0
    return float(trajectory.reward.score)


def summarize(
    outcomes: Sequence[RolloutOutcome],
    *,
    rollouts_per_task: int,
    skill: Skill | None,
) -> dict[str, Any]:
    rewards = [reward_of(outcome) for outcome in outcomes]
    task_rewards: dict[str, list[float]] = {}
    for outcome, reward in zip(outcomes, rewards, strict=True):
        task_rewards.setdefault(outcome.spec.task.task_id, []).append(reward)
    task_means = [mean(values) for values in task_rewards.values()]
    correct = sum(reward > 0 for reward in rewards)
    passed_tasks = sum(any(reward > 0 for reward in values) for values in task_rewards.values())
    loading_records = [skill_loading_record(outcome.trajectory, skill=skill) for outcome in outcomes]
    return {
        "mode": "skill" if skill is not None else "no_skill",
        "skill_content_hash": skill.content_hash if skill is not None else None,
        "tasks": len(task_rewards),
        "rollouts_per_task": rollouts_per_task,
        "total": len(outcomes),
        "completed": sum(outcome.trajectory is not None for outcome in outcomes),
        "failed": sum(outcome.trajectory is None for outcome in outcomes),
        "correct": correct,
        "accuracy": correct / len(outcomes) if outcomes else 0.0,
        "pass_at_k": passed_tasks / len(task_rewards) if task_rewards else 0.0,
        "mean_per_task": mean(task_means) if task_means else 0.0,
        "reward": distribution(rewards),
        "reward_histogram": histogram(rewards),
        "task_mean_reward": distribution(task_means),
        "task_mean_reward_histogram": histogram(task_means),
        "skill_loading": summarize_skill_loading(loading_records),
    }


def distribution(values: Sequence[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "mean": mean(values) if values else 0.0,
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def histogram(values: Sequence[float]) -> dict[str, int]:
    counts = Counter(f"{float(value):.6g}" for value in values)
    return dict(sorted(counts.items(), key=lambda item: float(item[0])))


def _tool_arguments(call: dict[str, Any]) -> dict[str, Any]:
    function = call.get("function")
    if not isinstance(function, dict):
        return {}
    arguments = function.get("arguments", {})
    try:
        loaded = json.loads(arguments) if isinstance(arguments, str) else arguments
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def skill_loading_record(trajectory: Trajectory | None, *, skill: Skill | None) -> dict[str, Any]:
    skill_calls: list[str | None] = []
    resource_calls: list[str | None] = []
    if trajectory is not None:
        for event in trajectory.events:
            for raw_call in event.get("tool_calls") or []:
                if not isinstance(raw_call, dict):
                    continue
                function = raw_call.get("function")
                if not isinstance(function, dict):
                    continue
                arguments = _tool_arguments(raw_call)
                if function.get("name") == "skill":
                    name = arguments.get("name")
                    skill_calls.append(name if isinstance(name, str) else None)
                elif function.get("name") == "load_skill_resource":
                    resource_id = arguments.get("resource_id")
                    resource_calls.append(resource_id if isinstance(resource_id, str) else None)

    component_by_resource: dict[str, str] = {}
    if skill is not None and skill.runtime_type == "virtual_components":
        components = skill.runtime_metadata.get("components", [])
        for component in components if isinstance(components, list) else []:
            if not isinstance(component, dict) or not isinstance(component.get("component_id"), str):
                continue
            component_id = component["component_id"]
            component_by_resource[f"skill-resource:{skill.version_id}:{component_id}"] = component_id

    loaded_resource_ids: list[str] = []
    if trajectory is not None:
        runtime_trace = trajectory.metadata.get("skill_runtime")
        if isinstance(runtime_trace, dict):
            for item in runtime_trace.get("skills", []):
                if isinstance(item, dict) and item.get("version_id") == (skill.version_id if skill else None):
                    loaded_resource_ids.extend(
                        value for value in item.get("loaded_resource_ids", []) if isinstance(value, str)
                    )
    requested_components = [component_by_resource.get(value, value) for value in resource_calls if value]
    loaded_components = [component_by_resource.get(value, value) for value in loaded_resource_ids]
    total_calls = len(skill_calls) + len(resource_calls)
    return {
        "total_load_calls": total_calls,
        "skill_tool_call_count": len(skill_calls),
        "virtual_skill_load_call_count": len(resource_calls),
        "requested_skill_names": [value for value in skill_calls if value],
        "requested_virtual_skill_ids": requested_components,
        "loaded_virtual_skill_ids": sorted(set(loaded_components)),
        "loaded_any_skill": total_calls > 0,
    }


def summarize_skill_loading(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    total_load_calls = sum(int(record["total_load_calls"]) for record in records)
    loaded_rollouts = sum(bool(record["loaded_any_skill"]) for record in records)
    virtual_calls = Counter(
        component_id for record in records for component_id in record["requested_virtual_skill_ids"]
    )
    skill_calls = Counter(name for record in records for name in record["requested_skill_names"])
    return {
        "total_load_calls": total_load_calls,
        "skill_tool_calls": sum(int(record["skill_tool_call_count"]) for record in records),
        "virtual_skill_load_calls": sum(int(record["virtual_skill_load_call_count"]) for record in records),
        "rollouts_with_load": loaded_rollouts,
        "rollouts_without_load": total - loaded_rollouts,
        "load_rate": loaded_rollouts / total if total else 0.0,
        "mean_load_calls_per_rollout": total_load_calls / total if total else 0.0,
        "skill_tool_calls_by_name": dict(sorted(skill_calls.items())),
        "virtual_skill_load_calls_by_id": dict(sorted(virtual_calls.items())),
    }


def result_record(outcome: RolloutOutcome, *, skill: Skill | None = None) -> dict[str, Any]:
    trajectory = outcome.trajectory
    last_attempt = outcome.attempts[-1] if outcome.attempts else None
    return {
        "split": "test",
        "task_id": outcome.spec.task.task_id,
        "rollout_id": outcome.spec.rollout_id,
        "sample_index": outcome.spec.sample_index,
        "skill_content_hashes": [skill.content_hash for skill in outcome.spec.skills],
        "reward": reward_of(outcome),
        "completed": trajectory is not None,
        "error_type": last_attempt.error_type if trajectory is None and last_attempt is not None else None,
        "error": last_attempt.error if trajectory is None and last_attempt is not None else None,
        "skill_loading": skill_loading_record(trajectory, skill=skill),
        "trajectory": trajectory.model_dump(mode="json") if trajectory is not None else None,
    }


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, payloads: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for payload in payloads:
            stream.write(json.dumps(payload, ensure_ascii=False) + "\n")


async def run(args: argparse.Namespace) -> dict[str, Any]:
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is required")
    if not args.api_base:
        raise ValueError("--api-base, OPENAI_ENDPOINT, or OPENAI_BASE_URL is required")
    dataset = build_dataset(
        args.benchmark,
        data_root=args.data_root,
        split_dir=args.split_dir,
        seed=args.seed,
        shuffle_choices=args.shuffle_choices,
    )
    tasks = limited_test_tasks(dataset, args.test_limit)
    skill = (
        None
        if args.no_skill
        else build_skill(
            args.skill,
            run_id=args.run_id,
            benchmark=args.benchmark,
            max_initial_components=args.max_initial_components,
        )
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output_dir / "arguments.json",
        {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    )
    client = build_client(
        model=args.target_model,
        api_base=args.api_base,
        request_timeout=args.request_timeout,
        model_retries=args.model_retries,
    )
    agent = build_agent(
        client=client,
        model=args.target_model,
        max_turns=args.max_turns,
        reasoning_effort=args.reasoning_effort,
        max_completion_tokens=args.max_completion_tokens,
    )
    summary = await evaluate_test_tasks(
        run_id=args.run_id,
        tasks=tasks,
        skill=skill,
        agent=agent,
        output_dir=args.output_dir / "test",
        config=TestEvaluationConfig(
            benchmark=args.benchmark,
            env_ref=args.env_ref,
            rollouts=args.test_rollouts,
            max_concurrent_rollouts=args.max_concurrent_rollouts,
            rollout_timeout=args.rollout_timeout,
            rollout_retries=args.rollout_retries,
            seed=args.seed,
            env_options=environment_options(
                args.benchmark,
                max_turns=args.max_turns,
                env_seed=args.env_seed,
                shell_timeout=args.shell_timeout,
                use_theorem=args.use_theorem,
                use_sketch=args.use_sketch,
            ),
        ),
    )
    write_json(args.output_dir / "summary.json", {"run_id": args.run_id, "test": summary})
    return summary


def main(argv: list[str] | None = None) -> None:
    print(json.dumps(asyncio.run(run(parse_args(argv))), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


__all__ = [
    "EvaluationProgress",
    "TestEvaluationConfig",
    "build_agent",
    "build_client",
    "build_dataset",
    "build_skill",
    "environment_options",
    "evaluate_test_tasks",
    "limited_test_tasks",
    "persist_test_artifacts",
    "skill_loading_record",
    "summarize",
    "summarize_skill_loading",
]
