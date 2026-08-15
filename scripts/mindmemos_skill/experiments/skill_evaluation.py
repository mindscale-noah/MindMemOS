#!/usr/bin/env python3
"""Script-side evaluation of a supplied Skill, or no Skill, on one test split."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
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
from mindmemos_skill.typing import (
    Skill,
    SkillInjectionMode,
    Task,
    TrajectoryStatus,
    compute_skill_content_hash,
)


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
        self.errors += int(not execution_succeeded(outcome))
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
    transactional_recalculation: bool = False,
    trace2skill_reference_mode: bool = False,
) -> dict[str, Any]:
    options: dict[str, Any] = {"max_turns": max_turns}
    if benchmark == "alfworld":
        options["seed"] = env_seed
    if benchmark == "spreadsheetbench":
        options["shell_timeout_seconds"] = shell_timeout
        if transactional_recalculation:
            options["transactional_recalculation"] = True
        if trace2skill_reference_mode:
            options["trace2skill_reference_mode"] = True
    if benchmark == "livemath":
        options.update({"use_theorem": use_theorem, "use_sketch": use_sketch})
    return options


def limited_test_tasks(dataset: TaskDataset, limit: int | None) -> list[Task]:
    if limit is not None and limit < 1:
        raise ValueError("test-limit must be at least 1")
    tasks = dataset.test_tasks()
    return tasks if limit is None else tasks[:limit]


def build_skill(
    path: Path,
    *,
    run_id: str,
    benchmark: str,
    include_resources: bool = False,
) -> Skill:
    source = path.expanduser()
    source_dir = source if source.is_dir() else source.parent
    if source.is_dir():
        source = source / "SKILL.md"
    if not source.is_file():
        raise FileNotFoundError(f"Skill file does not exist: {source}")
    if include_resources and source.name != "SKILL.md":
        raise ValueError("resource-bearing Skill packages must use a file named SKILL.md")
    blob = {"SKILL.md": source.read_text(encoding="utf-8")}
    resources: dict[str, str] = {}
    if include_resources:
        for candidate in sorted(source_dir.rglob("*")):
            if not candidate.is_file() or candidate == source:
                continue
            relative = candidate.relative_to(source_dir).as_posix()
            if relative == "SKILL.md":
                continue
            try:
                resources[relative] = candidate.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
    return Skill(
        skill_id=f"{benchmark}-evaluation-skill",
        version_id=f"{run_id}:evaluation",
        version_label="0.1.0",
        content_hash=compute_skill_content_hash(blob),
        name=f"{benchmark}-evaluation",
        blob=blob,
        resources=resources,
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
    skill_injection_mode: SkillInjectionMode = SkillInjectionMode.TOOL,
    tree_router_temperature: float = 0.0,
    tree_router_max_tokens: int = 512,
    temperature: float | None = None,
    stop: Sequence[str] | None = None,
) -> ReactAgent:
    model_kwargs: dict[str, Any] = {"max_completion_tokens": max_completion_tokens}
    if stop is not None:
        model_kwargs["stop"] = list(stop)
    return ReactAgent(
        {
            "model": model,
            "max_turns": max_turns,
            "reasoning_effort": reasoning_effort,
            "skill_injection_mode": skill_injection_mode,
            "tree_router_temperature": tree_router_temperature,
            "tree_router_max_tokens": tree_router_max_tokens,
            "temperature": temperature,
            "model_kwargs": model_kwargs,
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
    summary = persist_test_artifacts(
        output_dir=output_dir,
        outcomes=outcomes,
        skill=skill,
        rollouts_per_task=config.rollouts,
        create_output_dir=False,
    )
    if summary["execution_exceptions"]:
        raise RuntimeError(f"test evaluation encountered execution exceptions; see {output_dir / 'results.jsonl'}")
    return summary


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
    write_jsonl(output_dir / "results.jsonl", [result_record(outcome) for outcome in outcomes])
    write_json(output_dir / "skill.json", skill.model_dump(mode="json") if skill is not None else None)
    return summary


def reward_of(outcome: RolloutOutcome) -> float:
    trajectory = outcome.trajectory
    if trajectory is None or trajectory.reward.score is None:
        return 0.0
    return float(trajectory.reward.score)


def execution_succeeded(outcome: RolloutOutcome) -> bool:
    """Return whether the Agent completed execution, not merely returned evidence."""

    trajectory = outcome.trajectory
    return trajectory is not None and trajectory.execution.status is TrajectoryStatus.SUCCEEDED


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
    summary = {
        "mode": "skill" if skill is not None else "no_skill",
        "skill_content_hash": skill.content_hash if skill is not None else None,
        "tasks": len(task_rewards),
        "rollouts_per_task": rollouts_per_task,
        "total": len(outcomes),
        "completed": sum(execution_succeeded(outcome) for outcome in outcomes),
        "failed": sum(not execution_succeeded(outcome) for outcome in outcomes),
        "trajectories_returned": sum(outcome.trajectory is not None for outcome in outcomes),
        "trajectory_exceptions": sum(outcome.trajectory is None for outcome in outcomes),
        "execution_exceptions": sum(
            outcome.trajectory is not None and bool(outcome.trajectory.metadata.get("execution_exception_type"))
            for outcome in outcomes
        ),
        "correct": correct,
        "accuracy": correct / len(outcomes) if outcomes else 0.0,
        "pass_at_k": passed_tasks / len(task_rewards) if task_rewards else 0.0,
        "mean_per_task": mean(task_means) if task_means else 0.0,
        "reward": distribution(rewards),
        "reward_histogram": histogram(rewards),
        "task_mean_reward": distribution(task_means),
        "task_mean_reward_histogram": histogram(task_means),
    }
    routing = [
        trajectory.metadata["treeskill_routing"]
        for outcome in outcomes
        if (trajectory := outcome.trajectory) is not None
        and isinstance(trajectory.metadata.get("treeskill_routing"), dict)
    ]
    if routing:
        full_chars = sum(int(item.get("full_char_count", 0)) for item in routing)
        routed_chars = sum(int(item.get("routed_char_count", 0)) for item in routing)
        summary["treeskill_routing"] = {
            "requests": len(routing),
            "full_char_count": full_chars,
            "routed_char_count": routed_chars,
            "context_saving_ratio": 0.0 if full_chars == 0 else 1.0 - (routed_chars / full_chars),
            "fallback_requests": sum(
                any(bool(detail.get("fallback_used")) for detail in item.get("skills", {}).values()) for item in routing
            ),
        }
    return summary


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


def result_record(outcome: RolloutOutcome) -> dict[str, Any]:
    trajectory = outcome.trajectory
    last_attempt = outcome.attempts[-1] if outcome.attempts else None
    completed = execution_succeeded(outcome)
    if trajectory is not None and not completed:
        error_type = "TrajectoryExecutionFailed"
        error = trajectory.execution.error_info
    elif trajectory is None and last_attempt is not None:
        error_type = last_attempt.error_type
        error = last_attempt.error
    else:
        error_type = None
        error = None
    return {
        "split": "test",
        "task_id": outcome.spec.task.task_id,
        "rollout_id": outcome.spec.rollout_id,
        "sample_index": outcome.spec.sample_index,
        "skill_content_hashes": [skill.content_hash for skill in outcome.spec.skills],
        "reward": reward_of(outcome),
        "completed": completed,
        "error_type": error_type,
        "error": error,
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
    skill = None if args.no_skill else build_skill(args.skill, run_id=args.run_id, benchmark=args.benchmark)
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
    "summarize",
]
