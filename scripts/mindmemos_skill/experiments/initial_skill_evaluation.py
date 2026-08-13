#!/usr/bin/env python3
"""Script-side evaluation of the ALFWorld test split with one fixed initial Skill."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from mindmemos_skill.agents.react import ReactAgent
from mindmemos_skill.algos.evolve.skill_grpo_with_replay_buffer.config import RolloutConfig
from mindmemos_skill.algos.evolve.skill_grpo_with_replay_buffer.contracts import RolloutOutcome
from mindmemos_skill.algos.evolve.skill_grpo_with_replay_buffer.rollout import (
    FixedGroupPlan,
    MappingAgentResolver,
    RegistryEnvFactory,
    RolloutScheduler,
    RolloutStrategyRegistry,
)
from mindmemos_skill.datasets import ALFWorldPathSplitDataset
from mindmemos_skill.llm import LLMClient, get_router, llm_run_context
from mindmemos_skill.typing import Skill, compute_skill_content_hash


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--initial-skill", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--target-model", default="openai/gpt-5.4-mini")
    parser.add_argument("--api-base", default=os.getenv("OPENAI_ENDPOINT") or os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--request-timeout", type=float, default=1800.0)
    parser.add_argument("--model-retries", type=int, default=3)
    parser.add_argument("--max-completion-tokens", type=int, default=16384)
    parser.add_argument("--rollouts", type=int, default=3)
    parser.add_argument("--max-concurrent-rollouts", type=int, default=16)
    parser.add_argument("--rollout-timeout", type=float)
    parser.add_argument("--rollout-retries", type=int, default=1)
    parser.add_argument("--max-turns", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env-seed", type=int, default=42)
    parser.add_argument("--test-limit", type=int)
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args(argv)


def build_skill(path: Path, run_id: str) -> Skill:
    content = path.read_text(encoding="utf-8")
    blob = {"SKILL.md": content}
    return Skill(
        skill_id="alfworld-skill",
        version_id=f"{run_id}:init",
        version_label="0.1.0",
        content_hash=compute_skill_content_hash(blob),
        name="ALFWorld Embodied Agent Skill",
        description="Initial ALFWorld strategy aligned with the Skill-GRPO baseline.",
        blob=blob,
        created_at=datetime.now(UTC),
    )


def build_client(args: argparse.Namespace) -> LLMClient:
    endpoint = {
        "model": args.target_model,
        "api_key": os.getenv("OPENAI_API_KEY"),
        "api_base": args.api_base,
        "temperature": None,
        "timeout": args.request_timeout,
        "num_retries": args.model_retries,
    }
    router, retries = get_router({"endpoints": [endpoint]}, args.target_model, num_retries=args.model_retries)
    return LLMClient(router, default_model=args.target_model, max_attempts=retries + 1)


def limited(items: list[Any], limit: int | None) -> list[Any]:
    if limit is None:
        return items
    if limit < 1:
        raise ValueError("test-limit must be at least 1")
    return items[:limit]


def reward_of(outcome: RolloutOutcome) -> float:
    trajectory = outcome.trajectory
    if trajectory is None or trajectory.reward.score is None:
        return 0.0
    return float(trajectory.reward.score)


def summarize(outcomes: Sequence[RolloutOutcome], *, rollouts_per_task: int) -> dict[str, Any]:
    rewards = [reward_of(outcome) for outcome in outcomes]
    task_rewards: dict[str, list[float]] = {}
    for outcome, reward in zip(outcomes, rewards, strict=True):
        task_rewards.setdefault(outcome.spec.task.task_id, []).append(reward)
    task_means = [mean(values) for values in task_rewards.values()]
    correct = sum(reward > 0 for reward in rewards)
    passed_tasks = sum(any(reward > 0 for reward in values) for values in task_rewards.values())
    return {
        "total": len(rewards),
        "correct": correct,
        "accuracy": correct / len(rewards) if rewards else 0.0,
        "tasks": len(task_rewards),
        "rollouts_per_task": rollouts_per_task,
        "pass_at_k": passed_tasks / len(task_rewards) if task_rewards else 0.0,
        "mean_per_task": mean(task_means) if task_means else 0.0,
        "reward_mean": mean(rewards) if rewards else 0.0,
        "reward_min": min(rewards) if rewards else None,
        "reward_max": max(rewards) if rewards else None,
        "reward_std": pstdev(rewards) if len(rewards) > 1 else 0.0,
        "reward": distribution(rewards),
        "reward_histogram": histogram(rewards),
        "task_mean_reward": distribution(task_means),
        "task_mean_reward_histogram": histogram(task_means),
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


def result_record(outcome: RolloutOutcome) -> dict[str, Any]:
    trajectory = outcome.trajectory
    task = outcome.spec.task
    if trajectory is None:
        last_attempt = outcome.attempts[-1] if outcome.attempts else None
        metadata = {
            "error": last_attempt.error if last_attempt is not None else "rollout produced no trajectory",
            "error_type": last_attempt.error_type if last_attempt is not None else None,
            "sample_index": outcome.spec.sample_index,
        }
        events: list[dict[str, Any]] = []
        workspace = None
    else:
        metadata = trajectory.metadata
        events = trajectory.events
        workspace = trajectory.environment.running_dir
    return {
        "split": "test",
        "rollout": {
            "task": {
                "id": task.task_id,
                "instruction": task.instruction,
                "system_prompt": task.system_prompt,
                "tags": task.tags,
                "metadata": task.metadata,
            },
            "skills": [
                {"name": skill.name, "description": skill.description, "content": skill.content}
                for skill in outcome.spec.skills
            ],
            "trajectory": events,
            "reward": reward_of(outcome),
            "workspace": workspace,
            "metadata": metadata,
        },
    }


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


async def run(args: argparse.Namespace) -> dict[str, Any]:
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is required")
    if not args.api_base:
        raise ValueError("--api-base, OPENAI_ENDPOINT, or OPENAI_BASE_URL is required")
    if args.rollouts < 1:
        raise ValueError("rollouts must be at least 1")
    if args.progress_every < 1:
        raise ValueError("progress-every must be at least 1")

    os.environ["ALFWORLD_DATA"] = str(args.data_root.expanduser().resolve())
    dataset = ALFWorldPathSplitDataset(split_dir=args.split_dir, alfworld_data=args.data_root)
    tasks = limited(dataset.test_tasks(), args.test_limit)
    skill = build_skill(args.initial_skill, args.run_id)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output_dir / "arguments.json",
        {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    )
    write_json(args.output_dir / "skill_bank.json", [skill.model_dump(mode="json")])

    rollout_config = RolloutConfig.model_validate(
        {
            "max_concurrent_rollouts": args.max_concurrent_rollouts,
            "timeout_seconds": args.rollout_timeout,
            "retry": {"max_attempts": args.rollout_retries},
            "fail_fast": False,
            "workspace_root": args.output_dir / "workspace",
        }
    )
    agent = ReactAgent(
        {
            "model": args.target_model,
            "max_turns": 1,
            "reasoning_effort": args.reasoning_effort,
            "model_kwargs": {"max_completion_tokens": args.max_completion_tokens},
        },
        llm=build_client(args),
    )
    specs = (
        RolloutStrategyRegistry.with_builtins()
        .get("fixed_group")
        .plan(
            FixedGroupPlan(
                run_id=args.run_id,
                scope="test",
                phase="test",
                tasks=tasks,
                skills=[skill],
                sequence_start=0,
                group_size=args.rollouts,
                agent_ref="react",
                env_ref="alfworld_bounded_history",
                seed=args.seed,
                env_options={"max_turns": args.max_turns, "seed": args.env_seed},
            )
        )
    )
    result_path = args.output_dir / "result.jsonl"
    completed: list[RolloutOutcome] = []
    write_lock = asyncio.Lock()

    async def on_outcome(outcome: RolloutOutcome) -> None:
        async with write_lock:
            completed.append(outcome)
            with result_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(result_record(outcome), ensure_ascii=False) + "\n")
            count = len(completed)
            if count == len(specs) or count % args.progress_every == 0:
                correct = sum(reward_of(item) > 0 for item in completed)
                print(f"progress={count}/{len(specs)} correct={correct} accuracy={correct / count:.2%}", flush=True)

    scheduler = RolloutScheduler(
        agent_resolver=MappingAgentResolver({"react": agent}),
        env_factory=RegistryEnvFactory(),
        config=rollout_config,
        on_outcome=on_outcome,
    )
    with llm_run_context(args.run_id):
        outcomes = await scheduler.run(specs)

    result_path.write_text(
        "".join(json.dumps(result_record(outcome), ensure_ascii=False) + "\n" for outcome in outcomes),
        encoding="utf-8",
    )
    summary = {"test": summarize(outcomes, rollouts_per_task=args.rollouts)}
    write_json(args.output_dir / "summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> None:
    summary = asyncio.run(run(parse_args(argv)))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
