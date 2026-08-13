#!/usr/bin/env python3
"""Run Tree Skill GRPO on one registered benchmark."""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
from typing import Any

from mindmemos_skill.algos.evolve.tree_skill_grpo import TreeSkillGrpoAlgorithm
from mindmemos_skill.llm import DatabaseLLMCallSink
from mindmemos_skill.persistence import bootstrap_skill_database
from mindmemos_skill.typing import EvolveInput

from .skill_evaluation import (
    TestEvaluationConfig,
    build_agent,
    build_client,
    build_dataset,
    build_skill,
    environment_options,
    evaluate_test_tasks,
    write_json,
)


class ChatModelWithDefaults:
    def __init__(self, client: Any, **defaults: Any) -> None:
        self._client = client
        self._defaults = defaults

    async def chat(self, task: str, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        return await self._client.chat(task, messages, **{**self._defaults, **kwargs})


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("alfworld", "livemath", "spreadsheetbench"), required=True)
    parser.add_argument("--env-ref", help="registered Env override; ALFWorld defaults to alfworld_bounded_history")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path)
    parser.add_argument("--initial-skill", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)

    parser.add_argument("--target-model", default="openai/gpt-5.4-mini")
    parser.add_argument("--optimizer-model", default="openai/gpt-5.4-mini")
    parser.add_argument("--api-base", default=os.getenv("OPENAI_ENDPOINT") or os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--request-timeout", type=float, default=1800.0)
    parser.add_argument("--model-retries", type=int, default=3)
    parser.add_argument("--max-completion-tokens", type=int, default=16384)

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--max-task-attempts", type=int, default=4)
    parser.add_argument("--max-concurrency", type=int, default=32)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--test-limit", type=int)
    parser.add_argument("--test-rollouts", type=int, default=1)
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


def limited(items: list[Any], limit: int | None) -> list[Any]:
    if limit is not None and limit < 1:
        raise ValueError("task limits must be at least 1")
    return items if limit is None else items[:limit]


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
    train_tasks = limited(dataset.train_tasks(), args.train_limit)
    test_tasks = limited(dataset.test_tasks(), args.test_limit)
    base_skill = build_skill(args.initial_skill, run_id=args.run_id, benchmark=args.benchmark)
    env_options = environment_options(
        args.benchmark,
        max_turns=args.max_turns,
        env_seed=args.env_seed,
        shell_timeout=args.shell_timeout,
        use_theorem=args.use_theorem,
        use_sketch=args.use_sketch,
    )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output_dir / "arguments.json",
        {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    )
    database = await bootstrap_skill_database(args.output_dir / "state.db")
    sink = DatabaseLLMCallSink(database)
    target_client = build_client(
        model=args.target_model,
        api_base=args.api_base,
        request_timeout=args.request_timeout,
        model_retries=args.model_retries,
        call_sink=sink,
    )
    optimizer_client = ChatModelWithDefaults(
        build_client(
            model=args.optimizer_model,
            api_base=args.api_base,
            request_timeout=args.request_timeout,
            model_retries=args.model_retries,
            call_sink=sink,
        ),
        reasoning_effort=args.reasoning_effort,
        max_completion_tokens=args.max_completion_tokens,
        temperature=0.0,
    )
    agent = build_agent(
        client=target_client,
        model=args.target_model,
        max_turns=args.max_turns,
        reasoning_effort=args.reasoning_effort,
        max_completion_tokens=args.max_completion_tokens,
    )
    algorithm = TreeSkillGrpoAlgorithm(
        env_ref=args.env_ref or ("alfworld_bounded_history" if args.benchmark == "alfworld" else args.benchmark),
        chat_model=optimizer_client,
        agent=agent,
        env_options=env_options,
        workspace_root=args.output_dir / "workspace",
        checkpoint_dir=args.output_dir / "epochs",
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_task_attempts=args.max_task_attempts,
        max_concurrency=args.max_concurrency,
    )
    try:
        result = await algorithm.evolve(
            EvolveInput(run_id=args.run_id, base_skill=base_skill, train_tasks=train_tasks, test_tasks=test_tasks)
        )
        test_summary = await evaluate_test_tasks(
            run_id=args.run_id,
            tasks=test_tasks,
            skill=result.final_skill,
            agent=agent,
            output_dir=args.output_dir / "test",
            config=TestEvaluationConfig(
                benchmark=args.benchmark,
                env_ref=args.env_ref
                or ("alfworld_bounded_history" if args.benchmark == "alfworld" else args.benchmark),
                rollouts=args.test_rollouts,
                max_concurrent_rollouts=args.max_concurrent_rollouts,
                rollout_timeout=args.rollout_timeout,
                rollout_retries=args.rollout_retries,
                seed=args.seed,
                env_options=env_options,
            ),
        )
    finally:
        await database.close()

    (args.output_dir / "result.json").write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "final_skill.md").write_text(result.final_skill.content, encoding="utf-8")
    summary = {
        "run_id": result.run_id,
        "changed": result.changed,
        "trajectory_count": len(result.trajectories),
        "final_skill_hash": result.final_skill.content_hash,
        "test": test_summary,
    }
    write_json(args.output_dir / "summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> None:
    asyncio.run(run(parse_args(argv)))


if __name__ == "__main__":
    main()
