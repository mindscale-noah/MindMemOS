#!/usr/bin/env python3
"""Script-side replay-free Skill GRPO experiment on a registered benchmark."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from typing import Any

from mindmemos_skill.agents.react import ReactAgent
from mindmemos_skill.algos.evolve.skill_grpo_with_replay_buffer import MappingAgentResolver, RegistryEnvFactory
from mindmemos_skill.algos.evolve.skill_grpo_with_replay_buffer.contracts import RolloutPhase
from mindmemos_skill.algos.evolve.skill_grpo_without_replay_buffer import (
    SkillGrpoWithoutReplayBuffer,
    SkillGrpoWithoutReplayBufferEvolveInput,
    SkillGrpoWithoutReplayBufferRunConfig,
)
from mindmemos_skill.datasets import (
    ALFWorldPathSplitDataset,
    LiveMathIdSplitDataset,
    SpreadsheetBenchIdSplitDataset,
    TaskDataset,
)
from mindmemos_skill.llm import DatabaseLLMCallSink, LLMCallSink, LLMClient, get_router
from mindmemos_skill.logging import AlgorithmLogger
from mindmemos_skill.persistence import bootstrap_skill_database
from mindmemos_skill.typing import Skill, compute_skill_content_hash

from .skill_evaluation import persist_test_artifacts


class ChatModelWithDefaults:
    def __init__(self, client: LLMClient, **defaults: Any) -> None:
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
    parser.add_argument("--model-retries", type=int, default=10)
    parser.add_argument("--max-completion-tokens", type=int, default=16384)

    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--mini-batch-size", type=int, default=8)
    parser.add_argument("--train-rollouts", type=int, default=4)
    parser.add_argument("--validation-gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--validation-rollouts", type=int, default=1)
    parser.add_argument("--test-rollouts", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env-seed", type=int)
    parser.add_argument("--max-concurrent-rollouts", type=int, default=32)
    parser.add_argument("--max-concurrent-extractions", type=int, default=16)
    parser.add_argument("--reflection", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-concurrent-reflections", type=int, default=8)
    parser.add_argument("--rollout-retries", type=int, default=3)
    parser.add_argument("--rollout-timeout", type=float)
    parser.add_argument("--fail-fast", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--max-turns", type=int, required=True)
    parser.add_argument("--shell-timeout", type=int, default=120)
    parser.add_argument("--use-theorem", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-sketch", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--shuffle-choices", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--max-experiences-per-task", type=int, default=3)
    parser.add_argument("--max-patch-edits", type=int, default=6)
    parser.add_argument("--patch-attempts", type=int, default=2)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--validation-limit", type=int)
    parser.add_argument("--test-limit", type=int)
    return parser.parse_args(argv)


def build_dataset(args: argparse.Namespace) -> TaskDataset:
    if args.benchmark == "spreadsheetbench":
        return SpreadsheetBenchIdSplitDataset(data_root=args.data_root, split_dir=args.split_dir)
    if args.benchmark == "alfworld":
        split_dir = args.split_dir or Path("data/ALFWorld/alfworld_path_split")
        return ALFWorldPathSplitDataset(split_dir=split_dir, alfworld_data=args.data_root)
    split_dir = args.split_dir or Path("data/LiveMath/livemathematicianbench_id_split")
    return LiveMathIdSplitDataset(
        data_path=args.data_root,
        split_dir=split_dir,
        seed=args.seed,
        shuffle_choices=args.shuffle_choices,
    )


def build_client(
    model: str,
    args: argparse.Namespace,
    *,
    call_sink: LLMCallSink,
    temperature: float | None = None,
) -> LLMClient:
    endpoint = {
        "model": model,
        "api_key": os.getenv("OPENAI_API_KEY"),
        "api_base": args.api_base,
        "temperature": temperature,
        "timeout": args.request_timeout,
        "num_retries": args.model_retries,
    }
    router, retries = get_router({"endpoints": [endpoint]}, model, num_retries=args.model_retries)
    return LLMClient(router, default_model=model, max_attempts=retries + 1, call_sink=call_sink)


def build_skill(args: argparse.Namespace) -> Skill:
    content = args.initial_skill.read_text(encoding="utf-8")
    blob = {"SKILL.md": content}
    now = datetime.now(UTC)
    return Skill(
        skill_id=f"{args.benchmark}-skill",
        version_id=f"{args.run_id}:base",
        version_label="0.1.0",
        content_hash=compute_skill_content_hash(blob),
        name=args.benchmark,
        blob=blob,
        created_at=now,
    )


def limited(items: list[Any], limit: int | None) -> list[Any]:
    if limit is None:
        return items
    if limit < 0:
        raise ValueError("task limits must be non-negative")
    return items[:limit]


def build_run_config(args: argparse.Namespace) -> SkillGrpoWithoutReplayBufferRunConfig:
    env_options: dict[str, Any] = {"max_turns": args.max_turns}
    if args.benchmark == "spreadsheetbench":
        env_options["shell_timeout_seconds"] = args.shell_timeout
    elif args.benchmark == "alfworld":
        env_options["seed"] = args.seed if args.env_seed is None else args.env_seed
    else:
        env_options.update({"use_theorem": args.use_theorem, "use_sketch": args.use_sketch})
    return SkillGrpoWithoutReplayBufferRunConfig.model_validate(
        {
            "algorithm": {
                "reflection": {
                    "enabled": args.reflection,
                    "max_concurrent_reflections": args.max_concurrent_reflections,
                },
                "experience": {
                    "max_experiences_per_task": args.max_experiences_per_task,
                    "max_concurrent_extractions": args.max_concurrent_extractions,
                },
                "patch": {"max_edits": args.max_patch_edits, "max_attempts": args.patch_attempts},
                "validation": {"enabled": args.validation_gate},
            },
            "training": {
                "seed": args.seed,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "mini_batch_size": args.mini_batch_size,
                "success_reward": 1.0,
            },
            "rollout": {
                "max_concurrent_rollouts": args.max_concurrent_rollouts,
                "timeout_seconds": args.rollout_timeout,
                "retry": {"max_attempts": args.rollout_retries},
                "fail_fast": args.fail_fast,
                "workspace_root": args.output_dir / "workspace",
                "train": {"name": "fixed_group", "params": {"group_size": args.train_rollouts}},
                "validation": {
                    "name": "fixed_group",
                    "params": {"group_size": args.validation_rollouts},
                },
                "test": {"name": "fixed_group", "params": {"group_size": args.test_rollouts}},
            },
            "dataset": {
                "env_ref": args.env_ref
                or ("alfworld_bounded_history" if args.benchmark == "alfworld" else args.benchmark),
                "agent_ref": "react",
                "env_options": env_options,
                "agent_options": {},
            },
        }
    )


def safe_arguments(args: argparse.Namespace) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


async def run(args: argparse.Namespace) -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is required")
    started_at = monotonic()
    dataset = build_dataset(args)
    train_tasks = limited(dataset.train_tasks(), args.train_limit)
    validation_tasks = limited(dataset.validation_tasks(), args.validation_limit) if args.validation_gate else []
    test_tasks = limited(dataset.test_tasks(), args.test_limit)
    config = build_run_config(args)
    base_skill = build_skill(args)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / "arguments.json", safe_arguments(args))
    write_json(args.output_dir / "run_config.json", config.model_dump(mode="json"))
    write_json(args.output_dir / "base_skill.json", base_skill.model_dump(mode="json"))

    database = await bootstrap_skill_database(args.output_dir / "state.db")
    call_sink = DatabaseLLMCallSink(database)
    target_client = build_client(args.target_model, args, call_sink=call_sink)
    optimizer_client = ChatModelWithDefaults(
        build_client(args.optimizer_model, args, call_sink=call_sink, temperature=0.0),
        reasoning_effort=args.reasoning_effort,
        max_completion_tokens=args.max_completion_tokens,
    )
    agent = ReactAgent(
        {
            "model": args.target_model,
            "max_turns": args.max_turns,
            "reasoning_effort": args.reasoning_effort,
            "model_kwargs": {"max_completion_tokens": args.max_completion_tokens},
        },
        llm=target_client,
    )
    algorithm = SkillGrpoWithoutReplayBuffer(
        chat_model=optimizer_client,
        agent_resolver=MappingAgentResolver({"react": agent}),
        env_factory=RegistryEnvFactory(),
        logger=AlgorithmLogger(
            algorithm_name=SkillGrpoWithoutReplayBuffer.algorithm_name,
            algorithm_version=config.algorithm.version,
            database=database,
        ),
    )
    try:
        result = await algorithm.evolve(
            SkillGrpoWithoutReplayBufferEvolveInput(
                run_id=args.run_id,
                base_skill=base_skill,
                train_tasks=train_tasks,
                validation_tasks=validation_tasks,
                test_tasks=test_tasks,
                config=config,
            )
        )
    finally:
        await database.close()

    (args.output_dir / "result.json").write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "final_skill.md").write_text(result.final_skill.content, encoding="utf-8")
    test_outcomes = [outcome for outcome in result.rollouts if outcome.spec.phase is RolloutPhase.TEST]
    test_summary = persist_test_artifacts(
        output_dir=args.output_dir / "test",
        outcomes=test_outcomes,
        skill=result.final_skill,
        rollouts_per_task=args.test_rollouts,
    )
    write_json(
        args.output_dir / "summary.json",
        {
            "run_id": result.run_id,
            "changed": result.changed,
            "metrics": result.metrics.model_dump(mode="json"),
            "final_skill_hash": result.final_skill.content_hash,
            "test": test_summary,
            "elapsed_seconds": monotonic() - started_at,
        },
    )


def main(argv: list[str] | None = None) -> None:
    asyncio.run(run(parse_args(argv)))


if __name__ == "__main__":
    main()
