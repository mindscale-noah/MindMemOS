#!/usr/bin/env python3
"""Script-side ALFWorld trajectory library and paired top-k retrieval experiment."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mindmemos_skill.agents.react import ReactAgent
from mindmemos_skill.algos.evolve.skill_grpo_with_replay_buffer import MappingAgentResolver, RegistryEnvFactory
from mindmemos_skill.algos.evolve.trajectory_memory import (
    TrajectoryMemoryEvolve,
    TrajectoryMemoryEvolveInput,
    TrajectoryMemoryRunConfig,
    reconstruct_replay_free_trajectories,
)
from mindmemos_skill.datasets import ALFWorldPathSplitDataset
from mindmemos_skill.llm import DatabaseLLMCallSink, EmbedClient, LLMCallSink, LLMClient, get_router, llm_run_context
from mindmemos_skill.persistence import bootstrap_skill_database
from mindmemos_skill.typing import Skill, compute_skill_content_hash


class ChatModelWithDefaults:
    def __init__(self, client: LLMClient, **defaults: Any) -> None:
        self._client = client
        self._defaults = defaults

    async def chat(self, task: str, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        return await self._client.chat(task, messages, **{**self._defaults, **kwargs})


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--initial-skill", type=Path, required=True)
    parser.add_argument("--source-run-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--target-model", default="openai/gpt-5.4-mini")
    parser.add_argument("--summary-model", default="openai/gpt-5.4-mini")
    parser.add_argument("--embedding-model", default="openai/text-embedding-3-small")
    parser.add_argument("--api-base", default=os.getenv("OPENAI_ENDPOINT") or os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--request-timeout", type=float, default=1800.0)
    parser.add_argument("--model-retries", type=int, default=3)
    parser.add_argument("--max-completion-tokens", type=int, default=16384)
    parser.add_argument("--summary-max-completion-tokens", type=int, default=2048)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--train-rollouts", type=int, default=1)
    parser.add_argument("--test-rollouts", type=int, default=1)
    parser.add_argument("--run-baseline", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-concurrent-rollouts", type=int, default=32)
    parser.add_argument("--queue-capacity", type=int, default=32)
    parser.add_argument("--max-concurrent-summaries", type=int, default=8)
    parser.add_argument("--rollout-retries", type=int, default=1)
    parser.add_argument("--rollout-timeout", type=float)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--env-seed", type=int, default=42)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--test-limit", type=int)
    return parser.parse_args(argv)


def build_client(model: str, args: argparse.Namespace, *, call_sink: LLMCallSink) -> LLMClient:
    endpoint = {
        "model": model,
        "api_key": os.getenv("OPENAI_API_KEY"),
        "api_base": args.api_base,
        "timeout": args.request_timeout,
        "num_retries": args.model_retries,
    }
    router, retries = get_router({"endpoints": [endpoint]}, model, num_retries=args.model_retries)
    return LLMClient(router, default_model=model, max_attempts=retries + 1, call_sink=call_sink)


def build_embed_client(args: argparse.Namespace, *, call_sink: LLMCallSink) -> EmbedClient:
    endpoint = {
        "model": args.embedding_model,
        "api_key": os.getenv("OPENAI_API_KEY"),
        "api_base": args.api_base,
        "timeout": args.request_timeout,
        "num_retries": args.model_retries,
    }
    router, _ = get_router(
        {"endpoints": [endpoint]},
        args.embedding_model,
        num_retries=args.model_retries,
    )
    return EmbedClient(router, default_model=args.embedding_model, call_sink=call_sink)


def build_skill(args: argparse.Namespace) -> Skill:
    blob = {"SKILL.md": args.initial_skill.read_text(encoding="utf-8")}
    now = datetime.now(UTC)
    return Skill(
        skill_id="alfworld-skill",
        version_id=f"{args.run_id}:base",
        version_label="0.1.0",
        content_hash=compute_skill_content_hash(blob),
        name="alfworld",
        blob=blob,
        created_at=now,
    )


def limited(items: list[Any], limit: int | None) -> list[Any]:
    return items if limit is None else items[:limit]


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_rollouts(path: Path, outcomes: list[Any]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for outcome in outcomes:
            trajectory = outcome.trajectory
            payload = {
                "task_id": outcome.spec.task.task_id,
                "rollout_id": outcome.spec.rollout_id,
                "sample_index": outcome.spec.sample_index,
                "reward": trajectory.reward.score if trajectory is not None else None,
                "turns": trajectory.execution.n_turn if trajectory is not None else None,
                "error": trajectory.execution.error_info if trajectory is not None else None,
                "trajectory": trajectory.events if trajectory is not None else [],
            }
            stream.write(json.dumps(payload, ensure_ascii=False) + "\n")


async def run(args: argparse.Namespace) -> dict[str, Any]:
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is required")
    if not args.api_base:
        raise ValueError("--api-base, OPENAI_ENDPOINT, or OPENAI_BASE_URL is required")
    os.environ["ALFWORLD_DATA"] = str(args.data_root.expanduser().resolve())
    dataset = ALFWorldPathSplitDataset(split_dir=args.split_dir, alfworld_data=args.data_root)
    train_tasks = limited(dataset.train_tasks(), args.train_limit)
    test_tasks = limited(dataset.test_tasks(), args.test_limit)
    base_skill = build_skill(args)
    precollected = []
    if args.source_run_dir is not None:
        precollected = reconstruct_replay_free_trajectories(
            args.source_run_dir / "state.db",
            tasks=train_tasks,
        )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output_dir / "arguments.json", {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    )
    write_json(
        args.output_dir / "reconstruction_summary.json",
        {
            "source_run_dir": str(args.source_run_dir) if args.source_run_dir else None,
            "trajectory_count": len(precollected),
            "task_count": len({item.task.task_id for item in precollected}),
            "successful_trajectories": sum((item.reward_score or 0.0) >= 1.0 for item in precollected),
        },
    )
    database = await bootstrap_skill_database(args.output_dir / "state.db")
    sink = DatabaseLLMCallSink(database)
    target_client = build_client(args.target_model, args, call_sink=sink)
    summary_client = ChatModelWithDefaults(
        build_client(args.summary_model, args, call_sink=sink),
        reasoning_effort=args.reasoning_effort,
        max_completion_tokens=args.summary_max_completion_tokens,
        temperature=0.0,
    )
    embed_client = build_embed_client(args, call_sink=sink)
    agent = ReactAgent(
        {
            "model": args.target_model,
            "max_turns": 1,
            "reasoning_effort": args.reasoning_effort,
            "model_kwargs": {"max_completion_tokens": args.max_completion_tokens},
        },
        llm=target_client,
    )
    config = TrajectoryMemoryRunConfig.model_validate(
        {
            "algorithm": {
                "top_k": args.top_k,
                "train_rollouts_per_task": args.train_rollouts,
                "test_rollouts_per_task": args.test_rollouts,
                "max_concurrent_summaries": args.max_concurrent_summaries,
                "run_baseline": args.run_baseline,
            },
            "rollout": {
                "max_concurrent_rollouts": args.max_concurrent_rollouts,
                "queue_capacity": args.queue_capacity,
                "timeout_seconds": args.rollout_timeout,
                "retry": {"max_attempts": args.rollout_retries},
                "workspace_root": args.output_dir / "workspace",
            },
            "dataset": {
                "env_ref": "alfworld",
                "agent_ref": "react",
                "env_options": {"max_steps": args.max_steps, "seed": args.env_seed},
                "agent_options": {},
            },
        }
    )
    write_json(args.output_dir / "run_config.json", config.model_dump(mode="json"))
    algorithm = TrajectoryMemoryEvolve(
        chat_model=summary_client,
        embedding_model=embed_client,
        agent_resolver=MappingAgentResolver({"react": agent}),
        env_factory=RegistryEnvFactory(),
    )
    try:
        with llm_run_context(args.run_id):
            result = await algorithm.evolve(
                TrajectoryMemoryEvolveInput(
                    run_id=args.run_id,
                    base_skill=base_skill,
                    train_tasks=train_tasks,
                    test_tasks=test_tasks,
                    config=config,
                    precollected_train_trajectories=precollected,
                )
            )
    finally:
        await database.close()

    write_json(args.output_dir / "result.json", result.model_dump(mode="json"))
    write_json(
        args.output_dir / "trajectory_library.json",
        [item.model_dump(mode="json", exclude={"embedding"}) for item in result.memory_bank],
    )
    write_json(
        args.output_dir / "trajectory_library_embeddings.json",
        [
            {"memory_id": item.memory_id, "retrieval_key": item.retrieval_key, "embedding": item.embedding}
            for item in result.memory_bank
        ],
    )
    write_json(
        args.output_dir / "retrievals.json",
        [
            {
                "task_id": record.task_id,
                "retrieval_key": record.retrieval_key,
                "memories": [
                    {
                        "rank": value.rank,
                        "similarity": value.similarity,
                        "memory_id": value.item.memory_id,
                        "source_task_id": value.item.source_task_id,
                        "title": value.item.summary.title,
                    }
                    for value in record.memories
                ],
            }
            for record in result.retrievals
        ],
    )
    write_rollouts(args.output_dir / "baseline_results.jsonl", result.baseline_rollouts)
    write_rollouts(args.output_dir / "memory_results.jsonl", result.memory_rollouts)
    summary = {
        "run_id": args.run_id,
        "source_trajectory_count": len(precollected),
        "memory_item_count": len(result.memory_bank),
        "top_k": args.top_k,
        "metrics": result.metrics.model_dump(mode="json"),
    }
    write_json(args.output_dir / "summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> None:
    print(json.dumps(asyncio.run(run(parse_args(argv))), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
