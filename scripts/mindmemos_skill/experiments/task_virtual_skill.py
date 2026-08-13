#!/usr/bin/env python3
"""Run one rollout batch and split its Skill into evidence-grounded subtasks."""

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
from mindmemos_skill.algos.evolve.task_virtual_skill import (
    TaskVirtualSkillEvolve,
    TaskVirtualSkillInput,
    TaskVirtualSkillRefiner,
    TaskVirtualSkillResult,
    TaskVirtualSkillRunConfig,
)
from mindmemos_skill.datasets import (
    ALFWorldPathSplitDataset,
    LiveMathIdSplitDataset,
    SpreadsheetBenchIdSplitDataset,
    TaskDataset,
)
from mindmemos_skill.llm import DatabaseLLMCallSink, LLMCallSink, LLMClient, get_router
from mindmemos_skill.persistence import bootstrap_skill_database
from mindmemos_skill.skill_runtime.runtimes.virtual_components import VirtualComponentsMetadata
from mindmemos_skill.typing import Skill, compute_skill_content_hash


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("alfworld", "livemath", "spreadsheetbench"), required=True)
    parser.add_argument("--env-ref", help="registered Env override; ALFWorld defaults to alfworld_bounded_history")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path)
    parser.add_argument("--initial-skill", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-run-dir", type=Path)

    parser.add_argument("--target-model", default="openai/gpt-5.4-mini")
    parser.add_argument("--summary-model", default="openai/gpt-5.4-mini")
    parser.add_argument("--decomposition-model", default="openai/gpt-5.4-mini")
    parser.add_argument("--reflection-model", default="openai/gpt-5.4-mini")
    parser.add_argument("--change-model", default="openai/gpt-5.4-mini")
    parser.add_argument("--merge-model", default="openai/gpt-5.4-mini")
    parser.add_argument("--api-base", default=os.getenv("OPENAI_ENDPOINT") or os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--request-timeout", type=float, default=1800.0)
    parser.add_argument("--model-retries", type=int, default=3)
    parser.add_argument("--max-completion-tokens", type=int, default=16384)
    parser.add_argument("--summary-max-completion-tokens", type=int, default=2048)

    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--train-rollouts", type=int, default=1)
    parser.add_argument("--summary-sample-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-concurrent-rollouts", type=int, default=16)
    parser.add_argument("--max-concurrent-summaries", type=int, default=8)
    parser.add_argument("--transcript-max-chars", type=int, default=24000)
    parser.add_argument("--rollout-retries", type=int, default=1)
    parser.add_argument("--rollout-timeout", type=float)
    parser.add_argument("--fail-fast", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-virtual-skills", type=int, default=12)
    parser.add_argument("--max-initial-components", type=int, default=3)
    parser.add_argument("--retry-rounds", type=int, default=1)
    parser.add_argument("--success-reward", type=float, default=1.0)
    parser.add_argument("--max-concurrent-reflections", type=int, default=8)
    parser.add_argument("--max-trajectory-chars", type=int, default=24000)
    parser.add_argument("--max-previous-answer-chars", type=int, default=8000)
    parser.add_argument("--max-reflection-chars", type=int, default=4000)
    parser.add_argument("--max-concurrent-changes", type=int, default=8)
    parser.add_argument("--max-concurrent-merges", type=int, default=4)

    parser.add_argument("--max-turns", type=int, required=True)
    parser.add_argument("--shell-timeout", type=int, default=120)
    parser.add_argument("--env-seed", type=int)
    parser.add_argument("--use-theorem", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-sketch", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--shuffle-choices", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-limit", type=int)
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


def build_client(model: str, args: argparse.Namespace, *, call_sink: LLMCallSink) -> LLMClient:
    endpoint = {
        "model": model,
        "api_key": os.getenv("OPENAI_API_KEY"),
        "api_base": args.api_base,
        "timeout": args.request_timeout,
        "num_retries": args.model_retries,
        "temperature": 0.0,
    }
    router, retries = get_router({"endpoints": [endpoint]}, model, num_retries=args.model_retries)
    return LLMClient(router, default_model=model, max_attempts=retries + 1, call_sink=call_sink)


def build_skill(args: argparse.Namespace) -> Skill:
    blob = {"SKILL.md": args.initial_skill.read_text(encoding="utf-8")}
    now = datetime.now(UTC)
    return Skill(
        skill_id=f"{args.benchmark}-task-virtual-skill",
        version_id=f"{args.run_id}:base",
        version_label="0.1.0",
        content_hash=compute_skill_content_hash(blob),
        name=args.benchmark,
        blob=blob,
        created_at=now,
    )


def build_config(args: argparse.Namespace) -> TaskVirtualSkillRunConfig:
    env_options: dict[str, Any] = {"max_turns": args.max_turns}
    if args.benchmark == "spreadsheetbench":
        env_options["shell_timeout_seconds"] = args.shell_timeout
    elif args.benchmark == "alfworld":
        env_options["seed"] = args.seed if args.env_seed is None else args.env_seed
    else:
        env_options.update({"use_theorem": args.use_theorem, "use_sketch": args.use_sketch})
    return TaskVirtualSkillRunConfig.model_validate(
        {
            "batch": {
                "batch_size": args.batch_size,
                "rollouts_per_task": args.train_rollouts,
                "seed": args.seed,
            },
            "rollout": {
                "max_concurrent_rollouts": args.max_concurrent_rollouts,
                "timeout_seconds": args.rollout_timeout,
                "retry": {"max_attempts": args.rollout_retries},
                "fail_fast": args.fail_fast,
                "workspace_root": str(args.output_dir / "workspace"),
            },
            "summary": {
                "sample_size": args.summary_sample_size,
                "max_concurrent_summaries": args.max_concurrent_summaries,
                "transcript_max_chars": args.transcript_max_chars,
            },
            "decomposition": {
                "max_virtual_skills": args.max_virtual_skills,
                "max_initial_components": args.max_initial_components,
            },
            "refinement": {
                "retry_rounds": args.retry_rounds,
                "success_reward": args.success_reward,
                "max_concurrent_reflections": args.max_concurrent_reflections,
                "max_trajectory_chars": args.max_trajectory_chars,
                "max_previous_answer_chars": args.max_previous_answer_chars,
                "max_reflection_chars": args.max_reflection_chars,
                "max_concurrent_changes": args.max_concurrent_changes,
                "max_concurrent_merges": args.max_concurrent_merges,
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


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


async def run(args: argparse.Namespace) -> dict[str, Any]:
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is required")
    if not args.api_base:
        raise ValueError("--api-base, OPENAI_ENDPOINT, or OPENAI_BASE_URL is required")
    if args.source_run_dir is not None:
        return await run_refinement(args)
    dataset = build_dataset(args)
    tasks = dataset.train_tasks()
    if args.train_limit is not None:
        tasks = tasks[: args.train_limit]
    skill = build_skill(args)
    config = build_config(args)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / "run_config.json", config.model_dump(mode="json"))
    (args.output_dir / "source_skill.md").write_text(skill.content, encoding="utf-8")

    database = await bootstrap_skill_database(args.output_dir / "state.db")
    sink = DatabaseLLMCallSink(database)
    target_client = build_client(args.target_model, args, call_sink=sink)
    summary_client = build_client(args.summary_model, args, call_sink=sink)
    decomposition_client = build_client(args.decomposition_model, args, call_sink=sink)
    agent = ReactAgent(
        {
            "model": args.target_model,
            "max_turns": args.max_turns,
            "reasoning_effort": args.reasoning_effort,
            "model_kwargs": {"max_completion_tokens": args.max_completion_tokens},
        },
        llm=target_client,
    )
    algorithm = TaskVirtualSkillEvolve(
        summary_model=summary_client,
        decomposition_model=decomposition_client,
        agent_resolver=MappingAgentResolver({"react": agent}),
        env_factory=RegistryEnvFactory(),
        summary_chat_kwargs={
            "reasoning_effort": args.reasoning_effort,
            "max_completion_tokens": args.summary_max_completion_tokens,
        },
        decomposition_chat_kwargs={
            "reasoning_effort": args.reasoning_effort,
            "max_completion_tokens": args.max_completion_tokens,
        },
    )
    try:
        result = await algorithm.evolve(
            TaskVirtualSkillInput(
                run_id=args.run_id,
                base_skill=skill,
                train_tasks=tasks,
                config=config,
            )
        )
    finally:
        await database.close()

    (args.output_dir / "result.json").write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    write_json(
        args.output_dir / "trajectory_summaries.json",
        [item.model_dump(mode="json") for item in result.trajectory_summaries],
    )
    summary_dir = args.output_dir / "trajectory_summaries"
    summary_dir.mkdir()
    for item in result.trajectory_summaries:
        write_json(summary_dir / f"{item.trajectory_id}.json", item.model_dump(mode="json"))
    write_json(args.output_dir / "sampled_trajectory_ids.json", result.sampled_trajectory_ids)
    (args.output_dir / "raw_decomposition_response.md").write_text(
        result.raw_decomposition_response.rstrip() + "\n", encoding="utf-8"
    )
    if result.candidate is not None:
        (args.output_dir / "SKILL.md").write_text(result.candidate.blob["SKILL.md"], encoding="utf-8")
        write_json(args.output_dir / "runtime_metadata.json", result.candidate.runtime_metadata)
        for artifact in result.artifacts:
            target = args.output_dir / artifact.relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(artifact.markdown, encoding="utf-8")
    summary = {
        "run_id": args.run_id,
        "algorithm": TaskVirtualSkillEvolve.algorithm_name,
        "rollout_count": len(result.rollouts),
        "trajectory_count": len(result.trajectories),
        "trajectory_summary_count": len(result.trajectory_summaries),
        "failed_summary_count": len(result.failed_summary_trajectory_ids),
        "sampled_summary_count": len(result.sampled_trajectory_ids),
        "virtual_skill_count": len(result.artifacts),
        "changed": result.changed,
        "virtual_skill_files": [item.relative_path for item in result.artifacts],
    }
    write_json(args.output_dir / "summary.json", summary)
    return summary


def persist_virtual_skill_set(root: Path, skill: Skill) -> None:
    root.mkdir(parents=True, exist_ok=False)
    (root / "SKILL.md").write_text(skill.content, encoding="utf-8")
    metadata = VirtualComponentsMetadata.model_validate(skill.runtime_metadata)
    write_json(root / "runtime_metadata.json", metadata.model_dump(mode="json"))
    component_root = root / "virtual_skills"
    component_root.mkdir()
    for component in metadata.components:
        (component_root / f"{component.component_id}.md").write_text(
            component.content.rstrip() + "\n",
            encoding="utf-8",
        )


async def run_refinement(args: argparse.Namespace) -> dict[str, Any]:
    assert args.source_run_dir is not None
    source_path = args.source_run_dir / "result.json"
    if not source_path.is_file():
        raise FileNotFoundError(f"source task_virtual_skill result does not exist: {source_path}")
    source = TaskVirtualSkillResult.model_validate_json(source_path.read_text(encoding="utf-8"))
    config = build_config(args)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / "run_config.json", config.model_dump(mode="json"))
    write_json(args.output_dir / "source_run.json", {"path": str(args.source_run_dir), "run_id": source.run_id})

    database = await bootstrap_skill_database(args.output_dir / "state.db")
    sink = DatabaseLLMCallSink(database)
    target_client = build_client(args.target_model, args, call_sink=sink)
    summary_client = build_client(args.summary_model, args, call_sink=sink)
    reflection_client = build_client(args.reflection_model, args, call_sink=sink)
    change_client = build_client(args.change_model, args, call_sink=sink)
    merge_client = build_client(args.merge_model, args, call_sink=sink)
    agent = ReactAgent(
        {
            "model": args.target_model,
            "max_turns": args.max_turns,
            "reasoning_effort": args.reasoning_effort,
            "model_kwargs": {"max_completion_tokens": args.max_completion_tokens},
        },
        llm=target_client,
    )
    refiner = TaskVirtualSkillRefiner(
        config=config,
        reflection_model=reflection_client,
        summary_model=summary_client,
        change_model=change_client,
        merge_model=merge_client,
        agent_resolver=MappingAgentResolver({"react": agent}),
        env_factory=RegistryEnvFactory(),
        chat_kwargs={
            "reasoning_effort": args.reasoning_effort,
            "max_completion_tokens": args.max_completion_tokens,
        },
    )
    try:
        result = await refiner.refine(run_id=args.run_id, source=source)
    finally:
        await database.close()

    (args.output_dir / "result.json").write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    write_json(args.output_dir / "changes.json", [item.model_dump(mode="json") for item in result.changes])
    write_json(args.output_dir / "merges.json", [item.model_dump(mode="json") for item in result.merges])
    write_json(
        args.output_dir / "retry_trajectory_summaries.json",
        [item.model_dump(mode="json") for item in result.retry_summaries],
    )
    persist_virtual_skill_set(args.output_dir / "before", result.before_skill)
    persist_virtual_skill_set(args.output_dir / "after", result.after_skill)
    summary = {
        "run_id": args.run_id,
        "source_run_id": source.run_id,
        "retry_rollout_count": len(result.retry_rollouts),
        "retry_summary_count": len(result.retry_summaries),
        "change_decision_count": len(result.changes),
        "create_count": sum(item.operation == "create" for item in result.changes),
        "update_count": sum(item.operation == "update" for item in result.changes),
        "noop_count": sum(item.operation == "noop" for item in result.changes),
        "changed_virtual_skill_count": len(result.merges),
        "changed_virtual_skill_ids": [item.skill_id for item in result.merges],
        "test_started": False,
    }
    write_json(args.output_dir / "summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    print(json.dumps(asyncio.run(run(parse_args(argv))), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
