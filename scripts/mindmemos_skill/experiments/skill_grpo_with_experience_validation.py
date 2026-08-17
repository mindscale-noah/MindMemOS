#!/usr/bin/env python3
"""Script-side experience-validated, replay-free Skill GRPO experiment."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from typing import Any

from mindmemos_skill.agents.react import ReactAgent
from mindmemos_skill.algos.evolve.skill_grpo_with_experience_validation import (
    SkillGrpoWithExperienceValidation,
    SkillGrpoWithExperienceValidationEvolveInput,
    SkillGrpoWithExperienceValidationRunConfig,
)
from mindmemos_skill.algos.evolve.skill_grpo_with_replay_buffer import MappingAgentResolver, RegistryEnvFactory
from mindmemos_skill.algos.evolve.skill_grpo_with_replay_buffer.contracts import RolloutPhase
from mindmemos_skill.datasets import TaskDataset
from mindmemos_skill.llm import DatabaseLLMCallSink, LLMCallSink, LLMClient, get_router
from mindmemos_skill.logging import AlgorithmLogger
from mindmemos_skill.persistence import bootstrap_skill_database
from mindmemos_skill.typing import Skill, compute_skill_content_hash

from .component_selection import build_registered_dataset, build_registered_env_options, json_object
from .skill_evaluation import persist_test_artifacts


class ChatModelWithDefaults:
    def __init__(self, client: LLMClient, **defaults: Any) -> None:
        self._client = client
        self._defaults = defaults

    async def chat(self, task: str, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        return await self._client.chat(task, messages, **{**self._defaults, **kwargs})


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, help="artifact label and built-in component default")
    parser.add_argument("--dataset-ref", help="registered Dataset name")
    parser.add_argument("--dataset-options", type=json_object, default={}, help="JSON options for the Dataset factory")
    parser.add_argument("--env-ref", help="registered Env override; ALFWorld defaults to alfworld_bounded_history")
    parser.add_argument("--env-options", type=json_object, default={}, help="JSON options for the Env factory")
    parser.add_argument("--data-root", type=Path)
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
    parser.add_argument("--test-rollouts", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-concurrent-rollouts", type=int, default=32)
    parser.add_argument("--max-concurrent-extractions", type=int, default=16)
    parser.add_argument("--reflection", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-concurrent-reflections", type=int, default=8)
    parser.add_argument("--rollout-retries", type=int, default=3)
    parser.add_argument("--rollout-timeout", type=float)
    parser.add_argument("--fail-fast", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-turns", type=int, required=True)
    parser.add_argument("--docs-root", type=Path)
    parser.add_argument("--env-seed", type=int)
    parser.add_argument("--shell-timeout", type=int, default=120)
    parser.add_argument("--use-theorem", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-sketch", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--shuffle-choices", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-experiences-per-task", type=int, default=3)
    parser.add_argument("--max-tool-observation-chars", type=int)
    parser.add_argument(
        "--previous-batch-invalid-experience-feedback",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--max-patch-edits", type=int, default=6)
    parser.add_argument("--patch-attempts", type=int, default=2)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--test-limit", type=int)
    return parser.parse_args(argv)


def build_dataset(args: argparse.Namespace) -> TaskDataset:
    return build_registered_dataset(
        benchmark=args.benchmark,
        dataset_ref=args.dataset_ref,
        dataset_options=args.dataset_options,
        data_root=args.data_root,
        split_dir=args.split_dir,
        seed=args.seed,
        shuffle_choices=args.shuffle_choices,
    )


def build_client(
    model: str, args: argparse.Namespace, *, call_sink: LLMCallSink, temperature: float | None = None
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


def build_run_config(args: argparse.Namespace) -> SkillGrpoWithExperienceValidationRunConfig:
    env_ref, env_options = build_registered_env_options(
        benchmark=args.benchmark,
        env_ref=args.env_ref,
        env_options=args.env_options,
        max_turns=args.max_turns,
        env_seed=args.seed if args.env_seed is None else args.env_seed,
        shell_timeout=args.shell_timeout,
        use_theorem=args.use_theorem,
        use_sketch=args.use_sketch,
        docs_root=args.docs_root,
    )
    return SkillGrpoWithExperienceValidationRunConfig.model_validate(
        {
            "algorithm": {
                "reflection": {
                    "enabled": args.reflection,
                    "max_concurrent_reflections": args.max_concurrent_reflections,
                },
                "experience": {
                    "max_experiences_per_task": args.max_experiences_per_task,
                    "max_concurrent_extractions": args.max_concurrent_extractions,
                    "max_tool_observation_chars": args.max_tool_observation_chars,
                },
                "previous_batch_invalid_experience_feedback": {
                    "enabled": args.previous_batch_invalid_experience_feedback,
                },
                "patch": {"max_edits": args.max_patch_edits, "max_attempts": args.patch_attempts},
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
                "experience_validation": {"name": "fixed_group", "params": {"group_size": 1}},
                "test": {"name": "fixed_group", "params": {"group_size": args.test_rollouts}},
            },
            "dataset": {
                "env_ref": env_ref,
                "agent_ref": "react",
                "env_options": env_options,
                "agent_options": {},
            },
        }
    )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


class BatchCheckpointWriter:
    """Persist the exact Skill and audit record immediately after every batch."""

    def __init__(self, *, output_dir: Path) -> None:
        self.output_dir = output_dir

    async def handle(self, event: Any) -> None:
        if event.name == "invalid_experience_feedback_generated":
            batch_index = int(event.payload["source_batch_index"])
            batch_dir = self.output_dir / "batch_artifacts" / f"batch_{batch_index + 1:04d}"
            write_json_atomic(
                batch_dir / "invalid_experience_feedback.json",
                {"run_id": event.run_id, **event.payload},
            )
            return
        if event.name != "checkpoint_ready":
            return
        batch_index = int(event.payload["batch_index"])
        checkpoint = {
            "run_id": event.run_id,
            "batch_index": batch_index,
            "current_skill": event.payload["current_skill"],
            "batch": event.payload["batch"],
        }
        batch_dir = self.output_dir / "batch_artifacts" / f"batch_{batch_index + 1:04d}"
        write_json_atomic(batch_dir / "checkpoint.json", checkpoint)
        write_json_atomic(batch_dir / "skill.json", checkpoint["current_skill"])
        write_json_atomic(batch_dir / "batch.json", checkpoint["batch"])
        write_text_atomic(batch_dir / "skill.md", checkpoint["current_skill"]["blob"]["SKILL.md"])
        write_json_atomic(self.output_dir / "checkpoint.json", checkpoint)


async def run(args: argparse.Namespace) -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is required")
    if args.train_limit is not None and args.train_limit < 1:
        raise ValueError("train-limit must be positive")
    started_at = monotonic()
    dataset = build_dataset(args)
    train_tasks = dataset.train_tasks()
    if args.train_limit is not None:
        train_tasks = train_tasks[: args.train_limit]
    validation_tasks = dataset.validation_tasks()
    test_tasks = dataset.test_tasks()
    if args.test_limit is not None:
        if args.test_limit < 1:
            raise ValueError("test-limit must be positive")
        test_tasks = test_tasks[: args.test_limit]
    config = build_run_config(args)
    base_skill = build_skill(args)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output_dir / "arguments.json",
        {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    )
    write_json(args.output_dir / "run_config.json", config.model_dump(mode="json"))
    write_json(args.output_dir / "base_skill.json", base_skill.model_dump(mode="json"))
    write_json(
        args.output_dir / "dataset_counts.json",
        {"train": len(train_tasks), "validation": len(validation_tasks), "test": len(test_tasks)},
    )

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
            "max_turns": 1,
            "reasoning_effort": args.reasoning_effort,
            "model_kwargs": {"max_completion_tokens": args.max_completion_tokens},
        },
        llm=target_client,
    )
    checkpoint_writer = BatchCheckpointWriter(output_dir=args.output_dir)
    algorithm = SkillGrpoWithExperienceValidation(
        chat_model=optimizer_client,
        agent_resolver=MappingAgentResolver({"react": agent}),
        env_factory=RegistryEnvFactory(),
        on_event=checkpoint_writer.handle,
        logger=AlgorithmLogger(
            algorithm_name=SkillGrpoWithExperienceValidation.algorithm_name,
            algorithm_version=config.algorithm.version,
            database=database,
        ),
    )
    try:
        result = await algorithm.evolve(
            SkillGrpoWithExperienceValidationEvolveInput(
                run_id=args.run_id,
                base_skill=base_skill,
                train_tasks=train_tasks,
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
    accepted_by_source = Counter(
        experience.source.value for batch in result.batches for experience in batch.accepted_experiences
    )
    extracted_by_source = Counter(
        experience.source.value for batch in result.batches for experience in batch.experiences
    )
    rejected_by_source = Counter(
        validation.source.value
        for batch in result.batches
        for validation in batch.experience_validations
        if validation.decision.value == "rejected"
    )
    batch_summaries = []
    for batch in result.batches:
        extracted = Counter(experience.source.value for experience in batch.experiences)
        accepted = Counter(experience.source.value for experience in batch.accepted_experiences)
        rejected = Counter(
            validation.source.value
            for validation in batch.experience_validations
            if validation.decision.value == "rejected"
        )
        batch_summaries.append(
            {
                "epoch": batch.epoch,
                "batch_index": batch.batch_index,
                "extracted_by_source": dict(extracted),
                "accepted_by_source": dict(accepted),
                "rejected_by_source": dict(rejected),
                "applied_edits": len(batch.applied_edits),
                "patch_decision": batch.patch_decision.value,
                "skill_hash_after": batch.skill_hash_after,
            }
        )
    write_json(
        args.output_dir / "summary.json",
        {
            "run_id": result.run_id,
            "changed": result.changed,
            "metrics": result.metrics.model_dump(mode="json"),
            "extracted_by_source": dict(extracted_by_source),
            "accepted_by_source": dict(accepted_by_source),
            "rejected_by_source": dict(rejected_by_source),
            "batches": batch_summaries,
            "final_skill_hash": result.final_skill.content_hash,
            "test": test_summary,
            "elapsed_seconds": monotonic() - started_at,
        },
    )


def main(argv: list[str] | None = None) -> None:
    asyncio.run(run(parse_args(argv)))


if __name__ == "__main__":
    main()
