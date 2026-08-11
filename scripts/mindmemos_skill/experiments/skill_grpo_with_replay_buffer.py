#!/usr/bin/env python3
"""Script-side SkillGrpoWithReplayBuffer experiment on a registered benchmark."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from typing import Any

from mindmemos_skill.agents.react import ReactAgent
from mindmemos_skill.algos.evolve.skill_grpo_with_replay_buffer import (
    EvolutionState,
    MappingAgentResolver,
    RegistryEnvFactory,
    SkillGrpoEvolveInput,
    SkillGrpoRunConfig,
    SkillGrpoWithReplayBuffer,
)
from mindmemos_skill.algos.evolve.skill_grpo_with_replay_buffer.contracts import RolloutPhase
from mindmemos_skill.algos.evolve.skill_grpo_with_replay_buffer.state import (
    config_fingerprint,
    input_fingerprint,
    validate_resume,
)
from mindmemos_skill.datasets import LiveMathIdSplitDataset, SpreadsheetBenchIdSplitDataset, TaskDataset
from mindmemos_skill.llm import DatabaseLLMCallSink, EmbedClient, LLMCallSink, LLMClient, get_router
from mindmemos_skill.logging import AlgorithmLogger
from mindmemos_skill.persistence import bootstrap_skill_database
from mindmemos_skill.typing import Skill, compute_skill_content_hash

from .skill_evaluation import persist_test_artifacts

_ROLLOUT_PHASE_DIRECTORIES = ("train", "validation", "test", "ablation_before", "ablation_after")


class ChatModelWithDefaults:
    """Attach optimizer-only request options to every algorithm chat call."""

    def __init__(self, client: LLMClient, **defaults: Any) -> None:
        self.client = client
        self.defaults = defaults

    async def chat(self, task: str, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        return await self.client.chat(task, messages, **{**self.defaults, **kwargs})


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remainder = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{remainder:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m{remainder:02d}s"


def progress_log(message: str, started_at: float) -> None:
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp} +{format_duration(monotonic() - started_at)}] {message}", flush=True)


def write_json_atomic(path: Path, payload: Any) -> None:
    """Replace a JSON artifact only after its complete contents reach disk."""

    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def write_text_atomic(path: Path, content: str) -> None:
    """Replace a text artifact only after its complete contents reach disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_resume_artifacts(output_dir: Path) -> tuple[EvolutionState, Skill]:
    if not output_dir.is_dir():
        raise FileNotFoundError(f"resume output directory does not exist: {output_dir}")
    checkpoint_path = output_dir / "checkpoint.json"
    base_skill_path = output_dir / "base_skill.json"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"resume checkpoint does not exist: {checkpoint_path}")
    if not base_skill_path.is_file():
        raise FileNotFoundError(
            f"resume base skill does not exist: {base_skill_path}; this run predates runner resume support"
        )
    return (
        EvolutionState.model_validate_json(checkpoint_path.read_text(encoding="utf-8")),
        Skill.model_validate_json(base_skill_path.read_text(encoding="utf-8")),
    )


def validate_resume_arguments(output_dir: Path, args: argparse.Namespace) -> None:
    arguments_path = output_dir / "arguments.json"
    if not arguments_path.is_file():
        raise FileNotFoundError(f"resume arguments do not exist: {arguments_path}")
    saved = json.loads(arguments_path.read_text(encoding="utf-8"))
    current = safe_arguments(args)
    ignored = {"resume"}
    mismatches = sorted(
        key for key in saved.keys() | current.keys() if key not in ignored and saved.get(key) != current.get(key)
    )
    if mismatches:
        raise ValueError(f"resume argument mismatch: {', '.join(mismatches)}")


def clear_incomplete_rollout_workspaces(workspace_root: Path, preserved_rollout_ids: set[str]) -> int:
    """Remove only generated rollout directories absent from the durable checkpoint."""

    if not workspace_root.is_dir():
        return 0
    removed = 0
    for phase_name in _ROLLOUT_PHASE_DIRECTORIES:
        phase_dir = workspace_root / phase_name
        if not phase_dir.is_dir():
            continue
        for task_dir in phase_dir.iterdir():
            if not task_dir.is_dir():
                continue
            for rollout_dir in task_dir.iterdir():
                if rollout_dir.is_dir() and rollout_dir.name not in preserved_rollout_ids:
                    shutil.rmtree(rollout_dir)
                    removed += 1
    return removed


class CheckpointWriter:
    """Persist resumable checkpoints and per-batch Skill artifacts."""

    def __init__(self, *, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.saved_batch_skills: set[int] = set()

    async def handle(self, event: Any) -> None:
        if event.name != "checkpoint_ready":
            return
        checkpoint = event.payload.get("state")
        if checkpoint is None:
            return
        write_json_atomic(self.output_dir / "checkpoint.json", checkpoint)
        self._save_batch_skill(checkpoint)

    def _save_batch_skill(self, checkpoint: dict[str, Any]) -> tuple[int, Path] | None:
        batches = checkpoint.get("batches") or []
        if not batches:
            return None
        batch_index = int(batches[-1]["batch_index"])
        if batch_index in self.saved_batch_skills:
            return None
        content = checkpoint["current_skill"]["blob"]["SKILL.md"]
        skill_path = self.output_dir / "batch_artifacts" / f"batch_{batch_index + 1:04d}" / "skill.md"
        write_text_atomic(skill_path, content)
        self.saved_batch_skills.add(batch_index)
        return batch_index, skill_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("spreadsheetbench", "livemath"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path)
    parser.add_argument("--initial-skill", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--resume", action="store_true", help="resume from OUTPUT_DIR/checkpoint.json")

    parser.add_argument("--target-model", default="openai/gpt-5.4-mini")
    parser.add_argument("--optimizer-model", default="openai/gpt-5.4-mini")
    parser.add_argument("--embedding-model", default="openai/text-embedding-3-small")
    parser.add_argument("--api-base", default=os.getenv("OPENAI_ENDPOINT") or os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--request-timeout", type=float, default=1800.0)
    parser.add_argument("--model-retries", type=int, default=10)
    parser.add_argument("--max-completion-tokens", type=int, default=16384)

    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--train-rollouts", type=int, default=4)
    parser.add_argument("--validation-rollouts", type=int, default=3)
    parser.add_argument("--test-rollouts", type=int, default=3)
    parser.add_argument("--validate-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-concurrent-rollouts", type=int, default=32)
    parser.add_argument("--max-concurrent-extractions", type=int, default=16)
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
    parser.add_argument("--replay-similarity-threshold", type=float, default=0.9)
    parser.add_argument("--min-cluster-edits", type=int, default=2)
    parser.add_argument("--replay-capacity", type=int, default=512)
    parser.add_argument("--replay-max-uses", type=int, default=10)
    parser.add_argument("--ablation-source-cases", type=int, default=8)
    parser.add_argument("--ablation-rollouts", type=int, default=1)
    parser.add_argument("--ablation-commit-topk", type=int, default=5)
    parser.add_argument("--ablation-positive-only", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--validation-limit", type=int)
    parser.add_argument("--test-limit", type=int)
    return parser.parse_args(argv)


def build_dataset(args: argparse.Namespace) -> TaskDataset:
    if args.benchmark == "spreadsheetbench":
        return SpreadsheetBenchIdSplitDataset(data_root=args.data_root, split_dir=args.split_dir)
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


def build_embedding_client(args: argparse.Namespace, *, call_sink: LLMCallSink) -> EmbedClient:
    endpoint = {
        "model": args.embedding_model,
        "api_key": os.getenv("OPENAI_API_KEY"),
        "api_base": args.api_base,
        "timeout": args.request_timeout,
        "num_retries": args.model_retries,
        "encoding_format": "float",
    }
    router, _ = get_router({"endpoints": [endpoint]}, args.embedding_model, num_retries=args.model_retries)
    return EmbedClient(router, default_model=args.embedding_model, call_sink=call_sink)


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


def build_run_config(args: argparse.Namespace) -> SkillGrpoRunConfig:
    env_options: dict[str, Any] = {"max_turns": args.max_turns}
    if args.benchmark == "spreadsheetbench":
        env_options["shell_timeout_seconds"] = args.shell_timeout
    else:
        env_options.update({"use_theorem": args.use_theorem, "use_sketch": args.use_sketch})
    return SkillGrpoRunConfig.model_validate(
        {
            "algorithm": {
                "experience": {
                    "max_experiences_per_task": args.max_experiences_per_task,
                    "max_concurrent_extractions": args.max_concurrent_extractions,
                },
                "patch": {"max_edits": args.max_patch_edits, "max_attempts": args.patch_attempts},
                "replay": {
                    "embedding_model_id": args.embedding_model,
                    "similarity_threshold": args.replay_similarity_threshold,
                    "min_cluster_edits": args.min_cluster_edits,
                    "capacity": args.replay_capacity,
                    "max_uses": args.replay_max_uses,
                },
                "ablation": {
                    "max_source_cases_per_candidate": args.ablation_source_cases,
                    "positive_only": args.ablation_positive_only,
                    "commit_topk": args.ablation_commit_topk,
                    "seed": args.seed,
                },
                "validation": {"every_batches": args.validate_every},
            },
            "training": {
                "seed": args.seed,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "success_reward": 1.0,
            },
            "rollout": {
                "max_concurrent_rollouts": args.max_concurrent_rollouts,
                "timeout_seconds": args.rollout_timeout,
                "retry": {"max_attempts": args.rollout_retries},
                "fail_fast": args.fail_fast,
                "workspace_root": args.output_dir / "workspace",
                "train": {"name": "fixed_group", "params": {"group_size": args.train_rollouts}},
                "ablation": {
                    "name": "paired_ablation",
                    "params": {"samples_per_case": args.ablation_rollouts},
                },
                "validation": {"name": "fixed_group", "params": {"group_size": args.validation_rollouts}},
                "test": {"name": "fixed_group", "params": {"group_size": args.test_rollouts}},
            },
            "dataset": {
                "env_ref": args.benchmark,
                "agent_ref": "react",
                "env_options": env_options,
                "agent_options": {},
            },
        }
    )


def safe_arguments(args: argparse.Namespace) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


async def run(args: argparse.Namespace) -> None:
    started_at = monotonic()
    progress_log(f"loading {args.benchmark} dataset from {args.data_root}", started_at)
    dataset = build_dataset(args)
    train_tasks = limited(dataset.train_tasks(), args.train_limit)
    validation_tasks = limited(dataset.validation_tasks(), args.validation_limit)
    test_tasks = limited(dataset.test_tasks(), args.test_limit)
    progress_log(
        f"dataset loaded: train={len(train_tasks)}, validation={len(validation_tasks)}, test={len(test_tasks)}",
        started_at,
    )
    config = build_run_config(args)
    resume_state = None
    if args.resume:
        resume_state, base_skill = load_resume_artifacts(args.output_dir)
        validate_resume_arguments(args.output_dir, args)
        validate_resume(
            resume_state,
            run_id=args.run_id,
            algorithm_version=config.algorithm.version,
            expected_input_fingerprint=input_fingerprint(
                base_skill,
                train_tasks,
                validation_tasks,
                test_tasks,
            ),
            expected_config_fingerprint=config_fingerprint(config),
            base_skill_hash=base_skill.content_hash,
        )
        preserved_rollout_ids = {outcome.spec.rollout_id for outcome in resume_state.rollout_outcomes}
        removed = clear_incomplete_rollout_workspaces(args.output_dir / "workspace", preserved_rollout_ids)
        progress_log(
            f"resume checkpoint loaded: completed_batches={len(resume_state.batches)}, "
            f"final_test_completed={resume_state.final_test_completed}, cleared_partial_rollouts={removed}",
            started_at,
        )
    else:
        base_skill = build_skill(args)
        args.output_dir.mkdir(parents=True, exist_ok=False)
        write_json_atomic(args.output_dir / "arguments.json", safe_arguments(args))
        write_json_atomic(args.output_dir / "run_config.json", config.model_dump(mode="json"))
        write_json_atomic(args.output_dir / "base_skill.json", base_skill.model_dump(mode="json"))
        progress_log(f"output initialized at {args.output_dir}", started_at)

    progress_log("initializing database and model clients", started_at)
    database = await bootstrap_skill_database(args.output_dir / "state.db")
    call_sink = DatabaseLLMCallSink(database)
    target_client = build_client(args.target_model, args, call_sink=call_sink)
    optimizer_client = ChatModelWithDefaults(
        build_client(args.optimizer_model, args, call_sink=call_sink, temperature=0.0),
        reasoning_effort=args.reasoning_effort,
        max_completion_tokens=args.max_completion_tokens,
    )
    embedding_client = build_embedding_client(args, call_sink=call_sink)
    agent = ReactAgent(
        {
            "model": args.target_model,
            "max_turns": args.max_turns,
            "reasoning_effort": args.reasoning_effort,
            "model_kwargs": {"max_completion_tokens": args.max_completion_tokens},
        },
        llm=target_client,
    )
    progress_log(
        f"models ready: target={args.target_model}, optimizer={args.optimizer_model}, embedding={args.embedding_model}",
        started_at,
    )
    checkpoint_writer = CheckpointWriter(output_dir=args.output_dir)
    algorithm_logger = AlgorithmLogger(
        algorithm_name=SkillGrpoWithReplayBuffer.algorithm_name,
        algorithm_version=config.algorithm.version,
        database=database,
    )

    algorithm = SkillGrpoWithReplayBuffer(
        chat_model=optimizer_client,
        embedding_model=embedding_client,
        agent_resolver=MappingAgentResolver({"react": agent}),
        env_factory=RegistryEnvFactory(),
        on_event=checkpoint_writer.handle,
        logger=algorithm_logger,
    )
    try:
        result = await algorithm.evolve(
            SkillGrpoEvolveInput(
                run_id=args.run_id,
                base_skill=base_skill,
                train_tasks=train_tasks,
                validation_tasks=validation_tasks,
                test_tasks=test_tasks,
                config=config,
                resume_state=resume_state,
            )
        )
    finally:
        await database.close()
    (args.output_dir / "result.json").write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "final_skill.md").write_text(result.final_skill.content, encoding="utf-8")
    test_outcomes = [outcome for outcome in result.rollouts if outcome.spec.phase is RolloutPhase.TEST]
    test_output_dir = args.output_dir / "test"
    test_output_dir.mkdir(parents=True, exist_ok=True)
    test_summary = persist_test_artifacts(
        output_dir=test_output_dir,
        outcomes=test_outcomes,
        skill=result.final_skill,
        rollouts_per_task=args.test_rollouts,
        create_output_dir=False,
    )
    summary = {
        "run_id": result.run_id,
        "changed": result.changed,
        "metrics": result.metrics.model_dump(mode="json"),
        "final_skill_hash": result.final_skill.content_hash,
        "test": test_summary,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    progress_log("result.json, final_skill.md and summary.json saved", started_at)


def main(argv: list[str] | None = None) -> None:
    asyncio.run(run(parse_args(argv)))


if __name__ == "__main__":
    main()
