#!/usr/bin/env python3
"""Script-side fixed-Skill ALFWorld experience collection and embedding retrieval."""

import argparse
import asyncio
import hashlib
import json
import math
import os
import re
import sqlite3
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from time import monotonic
from typing import Any

import yaml
from mindmemos_skill.agents.react import ReactAgent
from mindmemos_skill.algos.evolve.skill_grpo_with_replay_buffer import MappingAgentResolver, RegistryEnvFactory
from mindmemos_skill.algos.evolve.skill_grpo_with_replay_buffer.contracts import (
    RolloutOutcome,
    RolloutPhase,
    RolloutSpec,
)
from mindmemos_skill.algos.evolve.skill_grpo_with_replay_buffer.models import embedding_vectors
from mindmemos_skill.algos.evolve.skill_grpo_with_replay_buffer.rollout import (
    FixedGroupPlan,
    RolloutScheduler,
    RolloutStrategyRegistry,
)
from mindmemos_skill.algos.evolve.skill_grpo_without_replay_buffer.config import (
    SkillGrpoWithoutReplayBufferRunConfig,
)
from mindmemos_skill.algos.evolve.skill_grpo_without_replay_buffer.contracts import (
    ExperienceSource,
    ReplayFreeExtractedExperience,
)
from mindmemos_skill.algos.evolve.skill_grpo_without_replay_buffer.experience import ExperienceExtractor
from mindmemos_skill.algos.evolve.skill_grpo_without_replay_buffer.reflection import (
    REFLECTION_PROMPT_VERSION,
    ReflectionGenerator,
    previous_answer,
    task_with_reflection,
)
from mindmemos_skill.algos.evolve.trajectory_memory.memory import task_retrieval_key
from mindmemos_skill.datasets import ALFWorldPathSplitDataset
from mindmemos_skill.llm import (
    DatabaseLLMCallSink,
    EmbedClient,
    LLMCallSink,
    LLMClient,
    get_router,
    llm_run_context,
)
from mindmemos_skill.persistence import bootstrap_skill_database
from mindmemos_skill.typing import Skill, Task, compute_skill_content_hash
from pydantic import BaseModel, ConfigDict, Field


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class AtomicExperience(_StrictModel):
    """One existing ``content.experiences[]`` item with its extraction provenance."""

    memory_id: str
    source_set_index: int = Field(ge=0)
    item_index: int = Field(ge=0)
    source: ExperienceSource
    topic: str
    lesson: str
    reason: str
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    task_ids: list[str]
    retrieval_document: str
    embedding: list[float] = Field(default_factory=list)


class RetrievedExperience(_StrictModel):
    rank: int = Field(ge=1)
    similarity: float = Field(ge=-1.0, le=1.0)
    memory: AtomicExperience


class ExperienceRetrievalRecord(_StrictModel):
    task_id: str
    query: str
    embedding_model: str
    experiences: list[RetrievedExperience]


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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)

    parser.add_argument("--target-model", default="openai/gpt-5.4-mini")
    parser.add_argument("--optimizer-model", default="openai/gpt-5.4-mini")
    parser.add_argument("--embedding-model", default="openai/qwen3-embedding-4b")
    parser.add_argument(
        "--embedding-config",
        type=Path,
        help="MindMemOS YAML containing embed_model_router endpoints; keys stay out of CLI arguments and artifacts.",
    )
    parser.add_argument(
        "--resume-experience-db",
        type=Path,
        help="Recover completed replay-free experience extraction calls from a previous state.db and skip training.",
    )
    parser.add_argument("--api-base", default=os.getenv("OPENAI_ENDPOINT") or os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--request-timeout", type=float, default=1800.0)
    parser.add_argument("--model-retries", type=int, default=3)
    parser.add_argument("--max-completion-tokens", type=int, default=16384)

    parser.add_argument("--train-rollouts", type=int, default=4)
    parser.add_argument("--test-rollouts", type=int, default=1)
    parser.add_argument("--mini-batch-size", type=int, default=8)
    parser.add_argument("--max-experiences-per-task", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--run-baseline", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reflection", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-concurrent-reflections", type=int, default=8)
    parser.add_argument("--max-concurrent-extractions", type=int, default=16)
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--max-concurrent-rollouts", type=int, default=32)
    parser.add_argument("--rollout-retries", type=int, default=3)
    parser.add_argument("--rollout-timeout", type=float)
    parser.add_argument("--max-turns", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--test-limit", type=int)
    return parser.parse_args(argv)


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
    if args.embedding_config is not None:
        loaded = yaml.safe_load(args.embedding_config.read_text(encoding="utf-8"))
        router_config = loaded.get("embed_model_router") if isinstance(loaded, dict) else None
        configured = router_config.get("endpoints") if isinstance(router_config, dict) else None
        if not isinstance(configured, list):
            raise ValueError(f"{args.embedding_config} has no embed_model_router.endpoints list")
        endpoints = [endpoint for endpoint in configured if endpoint.get("model") == args.embedding_model]
        if not endpoints:
            raise ValueError(f"{args.embedding_config} has no endpoint for {args.embedding_model}")
        routing_strategy = router_config.get("routing_strategy", "simple-shuffle")
    else:
        endpoints = [
            {
                "model": args.embedding_model,
                "api_key": os.getenv("OPENAI_API_KEY"),
                "api_base": args.api_base,
                "timeout": args.request_timeout,
                "num_retries": args.model_retries,
            }
        ]
        routing_strategy = "simple-shuffle"
    router, _ = get_router(
        {"routing_strategy": routing_strategy, "endpoints": endpoints},
        args.embedding_model,
        num_retries=args.model_retries,
    )
    return EmbedClient(router, default_model=args.embedding_model, call_sink=call_sink)


def recover_experience_sets(database_path: Path) -> list[ReplayFreeExtractedExperience]:
    """Recover extractor outputs recorded before a later embedding-stage failure."""

    with sqlite3.connect(database_path) as connection:
        rows = connection.execute(
            """
            SELECT task, request, response
            FROM llm_calls
            WHERE status = 'succeeded' AND task LIKE 'skill_grpo.experience.%'
            ORDER BY started_at, task
            """
        ).fetchall()
    recovered: list[ReplayFreeExtractedExperience] = []
    for call_task, raw_request, raw_response in rows:
        request = json.loads(raw_request)
        response = json.loads(raw_response)
        messages = request.get("messages", [])
        prompt = messages[-1].get("content", "") if messages else ""
        content = response["choices"][0]["message"]["content"]
        task_ids = _ordered_unique(re.findall(r"### Task ID\s*\n\s*([^\n]+)", prompt))
        rollout_count = prompt.count("## Task trajectory ")
        if call_task.startswith("skill_grpo.experience.contrast."):
            source = ExperienceSource.CONTRAST
            task_id = call_task.removeprefix("skill_grpo.experience.contrast.")
            task_ids = task_ids or [task_id]
        else:
            match = re.fullmatch(r"skill_grpo\.experience\.(failure|success)\.(\d+)", call_task)
            if match is None:
                raise ValueError(f"unsupported recorded extraction task: {call_task}")
            source = ExperienceSource(match.group(1))
            group_index = int(match.group(2))
            task_id = f"{source.value}-mini-batch-{group_index + 1}"
        if not task_ids:
            raise ValueError(f"could not recover source task IDs from {call_task}")
        recovered.append(
            ReplayFreeExtractedExperience(
                task_id=task_id,
                task_ids=task_ids,
                source=source,
                content=content,
                rollout_count=rollout_count,
            )
        )
    if not recovered:
        raise ValueError(f"no completed experience extraction calls found in {database_path}")
    source_order = {ExperienceSource.CONTRAST: 0, ExperienceSource.FAILURE: 1, ExperienceSource.SUCCESS: 2}
    recovered.sort(key=lambda item: (source_order[item.source], item.task_id))
    return recovered


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


def build_run_config(args: argparse.Namespace) -> SkillGrpoWithoutReplayBufferRunConfig:
    """Reuse existing rollout/extraction contracts while disabling patch validation."""

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
                "validation": {"enabled": False},
            },
            "training": {
                "seed": args.seed,
                "epochs": 1,
                "batch_size": 40,
                "mini_batch_size": args.mini_batch_size,
                "success_reward": 1.0,
            },
            "rollout": {
                "max_concurrent_rollouts": args.max_concurrent_rollouts,
                "timeout_seconds": args.rollout_timeout,
                "retry": {"max_attempts": args.rollout_retries},
                "fail_fast": False,
                "workspace_root": args.output_dir / "workspace",
                "train": {"name": "fixed_group", "params": {"group_size": args.train_rollouts}},
                "validation": {"name": "fixed_group", "params": {"group_size": 1}},
            },
            "dataset": {
                "env_ref": "alfworld",
                "agent_ref": "react",
                "env_options": {
                    "max_turns": args.max_turns,
                    "seed": args.seed,
                },
                "agent_options": {},
            },
        }
    )


def atomic_experiences(
    sets: list[ReplayFreeExtractedExperience],
    *,
    tasks: list[Task],
) -> list[AtomicExperience]:
    """Expand the existing JSON payload without changing its extraction schema."""

    task_by_id = {task.task_id: task for task in tasks}
    bank: list[AtomicExperience] = []
    for set_index, experience_set in enumerate(sets):
        try:
            payload = json.loads(experience_set.content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"experience set {set_index} did not contain valid JSON") from exc
        items = payload.get("experiences") if isinstance(payload, dict) else None
        if not isinstance(items, list):
            raise ValueError(f"experience set {set_index} is missing experiences[]")
        for item_index, item in enumerate(items):
            if not isinstance(item, dict):
                raise ValueError(f"experience set {set_index} item {item_index} is not an object")
            topic = _required_text(item, "topic", set_index=set_index, item_index=item_index)
            lesson = _required_text(item, "lesson", set_index=set_index, item_index=item_index)
            reason = _required_text(item, "reason", set_index=set_index, item_index=item_index)
            evidence = item.get("evidence")
            if not isinstance(evidence, list):
                evidence = []
            evidence = [value for value in evidence if isinstance(value, dict)]
            evidence_task_ids = [
                value.get("task_id")
                for value in evidence
                if isinstance(value.get("task_id"), str) and value.get("task_id") in task_by_id
            ]
            source_task_ids = _ordered_unique(evidence_task_ids or experience_set.task_ids)
            source_patterns = [
                task_retrieval_key(task_by_id[task_id]) for task_id in source_task_ids if task_id in task_by_id
            ]
            retrieval_document = render_retrieval_document(
                source=experience_set.source,
                topic=topic,
                lesson=lesson,
                reason=reason,
                source_patterns=source_patterns,
            )
            digest_payload = json.dumps(
                {
                    "source": experience_set.source.value,
                    "set": set_index,
                    "item": item_index,
                    "topic": topic,
                    "lesson": lesson,
                    "reason": reason,
                    "task_ids": source_task_ids,
                },
                sort_keys=True,
                ensure_ascii=False,
            )
            digest = hashlib.sha256(digest_payload.encode()).hexdigest()[:20]
            bank.append(
                AtomicExperience(
                    memory_id=f"experience-memory-{digest}",
                    source_set_index=set_index,
                    item_index=item_index,
                    source=experience_set.source,
                    topic=topic,
                    lesson=lesson,
                    reason=reason,
                    evidence=evidence,
                    task_ids=source_task_ids,
                    retrieval_document=retrieval_document,
                )
            )
    return bank


def render_retrieval_document(
    *,
    source: ExperienceSource,
    topic: str,
    lesson: str,
    reason: str,
    source_patterns: list[str],
) -> str:
    patterns = "\n".join(f"- {value}" for value in source_patterns) or "- unavailable"
    return (
        f"Experience source: {source.value}\n"
        f"Topic: {topic}\n"
        f"Reusable lesson: {lesson}\n"
        f"Evidence-grounded rationale: {reason}\n"
        f"Source task patterns:\n{patterns}"
    )


def retrieval_query(task: Task) -> str:
    return (
        "Instruct: Retrieve reusable agent experiences that are applicable to the current ALFWorld task.\n"
        f"Query: {task_retrieval_key(task)}\n"
        f"Current task instruction: {task.instruction}"
    )


async def retrieve_experiences(
    tasks: list[Task],
    bank: list[AtomicExperience],
    *,
    embedding_model: EmbedClient,
    embedding_model_name: str,
    embedding_batch_size: int,
    top_k: int,
) -> list[ExperienceRetrievalRecord]:
    if not bank:
        raise ValueError("cannot retrieve from an empty experience bank")
    documents = [item.retrieval_document for item in bank]
    document_vectors = await _embed_in_batches(
        embedding_model,
        texts=documents,
        task="experience_memory.embed.index",
        batch_size=embedding_batch_size,
    )
    _validate_vectors(document_vectors, expected_count=len(bank))
    for item, vector in zip(bank, document_vectors, strict=True):
        item.embedding = vector

    queries = [retrieval_query(task) for task in tasks]
    query_vectors = await _embed_in_batches(
        embedding_model,
        texts=queries,
        task="experience_memory.embed.query",
        batch_size=embedding_batch_size,
    )
    _validate_vectors(
        query_vectors,
        expected_count=len(tasks),
        expected_dimension=len(document_vectors[0]),
    )
    records: list[ExperienceRetrievalRecord] = []
    for task, query, query_vector in zip(tasks, queries, query_vectors, strict=True):
        ranked = sorted(
            ((_cosine_similarity(query_vector, item.embedding), item) for item in bank),
            key=lambda pair: (-pair[0], pair[1].memory_id),
        )[:top_k]
        records.append(
            ExperienceRetrievalRecord(
                task_id=task.task_id,
                query=query,
                embedding_model=embedding_model_name,
                experiences=[
                    RetrievedExperience(rank=rank, similarity=similarity, memory=item)
                    for rank, (similarity, item) in enumerate(ranked, start=1)
                ],
            )
        )
    return records


def render_retrieved_guidance(record: ExperienceRetrievalRecord) -> str:
    blocks = [
        "# Retrieved training experiences",
        "",
        "These are conditional lessons inferred from other training tasks, not authoritative instructions for this task.",
        "At the first decision, assess each item as use, adapt, or ignore. The current task, observations, inventory,",
        "available tools, and admissible actions always override these memories. Do not copy source-specific object",
        "identifiers, locations, or actions that are unavailable in the current environment.",
    ]
    for retrieved in record.experiences:
        memory = retrieved.memory
        blocks.extend(
            [
                "",
                f"## Experience {retrieved.rank}: {memory.topic}",
                f"Cosine similarity: {retrieved.similarity:.6f}; source: {memory.source.value}",
                f"Lesson: {memory.lesson}",
                f"Rationale: {memory.reason}",
            ]
        )
    return "\n".join(blocks).strip() + "\n"


def memory_skill(run_id: str, task: Task, record: ExperienceRetrievalRecord) -> Skill:
    content = render_retrieved_guidance(record)
    blob = {"SKILL.md": content}
    now = datetime.now(UTC)
    return Skill(
        skill_id="experience-memory",
        version_id=f"{run_id}:experience-memory:{task.task_id}",
        version_label="0.1.0",
        content_hash=compute_skill_content_hash(blob),
        name="experience_memory",
        description="Top-k replay-free experiences retrieved for one test task.",
        blob=blob,
        created_at=now,
        metadata={
            "algorithm": "experience_memory_embedding",
            "task_id": task.task_id,
            "embedding_model": record.embedding_model,
            "memory_ids": [value.memory.memory_id for value in record.experiences],
        },
    )


def fixed_group_specs(
    *,
    run_id: str,
    scope: str,
    phase: RolloutPhase,
    tasks: list[Task],
    skills: list[Skill],
    group_size: int,
    sequence_start: int,
    seed: int,
    max_turns: int,
) -> list[RolloutSpec]:
    strategies = RolloutStrategyRegistry.with_builtins()
    return strategies.get("fixed_group").plan(
        FixedGroupPlan(
            run_id=run_id,
            scope=scope,
            phase=phase.value,
            tasks=tasks,
            skills=skills,
            sequence_start=sequence_start,
            group_size=group_size,
            agent_ref="react",
            env_ref="alfworld",
            seed=seed,
            agent_options={},
            env_options={"max_turns": max_turns, "seed": seed},
        )
    )


async def run_reflective_training(
    specs: list[RolloutSpec],
    *,
    scheduler: RolloutScheduler,
    reflector: ReflectionGenerator,
    reflection_enabled: bool,
    success_reward: float,
    max_previous_answer_chars: int,
) -> list[RolloutOutcome]:
    """Match the current replay-free sequential-per-task early-stop behavior."""

    specs_by_task: dict[str, list[RolloutSpec]] = defaultdict(list)
    for spec in specs:
        specs_by_task[spec.task.task_id].append(spec)

    async def run_task_chain(task_specs: list[RolloutSpec]) -> list[RolloutOutcome]:
        outcomes: list[RolloutOutcome] = []
        next_spec = task_specs[0]
        for index, planned_spec in enumerate(task_specs):
            if index > 0:
                next_spec = next_spec.model_copy(
                    update={
                        "sequence_no": planned_spec.sequence_no,
                        "rollout_id": planned_spec.rollout_id,
                        "sample_index": planned_spec.sample_index,
                        "seed": planned_spec.seed,
                    },
                    deep=True,
                )
            outcome = (await scheduler.run([next_spec]))[0]
            outcomes.append(outcome)
            if _is_success(outcome, success_reward=success_reward) or index + 1 >= len(task_specs):
                break
            next_planned = task_specs[index + 1]
            next_spec = next_planned
            if not reflection_enabled or outcome.trajectory is None:
                continue
            reflection = await reflector.reflect(outcome.trajectory, sample_index=outcome.spec.sample_index)
            if not reflection:
                continue
            answer = previous_answer(outcome.trajectory, max_chars=max_previous_answer_chars)
            next_spec = next_planned.model_copy(
                update={
                    "task": task_with_reflection(next_planned.task, answer=answer, reflection=reflection),
                    "metadata": {
                        **next_planned.metadata,
                        "reflection_context": {
                            "prompt_version": REFLECTION_PROMPT_VERSION,
                            "source_rollout_id": outcome.spec.rollout_id,
                            "previous_answer": answer,
                            "content": reflection,
                        },
                    },
                },
                deep=True,
            )
        return outcomes

    grouped = await asyncio.gather(*(run_task_chain(task_specs) for task_specs in specs_by_task.values()))
    outcomes = [outcome for task_outcomes in grouped for outcome in task_outcomes]
    outcomes.sort(key=lambda item: item.spec.sequence_no)
    return outcomes


def paired_metrics(baseline: list[RolloutOutcome], memory: list[RolloutOutcome]) -> dict[str, Any]:
    baseline_scores = _scores_by_task(baseline)
    memory_scores = _scores_by_task(memory)
    if not baseline_scores:
        return {
            "task_count": len(memory_scores),
            "baseline_score": None,
            "memory_score": mean(memory_scores.values()) if memory_scores else None,
            "delta": None,
            "improved": 0,
            "regressed": 0,
            "unchanged": 0,
        }
    common = sorted(baseline_scores.keys() & memory_scores.keys())
    baseline_score = mean(baseline_scores[task_id] for task_id in common)
    memory_score = mean(memory_scores[task_id] for task_id in common)
    improved = sum(memory_scores[task_id] > baseline_scores[task_id] for task_id in common)
    regressed = sum(memory_scores[task_id] < baseline_scores[task_id] for task_id in common)
    return {
        "task_count": len(common),
        "baseline_score": baseline_score,
        "memory_score": memory_score,
        "delta": memory_score - baseline_score,
        "improved": improved,
        "regressed": regressed,
        "unchanged": len(common) - improved - regressed,
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is required")
    if not args.api_base:
        raise ValueError("--api-base, OPENAI_ENDPOINT, or OPENAI_BASE_URL is required")
    if args.top_k < 1:
        raise ValueError("--top-k must be positive")
    if args.resume_experience_db is not None and not args.resume_experience_db.is_file():
        raise FileNotFoundError(f"resume database does not exist: {args.resume_experience_db}")

    started_at = monotonic()
    os.environ["ALFWORLD_DATA"] = str(args.data_root.expanduser().resolve())
    dataset = ALFWorldPathSplitDataset(split_dir=args.split_dir, alfworld_data=args.data_root)
    train_tasks = _limited(dataset.train_tasks(), args.train_limit)
    test_tasks = _limited(dataset.test_tasks(), args.test_limit)
    base_skill = build_skill(args)
    config = build_run_config(args)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / "arguments.json", _safe_arguments(args))
    write_json(args.output_dir / "run_config.json", config.model_dump(mode="json"))
    write_json(args.output_dir / "base_skill.json", base_skill.model_dump(mode="json"))

    database = await bootstrap_skill_database(args.output_dir / "state.db")
    sink = DatabaseLLMCallSink(database)
    target_client = build_client(args.target_model, args, call_sink=sink)
    optimizer = ChatModelWithDefaults(
        build_client(args.optimizer_model, args, call_sink=sink, temperature=0.0),
        reasoning_effort=args.reasoning_effort,
        max_completion_tokens=args.max_completion_tokens,
    )
    embedding_model = build_embedding_client(args, call_sink=sink)
    agent = ReactAgent(
        {
            "model": args.target_model,
            "max_turns": 1,
            "reasoning_effort": args.reasoning_effort,
            "model_kwargs": {"max_completion_tokens": args.max_completion_tokens},
        },
        llm=target_client,
    )
    scheduler = RolloutScheduler(
        agent_resolver=MappingAgentResolver({"react": agent}),
        env_factory=RegistryEnvFactory(),
        config=config.rollout,
    )
    reflection_config = config.algorithm.reflection
    reflector = ReflectionGenerator(
        optimizer,
        max_trajectory_chars=reflection_config.max_trajectory_chars,
        max_reflection_chars=reflection_config.max_reflection_chars,
        max_concurrency=reflection_config.max_concurrent_reflections,
    )
    extractor = ExperienceExtractor(
        optimizer,
        max_experiences=config.algorithm.experience.max_experiences_per_task,
        max_concurrency=config.algorithm.experience.max_concurrent_extractions,
    )

    try:
        with llm_run_context(args.run_id):
            train_specs = fixed_group_specs(
                run_id=args.run_id,
                scope="experience-memory-train",
                phase=RolloutPhase.TRAIN,
                tasks=train_tasks,
                skills=[base_skill],
                group_size=args.train_rollouts,
                sequence_start=0,
                seed=args.seed,
                max_turns=args.max_turns,
            )
            if args.resume_experience_db is not None:
                train_outcomes = []
                experience_sets = recover_experience_sets(args.resume_experience_db)
            else:
                train_outcomes = await run_reflective_training(
                    train_specs,
                    scheduler=scheduler,
                    reflector=reflector,
                    reflection_enabled=reflection_config.enabled,
                    success_reward=config.training.success_reward,
                    max_previous_answer_chars=reflection_config.max_previous_answer_chars,
                )
                experience_sets = await extractor.extract(
                    train_outcomes,
                    base_skill,
                    mini_batch_size=config.training.mini_batch_size,
                    success_reward=config.training.success_reward,
                )
            bank = atomic_experiences(experience_sets, tasks=train_tasks)
            if not bank:
                raise RuntimeError("training produced no atomic experiences")
            write_json(
                args.output_dir / "experience_sets.json",
                [experience.model_dump(mode="json") for experience in experience_sets],
            )
            write_json(
                args.output_dir / "experience_bank.pre_embedding.json", [item.model_dump(mode="json") for item in bank]
            )
            retrievals = await retrieve_experiences(
                test_tasks,
                bank,
                embedding_model=embedding_model,
                embedding_model_name=args.embedding_model,
                embedding_batch_size=args.embedding_batch_size,
                top_k=args.top_k,
            )

            baseline_outcomes: list[RolloutOutcome] = []
            sequence_start = len(train_specs)
            if args.run_baseline:
                baseline_specs = fixed_group_specs(
                    run_id=args.run_id,
                    scope="experience-memory-baseline",
                    phase=RolloutPhase.TEST,
                    tasks=test_tasks,
                    skills=[base_skill],
                    group_size=args.test_rollouts,
                    sequence_start=sequence_start,
                    seed=args.seed,
                    max_turns=args.max_turns,
                )
                baseline_outcomes = await scheduler.run(baseline_specs)
                sequence_start += len(baseline_specs)

            retrieval_by_task = {record.task_id: record for record in retrievals}
            memory_specs: list[RolloutSpec] = []
            for task in test_tasks:
                record = retrieval_by_task[task.task_id]
                injected = memory_skill(args.run_id, task, record)
                specs = fixed_group_specs(
                    run_id=args.run_id,
                    scope="experience-memory-assisted",
                    phase=RolloutPhase.TEST,
                    tasks=[task],
                    skills=[base_skill, injected],
                    group_size=args.test_rollouts,
                    sequence_start=sequence_start,
                    seed=args.seed,
                    max_turns=args.max_turns,
                )
                memory_specs.extend(specs)
                sequence_start += len(specs)
            memory_outcomes = await scheduler.run(memory_specs)
    finally:
        await database.close()

    write_json(
        args.output_dir / "experience_sets.json",
        [experience.model_dump(mode="json") for experience in experience_sets],
    )
    write_json(args.output_dir / "experience_bank.json", [item.model_dump(mode="json") for item in bank])
    write_json(
        args.output_dir / "retrievals.json",
        [
            {
                "task_id": record.task_id,
                "query": record.query,
                "embedding_model": record.embedding_model,
                "experiences": [
                    {
                        "rank": value.rank,
                        "similarity": value.similarity,
                        "memory_id": value.memory.memory_id,
                        "source": value.memory.source.value,
                        "topic": value.memory.topic,
                        "lesson": value.memory.lesson,
                        "reason": value.memory.reason,
                        "task_ids": value.memory.task_ids,
                    }
                    for value in record.experiences
                ],
            }
            for record in retrievals
        ],
    )
    write_rollouts(args.output_dir / "train_results.jsonl", train_outcomes)
    write_rollouts(args.output_dir / "baseline_results.jsonl", baseline_outcomes)
    write_rollouts(args.output_dir / "memory_results.jsonl", memory_outcomes)
    metrics = paired_metrics(baseline_outcomes, memory_outcomes)
    summary = {
        "run_id": args.run_id,
        "skill_unchanged": True,
        "base_skill_hash": base_skill.content_hash,
        "retrieval_kind": "embedding_cosine",
        "embedding_model": args.embedding_model,
        "top_k": args.top_k,
        "resumed_from_experience_db": str(args.resume_experience_db) if args.resume_experience_db is not None else None,
        "train_task_count": len(train_tasks),
        "train_rollout_count": len(train_outcomes) if args.resume_experience_db is None else None,
        "train_successful_rollouts": (
            sum(_is_success(value, success_reward=1.0) for value in train_outcomes)
            if args.resume_experience_db is None
            else None
        ),
        "experience_set_count": len(experience_sets),
        "atomic_experience_count": len(bank),
        "test_task_count": len(test_tasks),
        "metrics": metrics,
        "elapsed_seconds": monotonic() - started_at,
    }
    write_json(args.output_dir / "summary.json", summary)
    return summary


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_rollouts(path: Path, outcomes: list[RolloutOutcome]) -> None:
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
            stream.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def _scores_by_task(outcomes: list[RolloutOutcome]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for outcome in outcomes:
        trajectory = outcome.trajectory
        if trajectory is not None and trajectory.reward.score is not None:
            grouped[outcome.spec.task.task_id].append(float(trajectory.reward.score))
    return {task_id: mean(scores) for task_id, scores in grouped.items()}


def _is_success(outcome: RolloutOutcome, *, success_reward: float) -> bool:
    trajectory = outcome.trajectory
    score = trajectory.reward.score if trajectory is not None else None
    return score is not None and score >= success_reward


def _required_text(item: dict[str, Any], key: str, *, set_index: int, item_index: int) -> str:
    value = item.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"experience set {set_index} item {item_index} has no non-empty {key}")
    return value.strip()


def _ordered_unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


async def _embed_in_batches(
    model: EmbedClient,
    *,
    texts: list[str],
    task: str,
    batch_size: int,
) -> list[list[float]]:
    if batch_size < 1:
        raise ValueError("embedding batch size must be positive")
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        vectors.extend(
            await embedding_vectors(
                model,
                task=f"{task}.{start // batch_size}",
                texts=texts[start : start + batch_size],
            )
        )
    return vectors


def _validate_vectors(
    vectors: list[list[float]],
    *,
    expected_count: int,
    expected_dimension: int | None = None,
) -> None:
    if len(vectors) != expected_count or not vectors:
        raise ValueError(f"embedding response count mismatch: expected {expected_count}, got {len(vectors)}")
    dimension = len(vectors[0])
    if dimension < 1 or any(len(vector) != dimension for vector in vectors):
        raise ValueError("embedding vectors must have one consistent non-zero dimension")
    if expected_dimension is not None and dimension != expected_dimension:
        raise ValueError(f"embedding dimension changed from {expected_dimension} to {dimension}")


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("cannot compare embeddings with different dimensions")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    value = sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)
    return max(-1.0, min(1.0, value))


def _limited(items: list[Any], limit: int | None) -> list[Any]:
    if limit is None:
        return items
    if limit < 0:
        raise ValueError("task limits must be non-negative")
    return items[:limit]


def _safe_arguments(args: argparse.Namespace) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def main(argv: list[str] | None = None) -> None:
    print(json.dumps(asyncio.run(run(parse_args(argv))), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
