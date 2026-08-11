"""Build a trajectory memory bank and evaluate guarded top-k retrieval."""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime
from statistics import mean

from ....typing import Skill, Task, compute_skill_content_hash
from ..base import trajectories_from_rollouts
from ..skill_grpo_with_replay_buffer.contracts import RolloutOutcome
from ..skill_grpo_with_replay_buffer.models import ChatModel, EmbeddingModel
from ..skill_grpo_with_replay_buffer.rollout.scheduler import AgentResolver, EnvFactory, RolloutScheduler
from ..skill_grpo_with_replay_buffer.rollout.strategy import FixedGroupPlan, RolloutStrategyRegistry
from .contracts import (
    PairedEvaluationMetrics,
    TaskRetrievalRecord,
    TrajectoryMemoryEvolveInput,
    TrajectoryMemoryEvolveResult,
)
from .memory import (
    TrajectoryMemoryBankBuilder,
    select_trajectory_snapshots,
    snapshot_from_outcome,
)
from .prompts import render_retrieved_memories


class TrajectoryMemoryEvolve:
    """Complete non-parametric evolution through summarized trajectory retrieval."""

    algorithm_name = "trajectory_memory"

    def __init__(
        self,
        *,
        chat_model: ChatModel,
        embedding_model: EmbeddingModel,
        agent_resolver: AgentResolver,
        env_factory: EnvFactory,
    ) -> None:
        self._chat_model = chat_model
        self._embedding_model = embedding_model
        self._agent_resolver = agent_resolver
        self._env_factory = env_factory
        self._strategies = RolloutStrategyRegistry.with_builtins()

    async def evolve(self, request: TrajectoryMemoryEvolveInput) -> TrajectoryMemoryEvolveResult:
        config = request.config
        algorithm_config = config.algorithm
        scheduler = RolloutScheduler(
            agent_resolver=self._agent_resolver,
            env_factory=self._env_factory,
            config=config.rollout,
        )

        train_rollouts: list[RolloutOutcome] = []
        snapshots = list(request.precollected_train_trajectories)
        if not snapshots:
            train_specs = self._plan(
                request=request,
                scope="trajectory-memory-train",
                phase="train",
                tasks=request.train_tasks,
                skills=[request.base_skill],
                group_size=algorithm_config.train_rollouts_per_task,
                sequence_start=0,
            )
            train_rollouts = await scheduler.run(train_specs)
            snapshots = [
                snapshot
                for outcome in train_rollouts
                if (snapshot := snapshot_from_outcome(outcome)) is not None
            ]
        selected = select_trajectory_snapshots(
            snapshots,
            success_reward=algorithm_config.success_reward,
            max_examples_per_task=algorithm_config.max_examples_per_task,
        )
        if not selected:
            raise ValueError("trajectory memory requires at least one completed training trajectory")

        builder = TrajectoryMemoryBankBuilder(
            chat_model=self._chat_model,
            embedding_model=self._embedding_model,
            max_trajectory_chars=algorithm_config.max_trajectory_chars,
            max_summary_chars=algorithm_config.max_summary_chars,
            max_concurrent_summaries=algorithm_config.max_concurrent_summaries,
        )
        memory_bank = await builder.build(selected)
        retrievals = await builder.retrieve(
            request.test_tasks,
            memory_bank,
            top_k=algorithm_config.top_k,
        )

        baseline_rollouts: list[RolloutOutcome] = []
        if algorithm_config.run_baseline:
            baseline_specs = self._plan(
                request=request,
                scope="trajectory-memory-baseline",
                phase="test",
                tasks=request.test_tasks,
                skills=[request.base_skill],
                group_size=algorithm_config.test_rollouts_per_task,
                sequence_start=0,
            )
            baseline_rollouts = await scheduler.run(baseline_specs)

        memory_specs = []
        retrieval_by_task = {record.task_id: record for record in retrievals}
        sequence = 0
        for task in request.test_tasks:
            record = retrieval_by_task[task.task_id]
            memory_skill = _memory_skill(request.run_id, task, record)
            specs = self._plan(
                request=request,
                scope="trajectory-memory-assisted",
                phase="test",
                tasks=[task],
                skills=[request.base_skill, memory_skill],
                group_size=algorithm_config.test_rollouts_per_task,
                sequence_start=sequence,
            )
            memory_specs.extend(specs)
            sequence += len(specs)
        memory_rollouts = await scheduler.run(memory_specs)
        metrics = paired_metrics(baseline_rollouts, memory_rollouts)
        return TrajectoryMemoryEvolveResult(
            run_id=request.run_id,
            final_skill=request.base_skill,
            changed=False,
            trajectories=trajectories_from_rollouts(
                [*train_rollouts, *baseline_rollouts, *memory_rollouts]
            ),
            memory_bank=memory_bank,
            retrievals=retrievals,
            train_rollouts=train_rollouts,
            baseline_rollouts=baseline_rollouts,
            memory_rollouts=memory_rollouts,
            metrics=metrics,
        )

    def _plan(
        self,
        *,
        request: TrajectoryMemoryEvolveInput,
        scope: str,
        phase: str,
        tasks: list[Task],
        skills: list[Skill],
        group_size: int,
        sequence_start: int,
    ):
        dataset = request.config.dataset
        return self._strategies.get("fixed_group").plan(
            FixedGroupPlan(
                run_id=request.run_id,
                scope=scope,
                phase=phase,
                tasks=tasks,
                skills=skills,
                sequence_start=sequence_start,
                group_size=group_size,
                agent_ref=dataset.agent_ref,
                env_ref=dataset.env_ref,
                seed=0,
                agent_options=dataset.agent_options,
                env_options=dataset.env_options,
            )
        )


def paired_metrics(
    baseline_rollouts: list[RolloutOutcome],
    memory_rollouts: list[RolloutOutcome],
) -> PairedEvaluationMetrics:
    baseline = _scores_by_task(baseline_rollouts)
    memory = _scores_by_task(memory_rollouts)
    memory_score = mean(memory.values()) if memory else None
    if not baseline:
        return PairedEvaluationMetrics(task_count=len(memory), memory_score=memory_score)
    common = sorted(baseline.keys() & memory.keys())
    improved = sum(memory[task_id] > baseline[task_id] for task_id in common)
    regressed = sum(memory[task_id] < baseline[task_id] for task_id in common)
    baseline_score = mean(baseline[task_id] for task_id in common) if common else None
    aligned_memory_score = mean(memory[task_id] for task_id in common) if common else None
    return PairedEvaluationMetrics(
        task_count=len(common),
        baseline_score=baseline_score,
        memory_score=aligned_memory_score,
        delta=(aligned_memory_score - baseline_score)
        if aligned_memory_score is not None and baseline_score is not None
        else None,
        improved=improved,
        regressed=regressed,
        unchanged=len(common) - improved - regressed,
    )


def _scores_by_task(outcomes: list[RolloutOutcome]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for outcome in outcomes:
        trajectory = outcome.trajectory
        if trajectory is not None and trajectory.reward.score is not None:
            grouped[outcome.spec.task.task_id].append(float(trajectory.reward.score))
    return {task_id: mean(scores) for task_id, scores in grouped.items()}


def _memory_skill(run_id: str, task: Task, retrieval: TaskRetrievalRecord) -> Skill:
    content = render_retrieved_memories(retrieval.memories)
    blob = {"SKILL.md": content}
    now = datetime.now(UTC)
    return Skill(
        skill_id="trajectory-memory",
        version_id=f"{run_id}:trajectory-memory:{task.task_id}",
        version_label="0.1.0",
        content_hash=compute_skill_content_hash(blob),
        name="trajectory_memory",
        description="Guarded top-k memories retrieved from successful training trajectories.",
        blob=blob,
        created_at=now,
        metadata={
            "algorithm": TrajectoryMemoryEvolve.algorithm_name,
            "task_id": task.task_id,
            "memory_ids": [item.item.memory_id for item in retrieval.memories],
        },
    )


__all__ = ["TrajectoryMemoryEvolve", "paired_metrics"]
