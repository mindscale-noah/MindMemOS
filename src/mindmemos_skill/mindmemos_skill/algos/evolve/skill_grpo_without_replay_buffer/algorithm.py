"""Replay-free Skill GRPO: rollout, extract, merge, patch, and gate."""

from __future__ import annotations

import asyncio
import statistics
from collections import Counter
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from time import monotonic
from typing import Any

from ....errors import SkillConfigurationError
from ....llm import llm_run_context
from ....logging import AlgorithmLogger, LogLevel, format_log_value
from ....persistence.enums import SkillVersionOrigin, SkillVersionStatus
from ....registry import ComponentRequirements, ComponentType, register
from ....typing import EvolveInput, Skill, Task, compute_skill_content_hash
from ..base import EvolveAlgorithmContext, trajectories_from_rollouts
from ..skill_grpo_with_replay_buffer.batch_planner import TaskBatchPlanner
from ..skill_grpo_with_replay_buffer.config import RolloutStrategyConfig
from ..skill_grpo_with_replay_buffer.contracts import EvolutionEvent, RolloutOutcome, RolloutPhase, RolloutSpec
from ..skill_grpo_with_replay_buffer.fileedit import apply_best_effort
from ..skill_grpo_with_replay_buffer.models import ChatModel
from ..skill_grpo_with_replay_buffer.rollout import (
    AgentResolver,
    EnvFactory,
    FixedGroupPlan,
    MappingAgentResolver,
    RegistryEnvFactory,
    RolloutScheduler,
    RolloutStrategyRegistry,
)
from .config import SkillGrpoWithoutReplayBufferRunConfig
from .contracts import (
    BatchEvolutionRecord,
    EvolutionMetrics,
    SkillGrpoWithoutReplayBufferEvolveInput,
    SkillGrpoWithoutReplayBufferEvolveResult,
    ValidationDecision,
)
from .experience import ExperienceExtractor
from .patch import PatchProposer
from .reflection import REFLECTION_PROMPT_VERSION, ReflectionGenerator, previous_answer, task_with_reflection

EventCallback = Callable[[EvolutionEvent], Awaitable[None]]


@register(
    type=ComponentType.ALGO,
    name="skill_grpo_without_replay_buffer",
    config_model=SkillGrpoWithoutReplayBufferRunConfig,
    capabilities={"evolve"},
    requirements=ComponentRequirements(required_model_roles=frozenset({"chat"})),
)
class SkillGrpoWithoutReplayBuffer:
    """Evolve a Skill directly from each batch, without replay or ablation."""

    algorithm_name = "skill_grpo_without_replay_buffer"

    def __init__(
        self,
        *,
        config: SkillGrpoWithoutReplayBufferRunConfig | None = None,
        context: EvolveAlgorithmContext | None = None,
        chat_model: ChatModel | None = None,
        agent_resolver: AgentResolver | None = None,
        env_factory: EnvFactory | None = None,
        rollout_strategies: RolloutStrategyRegistry | None = None,
        on_event: EventCallback | None = None,
        logger: AlgorithmLogger | None = None,
    ) -> None:
        if context is not None:
            chat_model = context.models.get("chat")
            agent_resolver = MappingAgentResolver(context.agents)
            env_factory = RegistryEnvFactory()
        if chat_model is None or agent_resolver is None or env_factory is None:
            raise SkillConfigurationError(
                "skill_grpo_without_replay_buffer requires chat_model, agent_resolver, and env_factory"
            )
        self._config = config
        self._chat_model = chat_model
        self._agent_resolver = agent_resolver
        self._env_factory = env_factory
        self._strategies = rollout_strategies or RolloutStrategyRegistry.with_builtins()
        self._on_event = on_event
        self._logger = logger or AlgorithmLogger(algorithm_name=self.algorithm_name)

    async def evolve(
        self,
        request: EvolveInput,
    ) -> SkillGrpoWithoutReplayBufferEvolveResult:
        request = self._normalize_request(request)
        with llm_run_context(request.run_id):
            return await self._evolve(request)

    def _normalize_request(self, request: EvolveInput) -> SkillGrpoWithoutReplayBufferEvolveInput:
        if isinstance(request, SkillGrpoWithoutReplayBufferEvolveInput):
            return request
        if self._config is None:
            raise SkillConfigurationError("skill_grpo_without_replay_buffer has no configured run settings")
        return SkillGrpoWithoutReplayBufferEvolveInput(
            **request.model_dump(),
            config=self._config,
        )

    async def _evolve(
        self,
        request: SkillGrpoWithoutReplayBufferEvolveInput,
    ) -> SkillGrpoWithoutReplayBufferEvolveResult:
        self._validate_request(request)
        config = request.config
        scheduler = RolloutScheduler(
            agent_resolver=self._agent_resolver,
            env_factory=self._env_factory,
            config=config.rollout,
            on_outcome=self._rollout_event_callback(request.run_id),
        )
        extractor = ExperienceExtractor(
            self._chat_model,
            max_experiences=config.algorithm.experience.max_experiences_per_task,
            max_concurrency=config.algorithm.experience.max_concurrent_extractions,
        )
        proposer = PatchProposer(
            self._chat_model,
            max_edits=config.algorithm.patch.max_edits,
            max_attempts=config.algorithm.patch.max_attempts,
        )
        reflection_config = config.algorithm.reflection
        reflector = ReflectionGenerator(
            self._chat_model,
            max_trajectory_chars=reflection_config.max_trajectory_chars,
            max_reflection_chars=reflection_config.max_reflection_chars,
            max_concurrency=reflection_config.max_concurrent_reflections,
        )
        plans = TaskBatchPlanner().build(
            request.train_tasks,
            epochs=config.training.epochs,
            batch_size=config.training.batch_size,
            seed=config.training.seed,
        )
        current_skill = request.base_skill
        rollout_sequence = 0
        all_outcomes: list[RolloutOutcome] = []
        batches: list[BatchEvolutionRecord] = []
        validation_score: float | None = None

        await self._emit(
            request.run_id,
            "run_started",
            {
                "base_skill_hash": request.base_skill.content_hash,
                "total_batches": len(plans),
                "reflection_enabled": reflection_config.enabled,
                "validation_gate_enabled": config.algorithm.validation.enabled,
            },
        )
        if config.algorithm.validation.enabled:
            baseline, rollout_sequence = await self._run_fixed_phase(
                request=request,
                scheduler=scheduler,
                scope="validation_baseline",
                phase=RolloutPhase.VALIDATION,
                tasks=request.validation_tasks,
                skill=current_skill,
                strategy_config=config.rollout.validation,
                rollout_sequence=rollout_sequence,
                reflector=reflector,
            )
            all_outcomes.extend(baseline)
            validation_score = self._mean_score(baseline)

        for batch in plans:
            skill_before = current_skill
            train_outcomes, rollout_sequence = await self._run_fixed_phase(
                request=request,
                scheduler=scheduler,
                scope=f"batch_{batch.batch_index}",
                phase=RolloutPhase.TRAIN,
                tasks=list(batch.tasks),
                skill=current_skill,
                strategy_config=config.rollout.train,
                rollout_sequence=rollout_sequence,
                reflector=reflector,
            )
            all_outcomes.extend(train_outcomes)
            rollout_summary = self._batch_rollout_summary(
                train_outcomes,
                task_ids=[task.task_id for task in batch.tasks],
                rollouts_per_case=self._positive_int(
                    config.rollout.train.params.get("group_size", 1),
                    "group_size",
                ),
                success_reward=config.training.success_reward,
            )
            await self._emit(
                request.run_id,
                "batch_rollout_summary",
                {"batch_index": batch.batch_index, **rollout_summary},
            )
            await self._emit_stage(
                request.run_id,
                "experience_extraction",
                "started",
                batch.batch_index,
                trajectory_count=sum(outcome.trajectory is not None for outcome in train_outcomes),
                mini_batch_size=config.training.mini_batch_size,
            )
            stage_started_at = monotonic()
            experiences = (
                await extractor.extract(
                    train_outcomes,
                    current_skill,
                    mini_batch_size=config.training.mini_batch_size,
                    success_reward=config.training.success_reward,
                )
                if train_outcomes
                else []
            )
            await self._emit_stage(
                request.run_id,
                "experience_extraction",
                "completed",
                batch.batch_index,
                duration_seconds=monotonic() - stage_started_at,
                experience_count=len(experiences),
                experience_sources=dict(Counter(experience.source.value for experience in experiences)),
            )
            await self._emit_stage(
                request.run_id,
                "patch_proposal",
                "started",
                batch.batch_index,
                experience_count=len(experiences),
            )
            stage_started_at = monotonic()
            patch = await proposer.propose(current_skill, experiences) if experiences else None
            await self._emit_stage(
                request.run_id,
                "patch_proposal",
                "completed",
                batch.batch_index,
                duration_seconds=monotonic() - stage_started_at,
                proposed_edits=patch.proposed_edit_count if patch is not None else 0,
                valid_edits=len(patch.edit_support) if patch is not None else 0,
            )
            proposed_edits = [item.edit for item in patch.edit_support] if patch is not None else []
            candidate_content, candidate_edits = apply_best_effort(proposed_edits, current_skill.content)
            candidate_skill = None
            if candidate_edits and candidate_content != current_skill.content:
                candidate_skill = self._evolved_skill(
                    current_skill,
                    candidate_content,
                    run_id=request.run_id,
                    batch_index=batch.batch_index,
                )

            score_before = validation_score
            score_after = None
            if candidate_skill is None:
                decision = ValidationDecision.NO_CANDIDATE
            elif not config.algorithm.validation.enabled:
                current_skill = candidate_skill
                decision = ValidationDecision.DISABLED
            else:
                validation_outcomes, rollout_sequence = await self._run_fixed_phase(
                    request=request,
                    scheduler=scheduler,
                    scope=f"validation_candidate_{batch.batch_index}",
                    phase=RolloutPhase.VALIDATION,
                    tasks=request.validation_tasks,
                    skill=candidate_skill,
                    strategy_config=config.rollout.validation,
                    rollout_sequence=rollout_sequence,
                    reflector=reflector,
                )
                all_outcomes.extend(validation_outcomes)
                score_after = self._mean_score(validation_outcomes)
                if score_before is None or score_after is None:
                    decision = ValidationDecision.SCORE_UNAVAILABLE
                elif score_after >= score_before:
                    current_skill = candidate_skill
                    validation_score = score_after
                    decision = ValidationDecision.ACCEPTED
                else:
                    decision = ValidationDecision.REJECTED

            accepted_edits = candidate_edits if current_skill is candidate_skill else []
            record = BatchEvolutionRecord(
                epoch=batch.epoch,
                batch_index=batch.batch_index,
                task_ids=[task.task_id for task in batch.tasks],
                skill_hash_before=skill_before.content_hash,
                candidate_skill_hash=candidate_skill.content_hash if candidate_skill is not None else None,
                skill_hash_after=current_skill.content_hash,
                experiences=experiences,
                patch=patch,
                candidate_edits=candidate_edits,
                applied_edits=accepted_edits,
                train_score=self._mean_score(train_outcomes),
                validation_score_before=score_before,
                validation_score_after=score_after,
                validation_decision=decision,
            )
            batches.append(record)
            await self._emit(
                request.run_id,
                "batch_completed",
                {
                    "batch_index": batch.batch_index,
                    "skill_hash_before": skill_before.content_hash,
                    "candidate_skill_hash": record.candidate_skill_hash,
                    "skill_hash_after": current_skill.content_hash,
                    "experience_count": len(experiences),
                    "candidate_edit_count": len(candidate_edits),
                    "applied_edit_count": len(accepted_edits),
                    "validation_score_before": score_before,
                    "validation_score_after": score_after,
                    "validation_decision": decision.value,
                },
            )

        test_score = None
        if request.test_tasks:
            test_outcomes, rollout_sequence = await self._run_fixed_phase(
                request=request,
                scheduler=scheduler,
                scope="final_test",
                phase=RolloutPhase.TEST,
                tasks=request.test_tasks,
                skill=current_skill,
                strategy_config=config.rollout.test,
                rollout_sequence=rollout_sequence,
                reflector=reflector,
            )
            all_outcomes.extend(test_outcomes)
            test_score = self._mean_score(test_outcomes)

        metrics = self._metrics(batches, all_outcomes, validation_score, test_score)
        changed = current_skill.content_hash != request.base_skill.content_hash
        await self._emit(
            request.run_id,
            "run_finished",
            {"changed": changed, "final_skill_hash": current_skill.content_hash, "metrics": metrics.model_dump()},
        )
        return SkillGrpoWithoutReplayBufferEvolveResult(
            run_id=request.run_id,
            final_skill=current_skill,
            changed=changed,
            trajectories=trajectories_from_rollouts(all_outcomes),
            metrics=metrics,
            batches=batches,
            rollouts=all_outcomes,
        )

    async def _run_fixed_phase(
        self,
        *,
        request: SkillGrpoWithoutReplayBufferEvolveInput,
        scheduler: RolloutScheduler,
        scope: str,
        phase: RolloutPhase,
        tasks: list[Task],
        skill: Skill,
        strategy_config: RolloutStrategyConfig,
        rollout_sequence: int,
        reflector: ReflectionGenerator,
    ) -> tuple[list[RolloutOutcome], int]:
        group_size = self._positive_int(strategy_config.params.get("group_size", 1), "group_size")
        plan = FixedGroupPlan(
            run_id=request.run_id,
            scope=scope,
            phase=phase.value,
            tasks=tasks,
            skills=[skill],
            sequence_start=rollout_sequence,
            group_size=group_size,
            agent_ref=request.config.dataset.agent_ref,
            env_ref=request.config.dataset.env_ref,
            seed=request.config.training.seed,
            temperature=strategy_config.temperature,
            agent_options=request.config.dataset.agent_options,
            env_options=request.config.dataset.env_options,
        )
        specs = self._strategies.get(strategy_config.name).plan(plan)
        await self._emit(
            request.run_id,
            "phase_started",
            {"scope": scope, "phase": phase.value, "rollout_count": len(specs)},
        )
        if phase is RolloutPhase.TRAIN:
            outcomes = await self._run_reflective_train_specs(
                request=request,
                scheduler=scheduler,
                specs=specs,
                reflector=reflector,
            )
        else:
            outcomes = await scheduler.run(specs)
        await self._emit(
            request.run_id,
            "phase_completed",
            {
                "scope": scope,
                "phase": phase.value,
                "rollout_count": len(outcomes),
                "score_mean": self._mean_score(outcomes),
            },
        )
        return outcomes, rollout_sequence + len(specs)

    async def _run_reflective_train_specs(
        self,
        *,
        request: SkillGrpoWithoutReplayBufferEvolveInput,
        scheduler: RolloutScheduler,
        specs: list[RolloutSpec],
        reflector: ReflectionGenerator,
    ) -> list[RolloutOutcome]:
        """Run samples sequentially per task while retaining cross-task concurrency."""

        specs_by_task: dict[str, list[RolloutSpec]] = {}
        for spec in specs:
            specs_by_task.setdefault(spec.task.task_id, []).append(spec)

        async def run_task_chain(task_specs: list[RolloutSpec]) -> list[RolloutOutcome]:
            task_outcomes: list[RolloutOutcome] = []
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
                task_outcomes.append(outcome)
                if self._is_task_success(outcome, success_reward=request.config.training.success_reward):
                    skipped = len(task_specs) - index - 1
                    if skipped:
                        await self._emit(
                            request.run_id,
                            "task_rollouts_early_stopped",
                            {
                                "task_id": outcome.spec.task.task_id,
                                "successful_sample_index": outcome.spec.sample_index,
                                "skipped_rollouts": skipped,
                            },
                        )
                    break
                if index + 1 >= len(task_specs):
                    break

                next_planned = task_specs[index + 1]
                next_spec = next_planned
                if not request.config.algorithm.reflection.enabled:
                    continue

                trajectory = outcome.trajectory
                if trajectory is None:
                    continue
                reflection = await reflector.reflect(trajectory, sample_index=outcome.spec.sample_index)
                if not reflection:
                    continue
                answer = previous_answer(
                    trajectory,
                    max_chars=request.config.algorithm.reflection.max_previous_answer_chars,
                )
                reflection_context = {
                    "prompt_version": REFLECTION_PROMPT_VERSION,
                    "source_rollout_id": outcome.spec.rollout_id,
                    "previous_answer": answer,
                    "content": reflection,
                }
                next_spec = next_planned.model_copy(
                    update={
                        "task": task_with_reflection(next_planned.task, answer=answer, reflection=reflection),
                        "metadata": {**next_planned.metadata, "reflection_context": reflection_context},
                    },
                    deep=True,
                )
                await self._emit(
                    request.run_id,
                    "reflection_generated",
                    {
                        "source_rollout_id": outcome.spec.rollout_id,
                        "target_rollout_id": next_planned.rollout_id,
                        "task_id": outcome.spec.task.task_id,
                        "sample_index": outcome.spec.sample_index,
                    },
                )
            return task_outcomes

        grouped_outcomes = await asyncio.gather(*(run_task_chain(items) for items in specs_by_task.values()))
        outcomes = [outcome for group in grouped_outcomes for outcome in group]
        outcomes.sort(key=lambda item: item.spec.sequence_no)
        return outcomes

    @staticmethod
    def _is_task_success(outcome: RolloutOutcome, *, success_reward: float) -> bool:
        if outcome.trajectory is None:
            return False
        score = outcome.trajectory.reward.score
        return score is not None and score >= success_reward

    def _rollout_event_callback(self, run_id: str) -> Callable[[RolloutOutcome], Awaitable[None]]:
        async def callback(outcome: RolloutOutcome) -> None:
            score = outcome.trajectory.reward.score if outcome.trajectory is not None else None
            await self._emit(
                run_id,
                "rollout_completed" if outcome.succeeded else "rollout_failed",
                {
                    "rollout_id": outcome.spec.rollout_id,
                    "phase": outcome.spec.phase.value,
                    "task_id": outcome.spec.task.task_id,
                    "attempts": len(outcome.attempts),
                    "score": score,
                },
            )

        return callback

    async def _emit(self, run_id: str, name: str, payload: dict[str, Any]) -> None:
        await self._logger.log(
            component_name=name.split("_", 1)[0],
            step_name=name,
            status=self._log_status(name),
            level=LogLevel.ERROR if name.endswith("_failed") else LogLevel.INFO,
            message=self._log_message(name, payload),
            payload={"run_id": run_id, **payload},
        )
        if self._on_event is not None:
            await self._on_event(EvolutionEvent(run_id=run_id, name=name, payload=payload))

    @staticmethod
    def _log_status(name: str) -> str | None:
        if name.endswith("_started"):
            return "started"
        if name.endswith("_failed"):
            return "failed"
        if name.endswith(("_completed", "_finished")) or name == "batch_rollout_summary":
            return "succeeded"
        return None

    @classmethod
    def _log_message(cls, name: str, payload: dict[str, Any]) -> str:
        if name == "run_started":
            return (
                f"run started: batches={payload['total_batches']}, "
                f"reflection={payload['reflection_enabled']}, "
                f"validation_gate={payload['validation_gate_enabled']}"
            )
        if name == "phase_started":
            return f"phase {payload['phase']} started: scope={payload['scope']}, rollouts={payload['rollout_count']}"
        if name in {"rollout_completed", "rollout_failed"}:
            return (
                f"rollout {payload['rollout_id']} {name.removeprefix('rollout_')}: "
                f"phase={payload['phase']}, task={payload['task_id']}, "
                f"score={format_log_value(payload['score'])}, attempts={payload['attempts']}"
            )
        if name == "reflection_generated":
            return (
                f"reflection generated: task={payload['task_id']}, "
                f"rollout={payload['source_rollout_id']}->{payload['target_rollout_id']}"
            )
        if name == "task_rollouts_early_stopped":
            return (
                f"task {payload['task_id']} solved at sample {int(payload['successful_sample_index']) + 1}; "
                f"skipped {payload['skipped_rollouts']} remaining rollout(s)"
            )
        if name == "phase_completed":
            return (
                f"phase {payload['phase']} completed: rollouts={payload['rollout_count']}, "
                f"reward_mean={format_log_value(payload['score_mean'])}"
            )
        if name == "batch_rollout_summary":
            return cls._batch_rollout_summary_message(int(payload["batch_index"]), payload)
        if name == "stage_started":
            return f"stage {payload['stage']} started (batch={int(payload['batch_index']) + 1})"
        if name == "stage_completed":
            return (
                f"stage {payload['stage']} completed (batch={int(payload['batch_index']) + 1}, "
                f"elapsed={float(payload['duration_seconds']):.1f}s)"
            )
        if name == "batch_completed":
            return (
                f"batch {int(payload['batch_index']) + 1} completed: "
                f"validation={payload['validation_decision']}, "
                f"experiences={payload['experience_count']}, applied_edits={payload['applied_edit_count']}"
            )
        if name == "run_finished":
            metrics = payload["metrics"]
            return (
                f"run finished: batches={metrics['batches_completed']}, "
                f"rollouts={metrics['rollouts_completed']} ok/{metrics['rollouts_failed']} failed, "
                f"train/val/test reward={format_log_value(metrics['train_score_mean'])}/"
                f"{format_log_value(metrics['validation_score'])}/"
                f"{format_log_value(metrics['test_score'])}"
            )
        return name

    async def _emit_stage(
        self,
        run_id: str,
        stage: str,
        status: str,
        batch_index: int,
        **payload: Any,
    ) -> None:
        await self._emit(
            run_id,
            f"stage_{status}",
            {"stage": stage, "batch_index": batch_index, **payload},
        )

    @classmethod
    def _evolved_skill(cls, base: Skill, content: str, *, run_id: str, batch_index: int) -> Skill:
        now = datetime.now(UTC)
        blob = {"SKILL.md": content}
        major, minor, patch = (int(value) for value in base.version_label.split("."))
        return base.model_copy(
            update={
                "version_id": f"evolve-run:{run_id}:batch:{batch_index}",
                "parent_version_ids": [base.version_id],
                "version_label": f"{major}.{minor}.{patch + 1}",
                "content_hash": compute_skill_content_hash(blob),
                "status": SkillVersionStatus.DRAFT,
                "origin": SkillVersionOrigin.EVOLUTION,
                "blob": blob,
                "commit_message": f"{cls.algorithm_name} batch:{batch_index}",
                "created_at": now,
                "updated_at": now,
                "metadata": {
                    **base.metadata,
                    "evolution": {
                        "algorithm": cls.algorithm_name,
                        "run_id": run_id,
                        "batch_index": batch_index,
                        "unpersisted_candidate": True,
                    },
                },
            },
            deep=True,
        )

    @staticmethod
    def _mean_score(outcomes: list[RolloutOutcome]) -> float | None:
        scores = [
            float(outcome.trajectory.reward.score)
            for outcome in outcomes
            if outcome.trajectory is not None and outcome.trajectory.reward.score is not None
        ]
        return statistics.fmean(scores) if scores else None

    @staticmethod
    def _metrics(
        batches: list[BatchEvolutionRecord],
        outcomes: list[RolloutOutcome],
        validation_score: float | None,
        test_score: float | None,
    ) -> EvolutionMetrics:
        train_scores = [batch.train_score for batch in batches if batch.train_score is not None]
        return EvolutionMetrics(
            train_score_mean=statistics.fmean(train_scores) if train_scores else None,
            validation_score=validation_score,
            test_score=test_score,
            batches_completed=len(batches),
            batches_accepted=sum(
                batch.validation_decision in {ValidationDecision.DISABLED, ValidationDecision.ACCEPTED}
                for batch in batches
            ),
            batches_rejected=sum(
                batch.validation_decision in {ValidationDecision.REJECTED, ValidationDecision.SCORE_UNAVAILABLE}
                for batch in batches
            ),
            rollouts_completed=sum(outcome.succeeded for outcome in outcomes),
            rollouts_failed=sum(not outcome.succeeded for outcome in outcomes),
            edits_applied=sum(len(batch.applied_edits) for batch in batches),
        )

    @staticmethod
    def _batch_rollout_summary(
        outcomes: list[RolloutOutcome],
        *,
        task_ids: list[str],
        rollouts_per_case: int,
        success_reward: float,
    ) -> dict[str, Any]:
        outcomes_by_task = {task_id: [] for task_id in task_ids}
        for outcome in outcomes:
            outcomes_by_task[outcome.spec.task.task_id].append(outcome)
        solved_at_attempt: Counter[int | None] = Counter()
        for task_outcomes in outcomes_by_task.values():
            task_outcomes.sort(key=lambda item: item.spec.sample_index)
            solved_attempt = next(
                (
                    index
                    for index, outcome in enumerate(task_outcomes, start=1)
                    if SkillGrpoWithoutReplayBuffer._is_task_success(
                        outcome,
                        success_reward=success_reward,
                    )
                ),
                None,
            )
            solved_at_attempt[solved_attempt] += 1
        planned_rollout_count = len(task_ids) * rollouts_per_case
        return {
            "case_count": len(task_ids),
            "rollouts_per_case": rollouts_per_case,
            "success_reward": success_reward,
            "actual_rollout_count": len(outcomes),
            "early_stopped_rollout_count": planned_rollout_count - len(outcomes),
            "solved_at_attempt_histogram": {
                **{str(attempt): solved_at_attempt.get(attempt, 0) for attempt in range(1, rollouts_per_case + 1)},
                "unsolved": solved_at_attempt.get(None, 0),
            },
        }

    @staticmethod
    def _batch_rollout_summary_message(batch_index: int, summary: dict[str, Any]) -> str:
        histogram = summary["solved_at_attempt_histogram"]
        rollouts_per_case = summary["rollouts_per_case"]
        distribution = ", ".join(
            [
                *(f"attempt {attempt}={histogram[str(attempt)]} cases" for attempt in range(1, rollouts_per_case + 1)),
                f"unsolved={histogram['unsolved']} cases",
            ]
        )
        return (
            f"batch={batch_index + 1} rollout stopping distribution: {distribution}; "
            f"actual={summary['actual_rollout_count']}, skipped={summary['early_stopped_rollout_count']}"
        )

    @staticmethod
    def _positive_int(value: object, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
        return value

    @staticmethod
    def _validate_request(request: SkillGrpoWithoutReplayBufferEvolveInput) -> None:
        if not request.train_tasks:
            raise ValueError("train_tasks must not be empty")
        if request.config.algorithm.validation.enabled and not request.validation_tasks:
            raise ValueError("validation_tasks must not be empty when validation gate is enabled")
        train_ids = [task.task_id for task in request.train_tasks]
        if len(train_ids) != len(set(train_ids)):
            raise ValueError("evolution requires unique train task IDs")
        seen: dict[str, Task] = {}
        for task in [*request.train_tasks, *request.validation_tasks, *request.test_tasks]:
            prior = seen.get(task.task_id)
            if prior is not None and prior != task:
                raise ValueError(f"task_id {task.task_id!r} maps to different task payloads")
            seen[task.task_id] = task


__all__ = ["SkillGrpoWithoutReplayBuffer"]
