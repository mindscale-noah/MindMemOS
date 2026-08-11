"""Replay-free Skill GRPO with behavioral validation of each extracted experience set."""

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
from .config import SkillGrpoWithExperienceValidationRunConfig
from .contracts import (
    BatchEvolutionRecord,
    EvolutionMetrics,
    ExperienceSource,
    ExperienceValidationDecision,
    ExperienceValidationRecord,
    ExtractedExperienceSet,
    PatchDecision,
    SkillGrpoWithExperienceValidationEvolveInput,
    SkillGrpoWithExperienceValidationEvolveResult,
)
from .experience import ExperienceExtractor
from .patch import PatchProposer
from .reflection import REFLECTION_PROMPT_VERSION, ReflectionGenerator, previous_answer, task_with_reflection
from .validation import assess_experience, inject_experience, rejected_empty_experience

EventCallback = Callable[[EvolutionEvent], Awaitable[None]]


@register(
    type=ComponentType.ALGO,
    name="skill_grpo_with_experience_validation",
    config_model=SkillGrpoWithExperienceValidationRunConfig,
    capabilities={"evolve"},
    requirements=ComponentRequirements(required_model_roles=frozenset({"chat"})),
)
class SkillGrpoWithExperienceValidation:
    """Accept experiences only when their targeted re-run satisfies its source-specific gate."""

    algorithm_name = "skill_grpo_with_experience_validation"

    def __init__(
        self,
        *,
        config: SkillGrpoWithExperienceValidationRunConfig | None = None,
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
                "skill_grpo_with_experience_validation requires chat_model, agent_resolver, and env_factory"
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
    ) -> SkillGrpoWithExperienceValidationEvolveResult:
        request = self._normalize_request(request)
        with llm_run_context(request.run_id):
            return await self._evolve(request)

    def _normalize_request(self, request: EvolveInput) -> SkillGrpoWithExperienceValidationEvolveInput:
        if isinstance(request, SkillGrpoWithExperienceValidationEvolveInput):
            return request
        if self._config is None:
            raise SkillConfigurationError("skill_grpo_with_experience_validation has no configured run settings")
        return SkillGrpoWithExperienceValidationEvolveInput(
            **request.model_dump(exclude={"validation_tasks"}),
            config=self._config,
        )

    async def _evolve(
        self,
        request: SkillGrpoWithExperienceValidationEvolveInput,
    ) -> SkillGrpoWithExperienceValidationEvolveResult:
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

        await self._emit(
            request.run_id,
            "run_started",
            {
                "base_skill_hash": request.base_skill.content_hash,
                "total_batches": len(plans),
                "reflection_enabled": reflection_config.enabled,
                "validation_set_gate_enabled": False,
                "experience_validation_enabled": True,
            },
        )

        for batch in plans:
            skill_before = current_skill
            train_outcomes, rollout_sequence = await self._run_phase(
                request=request,
                scheduler=scheduler,
                scope=f"batch_{batch.batch_index}",
                phase=RolloutPhase.TRAIN,
                tasks=list(batch.tasks),
                skill=current_skill,
                strategy_config=config.rollout.train,
                rollout_sequence=rollout_sequence,
                reflector=reflector,
                reflective=reflection_config.enabled,
            )
            all_outcomes.extend(train_outcomes)
            await self._emit(
                request.run_id,
                "batch_rollout_summary",
                {
                    "batch_index": batch.batch_index,
                    **self._batch_rollout_summary(
                        train_outcomes,
                        task_ids=[task.task_id for task in batch.tasks],
                        rollouts_per_case=self._positive_int(
                            config.rollout.train.params.get("group_size", 1), "group_size"
                        ),
                        success_reward=config.training.success_reward,
                    ),
                },
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
            experiences = await extractor.extract(
                train_outcomes,
                current_skill,
                mini_batch_size=config.training.mini_batch_size,
                success_reward=config.training.success_reward,
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

            validation_jobs: list[
                Awaitable[tuple[int, ExtractedExperienceSet, ExperienceValidationRecord, list[RolloutOutcome]]]
            ] = []
            task_by_id = {task.task_id: task for task in batch.tasks}
            for experience_index, experience in enumerate(experiences):
                baseline_outcomes = self._baseline_outcomes_for_experience(
                    experience,
                    train_outcomes,
                    success_reward=config.training.success_reward,
                )
                baseline_attempt = (
                    self._first_success_attempt(
                        baseline_outcomes,
                        success_reward=config.training.success_reward,
                    )
                    if experience.source is ExperienceSource.CONTRAST
                    else None
                )
                injected_skill = inject_experience(
                    current_skill,
                    experience,
                    run_id=request.run_id,
                    batch_index=batch.batch_index,
                    experience_index=experience_index,
                )
                validation_tasks = [task_by_id[task_id] for task_id in experience.task_ids]
                group_size = baseline_attempt if experience.source is ExperienceSource.CONTRAST else 1
                if group_size is None:
                    raise RuntimeError("contrast experience is missing its baseline success attempt")
                sequence_start = rollout_sequence
                if injected_skill is not None:
                    # Reserve a deterministic, non-overlapping range before the
                    # validation jobs start concurrently. Reflective early stop
                    # may leave gaps, matching the previous sequential behavior.
                    rollout_sequence += len(validation_tasks) * group_size
                validation_jobs.append(
                    self._validate_experience(
                        request=request,
                        scheduler=scheduler,
                        batch_index=batch.batch_index,
                        experience_index=experience_index,
                        experience=experience,
                        tasks=validation_tasks,
                        skill=injected_skill,
                        group_size=group_size,
                        baseline_outcomes=baseline_outcomes,
                        baseline_attempt=baseline_attempt,
                        rollout_sequence=sequence_start,
                        reflector=reflector,
                    )
                )

            validation_results = await asyncio.gather(*validation_jobs)
            validation_results.sort(key=lambda item: item[0])
            validations: list[ExperienceValidationRecord] = []
            accepted_experiences: list[ExtractedExperienceSet] = []
            for _experience_index, experience, validation, injected_outcomes in validation_results:
                all_outcomes.extend(injected_outcomes)
                validations.append(validation)
                if validation.decision is ExperienceValidationDecision.ACCEPTED:
                    accepted_experiences.append(experience)

            patch = None
            candidate_edits = []
            candidate_skill = None
            if accepted_experiences:
                await self._emit_stage(
                    request.run_id,
                    "patch_proposal",
                    "started",
                    batch.batch_index,
                    accepted_experience_count=len(accepted_experiences),
                )
                stage_started_at = monotonic()
                patch = await proposer.propose(current_skill, accepted_experiences)
                proposed_edits = [item.edit for item in patch.edit_support]
                candidate_content, candidate_edits = apply_best_effort(proposed_edits, current_skill.content)
                if candidate_edits and candidate_content != current_skill.content:
                    candidate_skill = self._evolved_skill(
                        current_skill,
                        candidate_content,
                        run_id=request.run_id,
                        batch_index=batch.batch_index,
                    )
                    current_skill = candidate_skill
                await self._emit_stage(
                    request.run_id,
                    "patch_proposal",
                    "completed",
                    batch.batch_index,
                    duration_seconds=monotonic() - stage_started_at,
                    proposed_edits=patch.proposed_edit_count,
                    applied_edits=len(candidate_edits),
                )

            if not accepted_experiences:
                patch_decision = PatchDecision.NO_ACCEPTED_EXPERIENCE
            elif candidate_skill is None:
                patch_decision = PatchDecision.NO_CANDIDATE
            else:
                patch_decision = PatchDecision.APPLIED

            record = BatchEvolutionRecord(
                epoch=batch.epoch,
                batch_index=batch.batch_index,
                task_ids=[task.task_id for task in batch.tasks],
                skill_hash_before=skill_before.content_hash,
                candidate_skill_hash=candidate_skill.content_hash if candidate_skill is not None else None,
                skill_hash_after=current_skill.content_hash,
                experiences=experiences,
                experience_validations=validations,
                accepted_experiences=accepted_experiences,
                patch=patch,
                candidate_edits=candidate_edits,
                applied_edits=candidate_edits if candidate_skill is not None else [],
                train_score=self._mean_score(train_outcomes),
                patch_decision=patch_decision,
            )
            batches.append(record)
            await self._emit(
                request.run_id,
                "batch_completed",
                {
                    "batch_index": batch.batch_index,
                    "experience_count": len(experiences),
                    "accepted_experience_count": len(accepted_experiences),
                    "applied_edit_count": len(record.applied_edits),
                    "patch_decision": patch_decision.value,
                    "skill_hash_after": current_skill.content_hash,
                },
            )

        test_score = None
        if request.test_tasks:
            test_outcomes, rollout_sequence = await self._run_phase(
                request=request,
                scheduler=scheduler,
                scope="final_test",
                phase=RolloutPhase.TEST,
                tasks=request.test_tasks,
                skill=current_skill,
                strategy_config=config.rollout.test,
                rollout_sequence=rollout_sequence,
                reflector=reflector,
                reflective=False,
            )
            all_outcomes.extend(test_outcomes)
            test_score = self._mean_score(test_outcomes)

        metrics = self._metrics(batches, all_outcomes, test_score)
        changed = current_skill.content_hash != request.base_skill.content_hash
        await self._emit(
            request.run_id,
            "run_finished",
            {"changed": changed, "final_skill_hash": current_skill.content_hash, "metrics": metrics.model_dump()},
        )
        return SkillGrpoWithExperienceValidationEvolveResult(
            run_id=request.run_id,
            final_skill=current_skill,
            changed=changed,
            trajectories=trajectories_from_rollouts(all_outcomes),
            metrics=metrics,
            batches=batches,
            rollouts=all_outcomes,
        )

    async def _validate_experience(
        self,
        *,
        request: SkillGrpoWithExperienceValidationEvolveInput,
        scheduler: RolloutScheduler,
        batch_index: int,
        experience_index: int,
        experience: ExtractedExperienceSet,
        tasks: list[Task],
        skill: Skill | None,
        group_size: int,
        baseline_outcomes: list[RolloutOutcome],
        baseline_attempt: int | None,
        rollout_sequence: int,
        reflector: ReflectionGenerator,
    ) -> tuple[int, ExtractedExperienceSet, ExperienceValidationRecord, list[RolloutOutcome]]:
        if skill is None:
            injected_outcomes: list[RolloutOutcome] = []
            validation = rejected_empty_experience(
                experience,
                experience_index=experience_index,
                baseline_first_success_attempt=baseline_attempt,
            )
        else:
            injected_outcomes, next_sequence = await self._run_experience_validation(
                request=request,
                scheduler=scheduler,
                batch_index=batch_index,
                experience_index=experience_index,
                experience=experience,
                tasks=tasks,
                skill=skill,
                group_size=group_size,
                baseline_outcomes=baseline_outcomes,
                rollout_sequence=rollout_sequence,
                reflector=reflector,
            )
            expected_next_sequence = rollout_sequence + len(tasks) * group_size
            if next_sequence != expected_next_sequence:
                raise RuntimeError(
                    "experience validation planned an unexpected rollout sequence range: "
                    f"expected {expected_next_sequence}, got {next_sequence}"
                )
            validation = assess_experience(
                experience,
                experience_index=experience_index,
                injected_outcomes=injected_outcomes,
                baseline_first_success_attempt=baseline_attempt,
                success_reward=request.config.training.success_reward,
            )
        await self._emit(
            request.run_id,
            "experience_validation_completed",
            {
                "batch_index": batch_index,
                "experience_index": experience_index,
                "source": experience.source.value,
                "task_ids": experience.task_ids,
                "decision": validation.decision.value,
                "baseline_success_rate": validation.baseline_success_rate,
                "injected_success_rate": validation.injected_success_rate,
                "baseline_first_success_attempt": validation.baseline_first_success_attempt,
                "injected_first_success_attempt": validation.injected_first_success_attempt,
                "reason": validation.reason,
            },
        )
        return experience_index, experience, validation, injected_outcomes

    async def _run_experience_validation(
        self,
        *,
        request: SkillGrpoWithExperienceValidationEvolveInput,
        scheduler: RolloutScheduler,
        batch_index: int,
        experience_index: int,
        experience: ExtractedExperienceSet,
        tasks: list[Task],
        skill: Skill,
        group_size: int,
        baseline_outcomes: list[RolloutOutcome],
        rollout_sequence: int,
        reflector: ReflectionGenerator,
    ) -> tuple[list[RolloutOutcome], int]:
        strategy = request.config.rollout.experience_validation
        scope = f"batch_{batch_index}_experience_{experience_index}_{experience.source.value}"
        plan = FixedGroupPlan(
            run_id=request.run_id,
            scope=scope,
            phase=RolloutPhase.VALIDATION.value,
            tasks=tasks,
            skills=[skill],
            sequence_start=rollout_sequence,
            group_size=group_size,
            agent_ref=request.config.dataset.agent_ref,
            env_ref=request.config.dataset.env_ref,
            seed=request.config.training.seed,
            temperature=strategy.temperature,
            agent_options=request.config.dataset.agent_options,
            env_options=request.config.dataset.env_options,
        )
        specs = self._strategies.get(strategy.name).plan(plan)
        specs = self._pair_validation_seeds(specs, baseline_outcomes, experience.source)
        await self._emit(
            request.run_id,
            "experience_validation_started",
            {
                "batch_index": batch_index,
                "experience_index": experience_index,
                "source": experience.source.value,
                "task_ids": experience.task_ids,
                "planned_rollouts": len(specs),
                "reflective": experience.source is ExperienceSource.CONTRAST,
            },
        )
        if experience.source is ExperienceSource.CONTRAST:
            outcomes = await self._run_reflective_specs(
                request=request,
                scheduler=scheduler,
                specs=specs,
                reflector=reflector,
                reflection_enabled=True,
            )
        else:
            outcomes = await scheduler.run(specs)
        return outcomes, rollout_sequence + len(specs)

    async def _run_phase(
        self,
        *,
        request: SkillGrpoWithExperienceValidationEvolveInput,
        scheduler: RolloutScheduler,
        scope: str,
        phase: RolloutPhase,
        tasks: list[Task],
        skill: Skill,
        strategy_config: RolloutStrategyConfig,
        rollout_sequence: int,
        reflector: ReflectionGenerator,
        reflective: bool,
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
        outcomes = (
            await self._run_reflective_specs(
                request=request,
                scheduler=scheduler,
                specs=specs,
                reflector=reflector,
                reflection_enabled=reflective,
            )
            if phase is RolloutPhase.TRAIN
            else await scheduler.run(specs)
        )
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

    async def _run_reflective_specs(
        self,
        *,
        request: SkillGrpoWithExperienceValidationEvolveInput,
        scheduler: RolloutScheduler,
        specs: list[RolloutSpec],
        reflector: ReflectionGenerator,
        reflection_enabled: bool,
    ) -> list[RolloutOutcome]:
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
                if self._is_success(outcome, request.config.training.success_reward):
                    break
                if index + 1 >= len(task_specs):
                    break
                next_planned = task_specs[index + 1]
                next_spec = next_planned
                if not reflection_enabled or outcome.trajectory is None:
                    continue
                reflection = await reflector.reflect(outcome.trajectory, sample_index=outcome.spec.sample_index)
                if not reflection:
                    continue
                answer = previous_answer(
                    outcome.trajectory,
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

    @classmethod
    def _baseline_outcomes_for_experience(
        cls,
        experience: ExtractedExperienceSet,
        outcomes: list[RolloutOutcome],
        *,
        success_reward: float,
    ) -> list[RolloutOutcome]:
        grouped: dict[str, list[RolloutOutcome]] = {task_id: [] for task_id in experience.task_ids}
        for outcome in outcomes:
            if outcome.spec.task.task_id in grouped and outcome.trajectory is not None:
                grouped[outcome.spec.task.task_id].append(outcome)
        for items in grouped.values():
            items.sort(key=lambda item: item.spec.sample_index)

        if experience.source is ExperienceSource.CONTRAST:
            return grouped[experience.task_ids[0]]
        selected: list[RolloutOutcome] = []
        for task_id in experience.task_ids:
            items = grouped[task_id]
            if experience.source is ExperienceSource.FAILURE:
                match = next((item for item in items if not cls._is_success(item, success_reward)), None)
            else:
                match = next((item for item in reversed(items) if cls._is_success(item, success_reward)), None)
            if match is None:
                raise RuntimeError(f"missing {experience.source.value} baseline for task {task_id!r}")
            selected.append(match)
        return selected

    @staticmethod
    def _pair_validation_seeds(
        specs: list[RolloutSpec],
        baseline_outcomes: list[RolloutOutcome],
        source: ExperienceSource,
    ) -> list[RolloutSpec]:
        if source is ExperienceSource.CONTRAST:
            seeds = {
                (outcome.spec.task.task_id, outcome.spec.sample_index): outcome.spec.seed
                for outcome in baseline_outcomes
            }
        else:
            seeds = {(outcome.spec.task.task_id, 0): outcome.spec.seed for outcome in baseline_outcomes}
        return [
            spec.model_copy(update={"seed": seeds.get((spec.task.task_id, spec.sample_index), spec.seed)}, deep=True)
            for spec in specs
        ]

    @classmethod
    def _first_success_attempt(cls, outcomes: list[RolloutOutcome], *, success_reward: float) -> int | None:
        ordered = sorted(outcomes, key=lambda item: item.spec.sample_index)
        return next(
            (outcome.spec.sample_index + 1 for outcome in ordered if cls._is_success(outcome, success_reward)),
            None,
        )

    @staticmethod
    def _is_success(outcome: RolloutOutcome, success_reward: float) -> bool:
        score = outcome.trajectory.reward.score if outcome.trajectory is not None else None
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
        status = None
        if name.endswith("_started"):
            status = "started"
        elif name.endswith(("_completed", "_finished")) or name == "batch_rollout_summary":
            status = "succeeded"
        elif name.endswith("_failed"):
            status = "failed"
        await self._logger.log(
            component_name=name.split("_", 1)[0],
            step_name=name,
            status=status,
            level=LogLevel.ERROR if name.endswith("_failed") else LogLevel.INFO,
            message=self._log_message(name, payload),
            payload={"run_id": run_id, **payload},
        )
        if self._on_event is not None:
            await self._on_event(EvolutionEvent(run_id=run_id, name=name, payload=payload))

    @staticmethod
    def _log_message(name: str, payload: dict[str, Any]) -> str:
        if name == "run_started":
            return (
                f"run started: batches={payload['total_batches']}, reflection={payload['reflection_enabled']}, "
                "validation_set_gate=false, experience_validation=true"
            )
        if name in {"rollout_completed", "rollout_failed"}:
            return (
                f"rollout {payload['rollout_id']}: phase={payload['phase']}, task={payload['task_id']}, "
                f"score={format_log_value(payload['score'])}"
            )
        if name == "experience_validation_completed":
            return (
                f"experience {int(payload['experience_index']) + 1} ({payload['source']}) "
                f"{payload['decision']}: {payload['reason']}"
            )
        if name == "batch_completed":
            return (
                f"batch {int(payload['batch_index']) + 1}: experiences={payload['experience_count']}, "
                f"accepted={payload['accepted_experience_count']}, edits={payload['applied_edit_count']}"
            )
        if name == "run_finished":
            metrics = payload["metrics"]
            return (
                f"run finished: batches={metrics['batches_completed']}, "
                f"experiences={metrics['experiences_accepted']}/{metrics['experiences_extracted']} accepted, "
                f"edits={metrics['edits_applied']}, test={format_log_value(metrics['test_score'])}"
            )
        return name

    async def _emit_stage(self, run_id: str, stage: str, status: str, batch_index: int, **payload: Any) -> None:
        await self._emit(run_id, f"stage_{status}", {"stage": stage, "batch_index": batch_index, **payload})

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
        test_score: float | None,
    ) -> EvolutionMetrics:
        train_scores = [batch.train_score for batch in batches if batch.train_score is not None]
        validation_rollouts = sum(
            len(validation.rollouts) for batch in batches for validation in batch.experience_validations
        )
        return EvolutionMetrics(
            train_score_mean=statistics.fmean(train_scores) if train_scores else None,
            test_score=test_score,
            batches_completed=len(batches),
            rollouts_completed=sum(outcome.succeeded for outcome in outcomes),
            rollouts_failed=sum(not outcome.succeeded for outcome in outcomes),
            experiences_extracted=sum(len(batch.experiences) for batch in batches),
            experiences_accepted=sum(len(batch.accepted_experiences) for batch in batches),
            experience_validation_rollouts=validation_rollouts,
            edits_applied=sum(len(batch.applied_edits) for batch in batches),
        )

    @classmethod
    def _batch_rollout_summary(
        cls,
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
                    if cls._is_success(outcome, success_reward)
                ),
                None,
            )
            solved_at_attempt[solved_attempt] += 1
        planned = len(task_ids) * rollouts_per_case
        return {
            "case_count": len(task_ids),
            "rollouts_per_case": rollouts_per_case,
            "success_reward": success_reward,
            "actual_rollout_count": len(outcomes),
            "early_stopped_rollout_count": planned - len(outcomes),
            "solved_at_attempt_histogram": {
                **{str(attempt): solved_at_attempt.get(attempt, 0) for attempt in range(1, rollouts_per_case + 1)},
                "unsolved": solved_at_attempt.get(None, 0),
            },
        }

    @staticmethod
    def _positive_int(value: object, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
        return value

    @staticmethod
    def _validate_request(request: SkillGrpoWithExperienceValidationEvolveInput) -> None:
        if not request.train_tasks:
            raise ValueError("train_tasks must not be empty")
        task_ids = [task.task_id for task in request.train_tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("evolution requires unique train task IDs")
        seen: dict[str, Task] = {}
        for task in [*request.train_tasks, *request.validation_tasks, *request.test_tasks]:
            prior = seen.get(task.task_id)
            if prior is not None and prior != task:
                raise ValueError(f"task_id {task.task_id!r} maps to different task payloads")
            seen[task.task_id] = task


__all__ = ["SkillGrpoWithExperienceValidation"]
