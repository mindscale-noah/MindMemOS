"""Complete Skill GRPO-style evolution with fused replay and ablation."""

from __future__ import annotations

import asyncio
import statistics
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
from .ablation import AblationCandidate, AblationEvaluator
from .batch_planner import TaskBatch, TaskBatchPlanner
from .config import RolloutStrategyConfig, SkillGrpoRunConfig
from .contracts import (
    BatchEvolutionRecord,
    CandidateEvaluationRecord,
    EvolutionEvent,
    EvolutionMetrics,
    EvolutionState,
    ProcessArtifact,
    RolloutOutcome,
    RolloutPhase,
    RolloutSpec,
    SkillGrpoEvolveInput,
    SkillGrpoEvolveResult,
    SkillTextEdit,
)
from .experience import ExperienceExtractor
from .fileedit import apply_best_effort
from .models import ChatModel, EmbeddingModel
from .patch import PatchProposer
from .replay_buffer import FusedReplayBuffer, TouchedCluster
from .rollout import (
    AblationTarget,
    AgentResolver,
    EnvFactory,
    FixedGroupPlan,
    MappingAgentResolver,
    PairedAblationPlan,
    RegistryEnvFactory,
    RolloutScheduler,
    RolloutStrategyRegistry,
)
from .state import config_fingerprint, input_fingerprint, validate_resume

EventCallback = Callable[[EvolutionEvent], Awaitable[None]]


@register(
    type=ComponentType.ALGO,
    name="skill_grpo_with_replay_buffer",
    config_model=SkillGrpoRunConfig,
    capabilities={"evolve"},
    requirements=ComponentRequirements(required_model_roles=frozenset({"chat"})),
)
class SkillGrpoWithReplayBuffer:
    """Own the full train/ablate/update/test loop and return unpersisted data."""

    algorithm_name = "skill_grpo_with_replay_buffer"

    def __init__(
        self,
        *,
        config: SkillGrpoRunConfig | None = None,
        context: EvolveAlgorithmContext | None = None,
        chat_model: ChatModel | None = None,
        agent_resolver: AgentResolver | None = None,
        env_factory: EnvFactory | None = None,
        embedding_model: EmbeddingModel | None = None,
        rollout_strategies: RolloutStrategyRegistry | None = None,
        on_event: EventCallback | None = None,
        logger: AlgorithmLogger | None = None,
    ) -> None:
        if context is not None:
            chat_model = context.models.get("chat")
            embedding_model = context.models.get("embedding", embedding_model)
            agent_resolver = MappingAgentResolver(context.agents)
            env_factory = RegistryEnvFactory()
        if chat_model is None or agent_resolver is None or env_factory is None:
            raise SkillConfigurationError(
                "skill_grpo_with_replay_buffer requires chat_model, agent_resolver, and env_factory"
            )
        self._config = config
        self._chat_model = chat_model
        self._embedding_model = embedding_model
        self._agent_resolver = agent_resolver
        self._env_factory = env_factory
        self._strategies = rollout_strategies or RolloutStrategyRegistry.with_builtins()
        self._on_event = on_event
        self._logger = logger or AlgorithmLogger(algorithm_name=self.algorithm_name)

    async def evolve(self, request: EvolveInput) -> SkillGrpoEvolveResult:
        request = self._normalize_request(request)
        with llm_run_context(request.run_id):
            return await self._evolve(request)

    def _normalize_request(self, request: EvolveInput) -> SkillGrpoEvolveInput:
        if isinstance(request, SkillGrpoEvolveInput):
            return request
        if self._config is None:
            raise SkillConfigurationError("skill_grpo_with_replay_buffer has no configured run settings")
        return SkillGrpoEvolveInput(
            **request.model_dump(),
            config=self._config,
        )

    async def _evolve(self, request: SkillGrpoEvolveInput) -> SkillGrpoEvolveResult:
        run_started_at = monotonic()
        self._validate_tasks(request)
        config = request.config
        input_hash = input_fingerprint(
            request.base_skill,
            request.train_tasks,
            request.validation_tasks,
            request.test_tasks,
        )
        config_hash = config_fingerprint(config)
        state = self._initial_state(request, input_hash=input_hash, config_hash=config_hash)
        current_skill = state.current_skill
        all_outcomes = [outcome.model_copy(deep=True) for outcome in state.rollout_outcomes]
        all_candidates = [
            candidate.model_copy(deep=True)
            for completed_batch in state.batches
            for candidate in completed_batch.candidates
        ]
        task_registry = {
            task.task_id: task for task in [*request.train_tasks, *request.validation_tasks, *request.test_tasks]
        }
        buffer = FusedReplayBuffer(
            chat_model=self._chat_model,
            embedding_model=self._embedding_model if config.algorithm.replay.use_embeddings else None,
            config=config.algorithm.replay,
            clusters=state.replay_clusters,
            embedding_dimension=state.embedding_dimension,
        )

        async def on_rollout(outcome: RolloutOutcome) -> None:
            score = outcome.trajectory.reward.score if outcome.trajectory is not None else None
            await self._emit(
                request.run_id,
                "rollout_completed" if outcome.succeeded else "rollout_failed",
                {
                    "rollout_id": outcome.spec.rollout_id,
                    "sequence_no": outcome.spec.sequence_no,
                    "phase": outcome.spec.phase.value,
                    "task_id": outcome.spec.task.task_id,
                    "attempts": len(outcome.attempts),
                    "score": score,
                },
            )

        scheduler = RolloutScheduler(
            agent_resolver=self._agent_resolver,
            env_factory=self._env_factory,
            config=config.rollout,
            on_outcome=on_rollout,
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
        ablation = AblationEvaluator(
            config.algorithm.ablation,
            success_reward=config.training.success_reward,
        )
        ablation.load_random_state(state.ablation_rng_state)
        planner = TaskBatchPlanner()

        plans = planner.build(
            request.train_tasks,
            epochs=config.training.epochs,
            batch_size=config.training.batch_size,
            seed=config.training.seed,
        )
        await self._emit(
            request.run_id,
            "run_started",
            {
                "base_skill_hash": request.base_skill.content_hash,
                "epochs": config.training.epochs,
                "batch_size": config.training.batch_size,
                "total_batches": len(plans),
                "completed_batches": len(state.batches),
                "train_tasks": len(request.train_tasks),
                "validation_tasks": len(request.validation_tasks),
                "test_tasks": len(request.test_tasks),
                "resumed": request.resume_state is not None,
            },
        )
        for batch in plans:
            if batch.batch_index <= state.completed_batch_index:
                continue
            batch_started_at = monotonic()
            await self._emit(
                request.run_id,
                "batch_started",
                {
                    "epoch": batch.epoch,
                    "batch_index": batch.batch_index,
                    "batch_number": batch.batch_index + 1,
                    "total_batches": len(plans),
                    "task_count": len(batch.tasks),
                },
            )
            skill_before_batch = current_skill
            batch_record, batch_outcomes, candidates, current_skill = await self._run_batch(
                request=request,
                batch=batch,
                current_skill=current_skill,
                state=state,
                task_registry=task_registry,
                buffer=buffer,
                scheduler=scheduler,
                extractor=extractor,
                proposer=proposer,
                ablation=ablation,
            )
            all_outcomes.extend(batch_outcomes)
            all_candidates.extend(candidates)
            state.current_skill = current_skill
            state.completed_batch_index = batch.batch_index
            state.replay_clusters = buffer.snapshot()
            state.embedding_model_identity = self._embedding_identity(config)
            state.embedding_dimension = buffer.embedding_dimension
            state.ablation_rng_state = ablation.random_state()
            state.batches.append(batch_record)
            state.completed_rollout_ids.extend(
                outcome.spec.rollout_id for outcome in batch_outcomes if outcome.succeeded
            )
            state.rollout_outcomes.extend(outcome.model_copy(deep=True) for outcome in batch_outcomes)
            self._update_metrics(state.metrics, batch_outcomes, candidates, state.batches)
            await self._emit(
                request.run_id,
                "batch_completed",
                {
                    "epoch": batch.epoch,
                    "batch_index": batch.batch_index,
                    "skill_hash": current_skill.content_hash,
                    "applied_edits": len(batch_record.applied_edits),
                    "experiences": len(batch_record.experiences),
                    "candidates": len(candidates),
                    "train_score": batch_record.train_score,
                    "validation_score": batch_record.validation_score,
                    "gate_kept": sum(candidate.chosen for candidate in candidates),
                    "gate_rejected": sum(not candidate.chosen for candidate in candidates),
                    "skill_chars_before": len(skill_before_batch.content),
                    "skill_chars_after": len(current_skill.content),
                    "skill_chars_delta": len(current_skill.content) - len(skill_before_batch.content),
                    "skill_lines_before": len(skill_before_batch.content.splitlines()),
                    "skill_lines_after": len(current_skill.content.splitlines()),
                    "applied_edit_details": [edit.model_dump(mode="json") for edit in batch_record.applied_edits],
                    "rollouts_completed": state.metrics.rollouts_completed,
                    "rollouts_failed": state.metrics.rollouts_failed,
                    "duration_seconds": monotonic() - batch_started_at,
                },
            )
            await self._emit(
                request.run_id,
                "checkpoint_ready",
                {"state": state.model_dump(mode="json")},
                critical=True,
            )

        test_outcomes: list[RolloutOutcome] = []
        if request.test_tasks and not state.final_test_completed:
            test_outcomes = await self._run_fixed_phase(
                run_id=request.run_id,
                scope="final_test",
                phase=RolloutPhase.TEST,
                tasks=request.test_tasks,
                skill=current_skill,
                strategy_config=config.rollout.test,
                config=config,
                state=state,
                scheduler=scheduler,
            )
            all_outcomes.extend(test_outcomes)
            state.completed_rollout_ids.extend(
                outcome.spec.rollout_id for outcome in test_outcomes if outcome.succeeded
            )
            state.rollout_outcomes.extend(outcome.model_copy(deep=True) for outcome in test_outcomes)
            state.metrics.test_score_mean = self._mean_score(test_outcomes)
            state.metrics.rollouts_completed += sum(outcome.succeeded for outcome in test_outcomes)
            state.metrics.rollouts_failed += sum(not outcome.succeeded for outcome in test_outcomes)
            state.final_test_completed = True
            await self._emit(
                request.run_id,
                "checkpoint_ready",
                {"state": state.model_dump(mode="json")},
                critical=True,
            )

        changed = current_skill.content_hash != request.base_skill.content_hash
        artifacts = [
            ProcessArtifact(
                name="evolution_state",
                content=state.model_dump(mode="json"),
            )
        ]
        await self._emit(
            request.run_id,
            "run_finished",
            {
                "changed": changed,
                "final_skill_hash": current_skill.content_hash,
                "metrics": state.metrics.model_dump(mode="json"),
                "duration_seconds": monotonic() - run_started_at,
            },
        )
        return SkillGrpoEvolveResult(
            run_id=request.run_id,
            final_skill=current_skill,
            changed=changed,
            trajectories=trajectories_from_rollouts(all_outcomes),
            metrics=state.metrics,
            state=state,
            batches=list(state.batches),
            rollouts=all_outcomes,
            candidates=all_candidates,
            artifacts=artifacts,
        )

    async def _run_batch(
        self,
        *,
        request: SkillGrpoEvolveInput,
        batch: TaskBatch,
        current_skill: Skill,
        state: EvolutionState,
        task_registry: dict[str, Task],
        buffer: FusedReplayBuffer,
        scheduler: RolloutScheduler,
        extractor: ExperienceExtractor,
        proposer: PatchProposer,
        ablation: AblationEvaluator,
    ) -> tuple[
        BatchEvolutionRecord,
        list[RolloutOutcome],
        list[CandidateEvaluationRecord],
        Skill,
    ]:
        config = request.config
        train_outcomes = await self._run_fixed_phase(
            run_id=request.run_id,
            scope=f"batch_{batch.batch_index}",
            phase=RolloutPhase.TRAIN,
            tasks=list(batch.tasks),
            skill=current_skill,
            strategy_config=config.rollout.train,
            config=config,
            state=state,
            scheduler=scheduler,
        )
        trajectories = [outcome.trajectory for outcome in train_outcomes if outcome.trajectory is not None]
        if config.algorithm.experience.skip_all_failed_tasks:
            by_task: dict[str, list] = {}
            for trajectory in trajectories:
                by_task.setdefault(trajectory.task.task_id, []).append(trajectory)
            trajectories = [
                trajectory
                for trajectory in trajectories
                if not all(
                    item.reward.score is None or item.reward.score < config.training.success_reward
                    for item in by_task[trajectory.task.task_id]
                )
            ]
        await self._emit_stage(
            request.run_id,
            "experience_extraction",
            "started",
            batch.batch_index,
            trajectory_count=len(trajectories),
            task_count=len({trajectory.task.task_id for trajectory in trajectories}),
        )
        stage_started_at = monotonic()
        experiences = await extractor.extract(trajectories, current_skill) if trajectories else []
        await self._emit_stage(
            request.run_id,
            "experience_extraction",
            "completed",
            batch.batch_index,
            duration_seconds=monotonic() - stage_started_at,
            experience_count=len(experiences),
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
        edits_with_meta: list[tuple[SkillTextEdit, str, float]] = []
        if patch is not None:
            for item in patch.edit_support:
                source_ids = [
                    experiences[index - 1].task_id
                    for index in item.supporting_experience_sets
                    if 1 <= index <= len(experiences)
                ]
                if not source_ids:
                    source_ids = [experience.task_id for experience in experiences]
                edits_with_meta.extend((item.edit, task_id, 0.0) for task_id in dict.fromkeys(source_ids))
        await self._emit_stage(
            request.run_id,
            "replay_ingest",
            "started",
            batch.batch_index,
            edit_records=len(edits_with_meta),
        )
        stage_started_at = monotonic()
        touched = await buffer.ingest(batch.batch_index, edits_with_meta)
        await self._emit_stage(
            request.run_id,
            "replay_ingest",
            "completed",
            batch.batch_index,
            duration_seconds=monotonic() - stage_started_at,
            touched_clusters=len(touched),
            replay_clusters=len(buffer.snapshot()),
        )
        case_scores = self._case_total_scores(train_outcomes)
        candidates, state.ablation_sample_counter = ablation.build_candidates(
            run_id=request.run_id,
            batch_index=batch.batch_index,
            current_skill=current_skill,
            touched=touched,
            task_registry=task_registry,
            case_total_scores=case_scores,
            skill_factory=lambda base, content, tag: self._evolved_skill(
                base,
                content,
                run_id=request.run_id,
                tag=tag,
            ),
            sample_counter=state.ablation_sample_counter,
            min_cluster_edits=config.algorithm.replay.min_cluster_edits,
        )
        await self._emit_stage(
            request.run_id,
            "candidate_selection",
            "completed",
            batch.batch_index,
            touched_clusters=len(touched),
            candidate_count=len(candidates),
            min_cluster_edits=config.algorithm.replay.min_cluster_edits,
            decisions=self._replay_gate_decisions(
                touched,
                candidates,
                min_cluster_edits=config.algorithm.replay.min_cluster_edits,
            ),
        )
        ablation_outcomes: list[RolloutOutcome] = []
        candidate_records: list[CandidateEvaluationRecord] = []
        if candidates:
            ablation_outcomes = await self._run_ablation(
                request=request,
                batch=batch,
                current_skill=current_skill,
                candidates=candidates,
                state=state,
                scheduler=scheduler,
            )
            candidate_records = ablation.score(candidates, ablation_outcomes)
        await self._emit_stage(
            request.run_id,
            "ablation_scoring",
            "completed",
            batch.batch_index,
            candidate_count=len(candidate_records),
            kept_count=sum(record.chosen for record in candidate_records),
            rejected_count=sum(not record.chosen for record in candidate_records),
            decisions=[
                {
                    **record.model_dump(mode="json"),
                    "decision": "kept" if record.chosen else "rejected",
                }
                for record in candidate_records
            ],
        )

        chosen = [record for record in candidate_records if record.chosen]
        updated_content, applied = apply_best_effort([record.edit for record in chosen], current_skill.content)
        updated_skill = current_skill
        if applied and updated_content != current_skill.content:
            updated_skill = self._evolved_skill(
                current_skill,
                updated_content,
                run_id=request.run_id,
                tag=f"batch:{batch.batch_index}",
            )
        if chosen:
            # A cluster that passed ablation/top-k counts as used even if final
            # best-effort conflict resolution keeps only a higher-ranked edit.
            buffer.mark_committed({record.cluster_id for record in chosen})
        await self._emit_stage(
            request.run_id,
            "skill_update",
            "completed",
            batch.batch_index,
            kept_by_gate=len(chosen),
            applied_count=len(applied),
            skipped_after_gate=len(chosen) - len(applied),
            chars_before=len(current_skill.content),
            chars_after=len(updated_content),
            chars_delta=len(updated_content) - len(current_skill.content),
            lines_before=len(current_skill.content.splitlines()),
            lines_after=len(updated_content.splitlines()),
            edits=[edit.model_dump(mode="json") for edit in applied],
        )

        validation_outcomes: list[RolloutOutcome] = []
        validation_score = None
        every = config.algorithm.validation.every_batches
        if request.validation_tasks and every > 0 and (batch.batch_index + 1) % every == 0:
            validation_outcomes = await self._run_fixed_phase(
                run_id=request.run_id,
                scope=f"validation_{batch.batch_index}",
                phase=RolloutPhase.VALIDATION,
                tasks=request.validation_tasks,
                skill=updated_skill,
                strategy_config=config.rollout.validation,
                config=config,
                state=state,
                scheduler=scheduler,
            )
            validation_score = self._mean_score(validation_outcomes)
        outcomes = [*train_outcomes, *ablation_outcomes, *validation_outcomes]
        record = BatchEvolutionRecord(
            epoch=batch.epoch,
            batch_index=batch.batch_index,
            task_ids=[task.task_id for task in batch.tasks],
            skill_hash_before=current_skill.content_hash,
            skill_hash_after=updated_skill.content_hash,
            experiences=experiences,
            patch=patch,
            candidates=candidate_records,
            applied_edits=applied,
            train_score=self._mean_score(train_outcomes),
            validation_score=validation_score,
        )
        return record, outcomes, candidate_records, updated_skill

    async def _run_fixed_phase(
        self,
        *,
        run_id: str,
        scope: str,
        phase: RolloutPhase,
        tasks: list[Task],
        skill: Skill,
        strategy_config: RolloutStrategyConfig,
        config: SkillGrpoRunConfig,
        state: EvolutionState,
        scheduler: RolloutScheduler,
    ) -> list[RolloutOutcome]:
        group_size = self._positive_int(strategy_config.params.get("group_size", 1), "group_size")
        plan = FixedGroupPlan(
            run_id=run_id,
            scope=scope,
            phase=phase.value,
            tasks=tasks,
            skills=[skill],
            sequence_start=state.rollout_sequence,
            group_size=group_size,
            agent_ref=config.dataset.agent_ref,
            env_ref=config.dataset.env_ref,
            seed=config.training.seed,
            temperature=strategy_config.temperature,
            agent_options=config.dataset.agent_options,
            env_options=config.dataset.env_options,
        )
        specs = self._strategies.get(strategy_config.name).plan(plan)
        state.rollout_sequence += len(specs)
        return await self._run_scheduled_phase(
            run_id=run_id,
            scope=scope,
            phase=phase,
            specs=specs,
            scheduler=scheduler,
        )

    async def _run_ablation(
        self,
        *,
        request: SkillGrpoEvolveInput,
        batch: TaskBatch,
        current_skill: Skill,
        candidates: list[AblationCandidate],
        state: EvolutionState,
        scheduler: RolloutScheduler,
    ) -> list[RolloutOutcome]:
        strategy_config = request.config.rollout.ablation
        samples = self._positive_int(strategy_config.params.get("samples_per_case", 1), "samples_per_case")
        tasks_by_id = {task.task_id: task for candidate in candidates for task in candidate.sampled_tasks}
        plan = PairedAblationPlan(
            run_id=request.run_id,
            scope=f"ablation_{batch.batch_index}",
            tasks=[tasks_by_id[key] for key in sorted(tasks_by_id)],
            before_skill=current_skill,
            targets=[
                AblationTarget(
                    candidate_id=item.candidate_id,
                    skill=item.skill,
                    task_ids=[task.task_id for task in item.sampled_tasks],
                )
                for item in candidates
            ],
            sequence_start=state.rollout_sequence,
            sample_index_start=state.ablation_rollout_index + 1,
            samples_per_case=samples,
            agent_ref=request.config.dataset.agent_ref,
            env_ref=request.config.dataset.env_ref,
            seed=request.config.algorithm.ablation.seed,
            temperature=strategy_config.temperature,
            agent_options=request.config.dataset.agent_options,
            env_options=request.config.dataset.env_options,
        )
        specs = self._strategies.get(strategy_config.name).plan(plan)
        state.rollout_sequence += len(specs)
        state.ablation_rollout_index += len(specs)
        before_specs = [spec for spec in specs if spec.phase is RolloutPhase.ABLATION_BEFORE]
        after_specs = [spec for spec in specs if spec.phase is RolloutPhase.ABLATION_AFTER]
        unsupported = [
            spec for spec in specs if spec.phase not in {RolloutPhase.ABLATION_BEFORE, RolloutPhase.ABLATION_AFTER}
        ]
        if unsupported:
            raise ValueError("ablation rollout strategy emitted a non-ablation phase")
        before, after = await asyncio.gather(
            self._run_scheduled_phase(
                run_id=request.run_id,
                scope=f"ablation_{batch.batch_index}",
                phase=RolloutPhase.ABLATION_BEFORE,
                specs=before_specs,
                scheduler=scheduler,
            ),
            self._run_scheduled_phase(
                run_id=request.run_id,
                scope=f"ablation_{batch.batch_index}",
                phase=RolloutPhase.ABLATION_AFTER,
                specs=after_specs,
                scheduler=scheduler,
            ),
        )
        return [*before, *after]

    async def _run_scheduled_phase(
        self,
        *,
        run_id: str,
        scope: str,
        phase: RolloutPhase,
        specs: list[RolloutSpec],
        scheduler: RolloutScheduler,
    ) -> list[RolloutOutcome]:
        started_at = monotonic()
        await self._emit(
            run_id,
            "phase_started",
            {
                "scope": scope,
                "phase": phase.value,
                "task_count": len({spec.task.task_id for spec in specs}),
                "rollout_count": len(specs),
            },
        )
        try:
            outcomes = await scheduler.run(specs)
        except Exception as exc:
            await self._emit(
                run_id,
                "phase_failed",
                {
                    "scope": scope,
                    "phase": phase.value,
                    "rollout_count": len(specs),
                    "duration_seconds": monotonic() - started_at,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            raise
        scores = [
            float(outcome.trajectory.reward.score)
            for outcome in outcomes
            if outcome.trajectory is not None and outcome.trajectory.reward.score is not None
        ]
        await self._emit(
            run_id,
            "phase_completed",
            {
                "scope": scope,
                "phase": phase.value,
                "rollout_count": len(outcomes),
                "succeeded": sum(outcome.succeeded for outcome in outcomes),
                "failed": sum(not outcome.succeeded for outcome in outcomes),
                "score_mean": statistics.fmean(scores) if scores else None,
                "score_min": min(scores) if scores else None,
                "score_max": max(scores) if scores else None,
                "duration_seconds": monotonic() - started_at,
            },
        )
        return outcomes

    def _initial_state(
        self,
        request: SkillGrpoEvolveInput,
        *,
        input_hash: str,
        config_hash: str,
    ) -> EvolutionState:
        version = request.config.algorithm.version
        if request.resume_state is not None:
            validate_resume(
                request.resume_state,
                run_id=request.run_id,
                algorithm_version=version,
                expected_input_fingerprint=input_hash,
                expected_config_fingerprint=config_hash,
                base_skill_hash=request.base_skill.content_hash,
            )
            return request.resume_state.model_copy(deep=True)
        return EvolutionState(
            algorithm_version=version,
            run_id=request.run_id,
            input_fingerprint=input_hash,
            config_fingerprint=config_hash,
            base_skill_hash=request.base_skill.content_hash,
            current_skill=request.base_skill.model_copy(deep=True),
            embedding_model_identity=self._embedding_identity(request.config),
        )

    def _embedding_identity(self, config: SkillGrpoRunConfig) -> str | None:
        if not config.algorithm.replay.use_embeddings or self._embedding_model is None:
            return None
        return config.algorithm.replay.embedding_model_id

    @staticmethod
    def _evolved_skill(base: Skill, content: str, *, run_id: str, tag: str) -> Skill:
        now = datetime.now(UTC)
        blob = {"SKILL.md": content}
        major, minor, patch = (int(value) for value in base.version_label.split("."))
        return base.model_copy(
            update={
                "version_id": f"evolve-run:{run_id}:{tag}",
                "parent_version_ids": [base.version_id],
                "version_label": f"{major}.{minor}.{patch + 1}",
                "content_hash": compute_skill_content_hash(blob),
                "status": SkillVersionStatus.DRAFT,
                "origin": SkillVersionOrigin.EVOLUTION,
                "blob": blob,
                "commit_message": f"{SkillGrpoWithReplayBuffer.algorithm_name} {tag}",
                "created_at": now,
                "updated_at": now,
                "metadata": {
                    **base.metadata,
                    "evolution": {
                        "algorithm": SkillGrpoWithReplayBuffer.algorithm_name,
                        "run_id": run_id,
                        "tag": tag,
                        "unpersisted_candidate": True,
                    },
                },
            },
            deep=True,
        )

    @staticmethod
    def _case_total_scores(outcomes: list[RolloutOutcome]) -> dict[str, float]:
        scores: dict[str, float] = {}
        for outcome in outcomes:
            value = outcome.trajectory.reward.score if outcome.trajectory is not None else None
            scores[outcome.spec.task.task_id] = scores.get(outcome.spec.task.task_id, 0.0) + float(value or 0.0)
        return scores

    @staticmethod
    def _replay_gate_decisions(
        touched: list[TouchedCluster],
        candidates: list[AblationCandidate],
        *,
        min_cluster_edits: int,
    ) -> list[dict[str, Any]]:
        candidates_by_cluster = {candidate.cluster_id: candidate for candidate in candidates}
        decisions: list[dict[str, Any]] = []
        for item in touched:
            cluster = item.cluster
            candidate = candidates_by_cluster.get(cluster.cluster_id)
            if candidate is not None:
                decision = "kept"
                reason = "passed_record_count_gate"
                edit = candidate.edit.model_dump(mode="json")
            elif len(cluster.records) < min_cluster_edits:
                decision = "rejected"
                reason = "min_cluster_edits"
                edit = {
                    "find": item.find_sources[0][0] if item.find_sources else "",
                    "replace": cluster.committed_replace or "",
                }
            elif cluster.committed_replace is None:
                decision = "rejected"
                reason = "missing_committed_replace"
                edit = {
                    "find": item.find_sources[0][0] if item.find_sources else "",
                    "replace": "",
                }
            else:
                decision = "rejected"
                reason = "edit_not_applicable"
                edit = {
                    "find": item.find_sources[0][0] if item.find_sources else "",
                    "replace": cluster.committed_replace,
                }
            decisions.append(
                {
                    "cluster_id": cluster.cluster_id,
                    "decision": decision,
                    "reason": reason,
                    "record_count": len(cluster.records),
                    "min_cluster_edits": min_cluster_edits,
                    "source_task_ids": list(dict.fromkeys(source for _, source in item.find_sources)),
                    "edit": edit,
                }
            )
        return decisions

    @staticmethod
    def _mean_score(outcomes: list[RolloutOutcome]) -> float | None:
        scores = [
            float(outcome.trajectory.reward.score)
            for outcome in outcomes
            if outcome.trajectory is not None and outcome.trajectory.reward.score is not None
        ]
        return statistics.fmean(scores) if scores else None

    @staticmethod
    def _update_metrics(
        metrics: EvolutionMetrics,
        outcomes: list[RolloutOutcome],
        candidates: list[CandidateEvaluationRecord],
        batches: list[BatchEvolutionRecord],
    ) -> None:
        metrics.batches_completed += 1
        metrics.rollouts_completed += sum(outcome.succeeded for outcome in outcomes)
        metrics.rollouts_failed += sum(not outcome.succeeded for outcome in outcomes)
        metrics.candidates_evaluated += len(candidates)
        metrics.edits_applied = sum(len(batch.applied_edits) for batch in batches)
        train_scores = [batch.train_score for batch in batches if batch.train_score is not None]
        metrics.train_score_mean = statistics.fmean(train_scores) if train_scores else None
        validation_scores = [batch.validation_score for batch in batches if batch.validation_score is not None]
        metrics.validation_score_mean = statistics.fmean(validation_scores) if validation_scores else None

    @staticmethod
    def _positive_int(value: Any, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"rollout strategy parameter {name!r} must be a positive integer")
        return value

    @staticmethod
    def _validate_tasks(request: SkillGrpoEvolveInput) -> None:
        train_ids = [task.task_id for task in request.train_tasks]
        if len(train_ids) != len(set(train_ids)):
            raise ValueError("evolution requires unique train task IDs")
        seen: dict[str, Task] = {}
        for task in [*request.train_tasks, *request.validation_tasks, *request.test_tasks]:
            prior = seen.get(task.task_id)
            if prior is not None and prior != task:
                raise ValueError(f"task_id {task.task_id!r} maps to different task payloads")
            seen[task.task_id] = task

    async def _emit(self, run_id: str, name: str, payload: dict[str, Any], *, critical: bool = False) -> None:
        await self._logger.log(
            component_name=name.split("_", 1)[0],
            step_name=name,
            status=self._log_status(name),
            level=LogLevel.ERROR if name.endswith("_failed") else LogLevel.INFO,
            message=self._log_message(name, payload),
            payload={"run_id": run_id, **payload},
        )
        if self._on_event is None:
            return
        try:
            await self._on_event(EvolutionEvent(run_id=run_id, name=name, payload=payload))
        except Exception:
            if critical:
                raise
            return

    @staticmethod
    def _log_status(name: str) -> str | None:
        if name.endswith("_started"):
            return "started"
        if name.endswith("_failed"):
            return "failed"
        if name.endswith(("_completed", "_finished", "_ready")):
            return "succeeded"
        return None

    @staticmethod
    def _log_message(name: str, payload: dict[str, Any]) -> str:
        if name == "run_started":
            return (
                "run started: "
                f"tasks train/val/test={payload['train_tasks']}/{payload['validation_tasks']}/"
                f"{payload['test_tasks']}, epochs={payload['epochs']}, "
                f"batches={payload['completed_batches']}/{payload['total_batches']}, resumed={payload['resumed']}"
            )
        if name == "batch_started":
            return (
                f"batch {payload['batch_number']}/{payload['total_batches']} started "
                f"(epoch={int(payload['epoch']) + 1}, tasks={payload['task_count']})"
            )
        if name == "phase_started":
            return (
                f"phase {payload['phase']} started: scope={payload['scope']}, "
                f"tasks={payload['task_count']}, rollouts={payload['rollout_count']}"
            )
        if name in {"rollout_completed", "rollout_failed"}:
            return (
                f"rollout {payload['rollout_id']} {name.removeprefix('rollout_')}: "
                f"phase={payload['phase']}, task={payload['task_id']}, "
                f"score={format_log_value(payload.get('score'))}, attempts={payload['attempts']}"
            )
        if name == "phase_completed":
            return (
                f"phase {payload['phase']} completed: {payload['succeeded']}/{payload['rollout_count']} succeeded, "
                f"reward mean/min/max={format_log_value(payload['score_mean'])}/"
                f"{format_log_value(payload['score_min'])}/{format_log_value(payload['score_max'])}, "
                f"elapsed={float(payload['duration_seconds']):.1f}s"
            )
        if name == "phase_failed":
            return (
                f"phase {payload['phase']} failed after {float(payload['duration_seconds']):.1f}s: "
                f"{payload['error_type']}: {payload['error']}"
            )
        if name == "stage_started":
            return f"stage {payload['stage']} started"
        if name == "stage_completed":
            return f"stage {payload['stage']} completed"
        if name == "batch_completed":
            return (
                f"batch {int(payload['batch_index']) + 1} completed: "
                f"train_reward={format_log_value(payload['train_score'])}, "
                f"validation_reward={format_log_value(payload['validation_score'])}, "
                f"applied_edits={payload['applied_edits']}, "
                f"skill_chars={payload['skill_chars_before']}->{payload['skill_chars_after']}"
            )
        if name == "checkpoint_ready":
            return "checkpoint ready"
        if name == "run_finished":
            metrics = payload["metrics"]
            return (
                f"run finished: batches={metrics['batches_completed']}, "
                f"rollouts={metrics['rollouts_completed']} ok/{metrics['rollouts_failed']} failed, "
                f"train/val/test reward={format_log_value(metrics['train_score_mean'])}/"
                f"{format_log_value(metrics['validation_score_mean'])}/"
                f"{format_log_value(metrics['test_score_mean'])}, "
                f"elapsed={float(payload['duration_seconds']):.1f}s"
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


__all__ = ["SkillGrpoWithReplayBuffer"]
