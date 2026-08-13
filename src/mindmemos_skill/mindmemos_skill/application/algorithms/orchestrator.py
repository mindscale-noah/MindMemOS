"""Application use cases that connect pure algorithms to durable Skill state."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any

from ...management import LocalSkillManager, PushResult
from ...service import SkillAlgorithms
from ...typing import (
    AlgorithmIdentity,
    AlgorithmLog,
    AlgorithmStep,
    EvolveInput,
    EvolveOutput,
    Skill,
    SkillCandidate,
    Trace2SkillInput,
    Trace2SkillOutput,
    Trajectory,
)
from .models import (
    AlgorithmCommitPolicy,
    EvolveRunRequest,
    SkillAlgorithmRunResult,
    Trace2SkillRunRequest,
)

TrajectoryLoader = Callable[[str], Awaitable[Trajectory]]
TrajectoryRecorder = Callable[[Trajectory], Awaitable[None]]
AlgorithmLogRecorder = Callable[[AlgorithmLog], Awaitable[None]]
VersionPusher = Callable[[str, str | None], Awaitable[PushResult]]


class SkillAlgorithmOrchestrator:
    """Resolve inputs, execute one algorithm and commit its application-owned effects."""

    def __init__(
        self,
        *,
        algorithms: SkillAlgorithms,
        manager: LocalSkillManager,
        load_trajectory: TrajectoryLoader,
        record_trajectory: TrajectoryRecorder,
        record_algorithm_log: AlgorithmLogRecorder,
        push_version: VersionPusher,
        config_hash: str,
        clock: Callable[[], datetime],
        id_generator: Callable[[], str],
    ) -> None:
        self._algorithms = algorithms
        self._manager = manager
        self._load_trajectory = load_trajectory
        self._record_trajectory = record_trajectory
        self._record_algorithm_log = record_algorithm_log
        self._push_version = push_version
        self._config_hash = config_hash
        self._clock = clock
        self._id_generator = id_generator

    async def run_trace2skill(self, request: Trace2SkillRunRequest) -> SkillAlgorithmRunResult:
        base_skill = await self._resolve_base_skill(request.skill_ref, request.base_version_id)
        trajectories = [await self._load_trajectory(trajectory_id) for trajectory_id in request.trajectory_ids]
        algorithm_input = Trace2SkillInput(
            run_id=request.run_id,
            base_skill=base_skill,
            trajectories=trajectories,
            tasks=request.tasks,
        )
        try:
            output = await self._algorithms.optimize(
                algorithm_input,
                algorithm_name=request.algorithm_name,
            )
            return await self._commit_trace2skill(request, base_skill, output)
        except Exception as exc:
            await self._try_record_failure(request, exc)
            raise

    async def run_evolve(self, request: EvolveRunRequest) -> SkillAlgorithmRunResult:
        base_skill = await self._resolve_base_skill(request.skill_ref, request.base_version_id)
        algorithm_input = EvolveInput(
            run_id=request.run_id,
            base_skill=base_skill,
            train_tasks=request.train_tasks,
            validation_tasks=request.validation_tasks,
            test_tasks=request.test_tasks,
        )
        try:
            output = await self._algorithms.evolve(
                algorithm_input,
                algorithm_name=request.algorithm_name,
            )
            candidate = self._candidate_from_evolve(request, output) if output.changed else None
            return await self._commit(
                request=request,
                base_skill=base_skill,
                candidate=candidate,
                input_trajectory_ids=[],
                generated_trajectories=output.trajectories,
                summary=self._evolve_summary(output),
            )
        except Exception as exc:
            await self._try_record_failure(request, exc)
            raise

    async def _commit_trace2skill(
        self,
        request: Trace2SkillRunRequest,
        base_skill: Skill,
        output: Trace2SkillOutput[Any],
    ) -> SkillAlgorithmRunResult:
        return await self._commit(
            request=request,
            base_skill=base_skill,
            candidate=self._decorate_candidate(request, output.candidate),
            input_trajectory_ids=list(request.trajectory_ids),
            generated_trajectories=output.trajectories,
            summary={"report": output.model_dump(mode="json")["report"]},
        )

    async def _commit(
        self,
        *,
        request: Trace2SkillRunRequest | EvolveRunRequest,
        base_skill: Skill,
        candidate: SkillCandidate | None,
        input_trajectory_ids: list[str],
        generated_trajectories: list[Trajectory],
        summary: dict[str, Any],
    ) -> SkillAlgorithmRunResult:
        generated_trajectory_ids = [trajectory.trajectory_id for trajectory in generated_trajectories]
        if request.commit_policy is AlgorithmCommitPolicy.DRY_RUN:
            return SkillAlgorithmRunResult(
                run_id=request.run_id,
                algorithm_name=request.algorithm_name,
                base_version_id=base_skill.version_id,
                changed=candidate is not None,
                candidate=candidate,
                input_trajectory_ids=input_trajectory_ids,
                generated_trajectory_ids=generated_trajectory_ids,
            )

        persisted_trajectory_ids = await self._persist_trajectories(request, generated_trajectories)
        persisted = None
        if candidate is not None:
            persisted = await self._manager.persist_algorithm_candidate(
                candidate,
                base_version_id=base_skill.version_id,
            )

        push_operation_id = None
        if persisted is not None and request.commit_policy is AlgorithmCommitPolicy.PERSIST_AND_PUSH:
            pushed = await self._push_version(persisted.skill_id, persisted.version_id)
            push_operation_id = pushed.operation_id

        log = self._success_log(
            request=request,
            base_skill=base_skill,
            persisted_version_id=persisted.version_id if persisted is not None else None,
            input_trajectory_ids=input_trajectory_ids,
            generated_trajectory_ids=generated_trajectory_ids,
            persisted_trajectory_ids=persisted_trajectory_ids,
            summary=summary,
        )
        await self._record_algorithm_log(log)
        return SkillAlgorithmRunResult(
            run_id=request.run_id,
            algorithm_name=request.algorithm_name,
            base_version_id=base_skill.version_id,
            changed=candidate is not None,
            candidate=candidate,
            persisted_version_id=persisted.version_id if persisted is not None else None,
            input_trajectory_ids=input_trajectory_ids,
            generated_trajectory_ids=generated_trajectory_ids,
            persisted_trajectory_ids=persisted_trajectory_ids,
            algorithm_log_ids=[log.log_id],
            push_operation_id=push_operation_id,
        )

    async def _persist_trajectories(
        self,
        request: Trace2SkillRunRequest | EvolveRunRequest,
        trajectories: list[Trajectory],
    ) -> list[str]:
        persisted_ids: list[str] = []
        seen: set[str] = set()
        for trajectory in trajectories:
            if trajectory.trajectory_id in seen:
                continue
            seen.add(trajectory.trajectory_id)
            decorated = trajectory.model_copy(
                update={
                    "metadata": {
                        **trajectory.metadata,
                        "algorithm_run_id": request.run_id,
                        "algorithm_name": request.algorithm_name,
                    }
                }
            )
            await self._record_trajectory(decorated)
            persisted_ids.append(decorated.trajectory_id)
        return persisted_ids

    async def _resolve_base_skill(self, skill_ref: str, version_id: str | None) -> Skill:
        detail = await self._manager.get_skill(skill_ref)
        resolved_version_id = version_id or detail.latest_version.version_id
        return Skill.from_record(await self._manager.get_version(detail.skill.skill_id, resolved_version_id))

    def _candidate_from_evolve(self, request: EvolveRunRequest, output: EvolveOutput) -> SkillCandidate:
        return SkillCandidate(
            blob=output.final_skill.blob,
            resources=output.final_skill.resources,
            runtime_type=output.final_skill.runtime_type,
            runtime_schema_version=output.final_skill.runtime_schema_version,
            runtime_metadata=output.final_skill.runtime_metadata,
            commit_message=output.final_skill.commit_message or f"evolve: {request.algorithm_name}",
            metadata={
                **output.final_skill.metadata,
                "skill_application": {"config_hash": self._config_hash},
                "algorithm_run": {
                    "run_id": request.run_id,
                    "algorithm_name": request.algorithm_name,
                },
            },
        )

    def _decorate_candidate(
        self,
        request: Trace2SkillRunRequest,
        candidate: SkillCandidate | None,
    ) -> SkillCandidate | None:
        if candidate is None:
            return None
        return candidate.model_copy(
            update={
                "metadata": {
                    **candidate.metadata,
                    "skill_application": {"config_hash": self._config_hash},
                    "algorithm_run": {
                        "run_id": request.run_id,
                        "algorithm_name": request.algorithm_name,
                    },
                }
            }
        )

    @staticmethod
    def _evolve_summary(output: EvolveOutput) -> dict[str, Any]:
        metrics = getattr(output, "metrics", None)
        return {
            "finished_at": output.finished_at.isoformat(),
            "metrics": metrics.model_dump(mode="json") if hasattr(metrics, "model_dump") else None,
        }

    def _success_log(
        self,
        *,
        request: Trace2SkillRunRequest | EvolveRunRequest,
        base_skill: Skill,
        persisted_version_id: str | None,
        input_trajectory_ids: list[str],
        generated_trajectory_ids: list[str],
        persisted_trajectory_ids: list[str],
        summary: dict[str, Any],
    ) -> AlgorithmLog:
        return AlgorithmLog(
            log_id=self._id_generator(),
            algorithm=AlgorithmIdentity(name=request.algorithm_name),
            step=AlgorithmStep(
                component_name=request.algorithm_name,
                name="run",
                status="succeeded",
                payload={
                    "run_id": request.run_id,
                    "base_version_id": base_skill.version_id,
                    "persisted_version_id": persisted_version_id,
                    "input_trajectory_ids": input_trajectory_ids,
                    "generated_trajectory_ids": generated_trajectory_ids,
                    "persisted_trajectory_ids": persisted_trajectory_ids,
                    "summary": summary,
                },
                created_at=self._clock(),
            ),
        )

    async def _try_record_failure(
        self,
        request: Trace2SkillRunRequest | EvolveRunRequest,
        error: Exception,
    ) -> None:
        if request.commit_policy is AlgorithmCommitPolicy.DRY_RUN:
            return
        try:
            await self._record_algorithm_log(
                AlgorithmLog(
                    log_id=self._id_generator(),
                    algorithm=AlgorithmIdentity(name=request.algorithm_name),
                    step=AlgorithmStep(
                        component_name=request.algorithm_name,
                        name="run",
                        status="failed",
                        payload={
                            "run_id": request.run_id,
                            "error_type": type(error).__name__,
                            "error": str(error),
                        },
                        created_at=self._clock(),
                    ),
                )
            )
        except Exception:
            return


__all__ = ["SkillAlgorithmOrchestrator"]
