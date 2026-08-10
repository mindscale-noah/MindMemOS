"""Cloud Skill v2 API orchestration over the relational repository."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from mindmemos_skill.contracts import SkillBundle, SkillRemoteOperationType

from ...errors import (
    BadRequestError,
    ResourceNotFoundError,
    SkillConflictError,
    SkillNotFoundError,
    SkillVersionNotFoundError,
)
from ...infra.db import SkillRelationalRepository, get_database_clients
from ...infra.kafka import get_producer
from ...logging import get_logger, traced
from ..deps import annotate_request_trace
from ..schemas import AuthContext
from ..skill_schemas import (
    SkillContentData,
    SkillEvolveData,
    SkillEvolveRequest,
    SkillListData,
    SkillRegisterData,
    SkillRegisterRequest,
    SkillRemoteSyncData,
    SkillRemoteSyncRequest,
    SkillRemoteSyncResultItem,
    SkillSummaryData,
    SkillTrajectoryPageData,
    SkillTrajectoryReportData,
    SkillTrajectoryReportRequest,
    SkillVersionsData,
    SkillVersionStatusRequest,
)

SKILL_TRAJECTORY_INGEST_TOPIC = "skill.trajectory.ingest"
SKILL_EVOLUTION_TOPIC = "skill.evolve"
logger = get_logger(__name__)


class SkillService:
    """Project-scoped facade for version, operation and trajectory facts."""

    def __init__(
        self,
        *,
        repository: SkillRelationalRepository | None = None,
        evolver: Any | None = None,
        producer: Any | None = None,
    ) -> None:
        self._repository = repository
        self._evolver = evolver
        self._producer = producer

    @property
    def repository(self) -> SkillRelationalRepository:
        return self._repository or get_database_clients().skill

    @traced("skill_service.register")
    async def register(self, auth: AuthContext, request: SkillRegisterRequest) -> SkillRegisterData:
        annotate_request_trace(auth)
        try:
            version = await self.repository.create_version(
                project_id=auth.project_id,
                operation_id=request.operation_id,
                version=request.version,
                bundle=request.bundle,
            )
        except SkillVersionNotFoundError as exc:
            raise ResourceNotFoundError(str(exc), code="skill.version_not_found") from exc
        except (SkillConflictError, ValueError) as exc:
            raise BadRequestError(str(exc), code="skill.version_conflict", status_code=409) from exc
        return SkillRegisterData(version=version)

    @traced("skill_service.list")
    async def list_skills(self, auth: AuthContext) -> SkillListData:
        annotate_request_trace(auth)
        versions = await self.repository.list_latest_available_versions(auth.project_id)
        return SkillListData(
            skills=[
                SkillSummaryData(
                    cloud_skill_id=version.cloud_skill_id or "",
                    name=version.name,
                    latest_version=version,
                )
                for version in versions
            ]
        )

    @traced("skill_service.get")
    async def get_skill(self, auth: AuthContext, cloud_skill_id: str) -> SkillSummaryData:
        annotate_request_trace(auth)
        try:
            version = await self.repository.latest_available_version(auth.project_id, cloud_skill_id)
        except SkillNotFoundError as exc:
            raise ResourceNotFoundError(str(exc), code="skill.not_found") from exc
        return SkillSummaryData(cloud_skill_id=cloud_skill_id, name=version.name, latest_version=version)

    @traced("skill_service.versions")
    async def versions(self, auth: AuthContext, cloud_skill_id: str) -> SkillVersionsData:
        annotate_request_trace(auth)
        try:
            versions = await self.repository.list_versions(auth.project_id, cloud_skill_id)
        except SkillNotFoundError as exc:
            raise ResourceNotFoundError(str(exc), code="skill.not_found") from exc
        return SkillVersionsData(versions=versions)

    @traced("skill_service.content")
    async def content(self, auth: AuthContext, cloud_skill_id: str, version_id: str) -> SkillContentData:
        annotate_request_trace(auth)
        try:
            version = await self.repository.get_version(auth.project_id, version_id)
            if version.cloud_skill_id != cloud_skill_id:
                raise SkillVersionNotFoundError(f"version {version_id} does not belong to {cloud_skill_id}")
            bundle = await self.repository.get_bundle(auth.project_id, version_id)
        except SkillVersionNotFoundError as exc:
            raise ResourceNotFoundError(str(exc), code="skill.version_not_found") from exc
        return SkillContentData(version=version, bundle=bundle)

    @traced("skill_service.sync")
    async def sync_remote(self, auth: AuthContext, request: SkillRemoteSyncRequest) -> SkillRemoteSyncData:
        annotate_request_trace(auth)
        items: list[SkillRemoteSyncResultItem] = []
        try:
            for item in request.items:
                versions = await self.repository.sync_versions(
                    project_id=auth.project_id,
                    cloud_skill_id=item.cloud_skill_id,
                    known_version_revisions=item.known_version_revisions,
                )
                items.append(SkillRemoteSyncResultItem(cloud_skill_id=item.cloud_skill_id, versions=versions))
        except SkillNotFoundError as exc:
            raise ResourceNotFoundError(str(exc), code="skill.not_found") from exc
        return SkillRemoteSyncData(items=items)

    @traced("skill_service.status")
    async def update_status(
        self,
        auth: AuthContext,
        version_id: str,
        request: SkillVersionStatusRequest,
    ) -> SkillRegisterData:
        annotate_request_trace(auth)
        try:
            version = await self.repository.update_status(
                project_id=auth.project_id,
                version_id=version_id,
                status=request.status,
                expected_revision=request.expected_revision,
            )
        except SkillVersionNotFoundError as exc:
            raise ResourceNotFoundError(str(exc), code="skill.version_not_found") from exc
        except SkillConflictError as exc:
            raise BadRequestError(str(exc), code="skill.revision_conflict", status_code=409) from exc
        return SkillRegisterData(version=version)

    @traced("skill_service.evolve")
    async def evolve(self, auth: AuthContext, request: SkillEvolveRequest) -> SkillEvolveData:
        annotate_request_trace(auth)
        try:
            run_id, base, bundle, evidence, replay = await self.repository.prepare_evolution(
                project_id=auth.project_id,
                operation_id=request.operation_id,
                cloud_skill_id=request.cloud_skill_id,
                base_version_id=request.base_version_id,
                algorithm=request.algorithm,
                mode=request.mode,
                reuse_evidence=request.reuse_evidence,
                trajectory_ids=request.trajectory_ids,
            )
        except SkillVersionNotFoundError as exc:
            raise ResourceNotFoundError(str(exc), code="skill.version_not_found") from exc
        except (SkillConflictError, ValueError) as exc:
            raise BadRequestError(str(exc), code="skill.evolution_conflict", status_code=409) from exc
        if replay is not None:
            return _evolve_data(replay)
        if request.mode == "async":
            if self._producer is not None:
                try:
                    await self._producer.send(
                        SKILL_EVOLUTION_TOPIC,
                        {"project_id": auth.project_id, "operation_id": request.operation_id},
                        dispatch_key=request.cloud_skill_id,
                        wait=True,
                    )
                except Exception as exc:
                    logger.warning(
                        "queued Skill evolution Kafka dispatch failed",
                        project_id=auth.project_id,
                        operation_id=request.operation_id,
                        error=str(exc),
                    )
            return SkillEvolveData(
                operation_id=request.operation_id,
                evolution_run_id=run_id,
                cloud_skill_id=request.cloud_skill_id,
                base_version_id=request.base_version_id,
                status="queued",
            )
        if not evidence:
            result = _evolve_result(
                request=request,
                run_id=run_id,
                status="no_change",
                candidate_version_ids=[],
                selected_version_id=None,
            )
            await self.repository.finish_evolution_without_version(
                project_id=auth.project_id,
                operation_id=request.operation_id,
                result=result,
            )
            return _evolve_data(result)
        if self._evolver is None:
            result = _evolve_result(
                request=request,
                run_id=run_id,
                status="failed",
                candidate_version_ids=[],
                selected_version_id=None,
            )
            await self.repository.finish_evolution_without_version(
                project_id=auth.project_id,
                operation_id=request.operation_id,
                result=result,
                failed=True,
                error_code="skill.evolver_unavailable",
            )
            return _evolve_data(result)
        try:
            output = await self._evolver.evolve(
                base_version=base,
                base_bundle=bundle,
                trajectories=evidence,
                algorithm=request.algorithm,
            )
            raw_candidates = output.get("candidates", []) if isinstance(output, dict) else output
            selected_index = int(output.get("selected_index", 0)) if isinstance(output, dict) else 0
            candidates = [
                item if isinstance(item, SkillBundle) else SkillBundle.model_validate(item)
                for item in raw_candidates
            ]
            versions = await self.repository.complete_evolution_candidates(
                project_id=auth.project_id,
                operation_id=request.operation_id,
                evolution_run_id=run_id,
                base=base,
                algorithm=request.algorithm,
                evidence=evidence,
                candidates=candidates,
                selected_index=selected_index,
            )
        except Exception as exc:
            result = _evolve_result(
                request=request,
                run_id=run_id,
                status="failed",
                candidate_version_ids=[],
                selected_version_id=None,
            )
            await self.repository.finish_evolution_without_version(
                project_id=auth.project_id,
                operation_id=request.operation_id,
                result=result,
                failed=True,
                error_code=type(exc).__name__,
            )
            return _evolve_data(result)
        status = "succeeded" if versions else "no_change"
        result = _evolve_result(
            request=request,
            run_id=run_id,
            status=status,
            candidate_version_ids=[item.version_id for item in versions],
            selected_version_id=versions[selected_index].version_id if versions else None,
        )
        return _evolve_data(result)

    async def resume_evolution(self, *, project_id: str, operation_id: str) -> SkillEvolveData:
        """Resume one durable async evolution from its frozen operation evidence."""

        operation = await self.repository.get_operation_payload(project_id, operation_id)
        payload = operation.get("request_payload") or {}
        if operation.get("operation_type") != SkillRemoteOperationType.EVOLVE.value:
            raise SkillConflictError(f"operation is not an evolution request: {operation_id}")
        replay = operation.get("result") or {}
        if operation.get("status") in {"succeeded", "no_change", "failed", "cancelled"}:
            return _evolve_data(replay)
        request = SkillEvolveRequest(
            operation_id=operation_id,
            cloud_skill_id=str(payload["cloud_skill_id"]),
            base_version_id=str(payload["base_version_id"]),
            algorithm=str(payload["algorithm"]),
            mode="sync",
            reuse_evidence=bool(payload.get("reuse_evidence", False)),
            trajectory_ids=list(payload.get("selected_trajectory_ids") or []),
        )
        base = await self.repository.get_version(project_id, request.base_version_id)
        bundle = await self.repository.get_bundle(project_id, request.base_version_id)
        evidence = [
            await self.repository.get_trajectory(project_id, trajectory_id)
            for trajectory_id in request.trajectory_ids or []
        ]
        run_id = str(payload["evolution_run_id"])
        if not evidence:
            result = _evolve_result(
                request=request,
                run_id=run_id,
                status="no_change",
                candidate_version_ids=[],
                selected_version_id=None,
            )
            await self.repository.finish_evolution_without_version(
                project_id=project_id,
                operation_id=operation_id,
                result=result,
            )
            return _evolve_data(result)
        if self._evolver is None:
            result = _evolve_result(
                request=request,
                run_id=run_id,
                status="failed",
                candidate_version_ids=[],
                selected_version_id=None,
            )
            await self.repository.finish_evolution_without_version(
                project_id=project_id,
                operation_id=operation_id,
                result=result,
                failed=True,
                error_code="skill.evolver_unavailable",
            )
            return _evolve_data(result)
        try:
            output = await self._evolver.evolve(
                base_version=base,
                base_bundle=bundle,
                trajectories=evidence,
                algorithm=request.algorithm,
            )
            raw_candidates = output.get("candidates", []) if isinstance(output, dict) else output
            selected_index = int(output.get("selected_index", 0)) if isinstance(output, dict) else 0
            candidates = [
                item if isinstance(item, SkillBundle) else SkillBundle.model_validate(item)
                for item in raw_candidates
            ]
            versions = await self.repository.complete_evolution_candidates(
                project_id=project_id,
                operation_id=operation_id,
                evolution_run_id=run_id,
                base=base,
                algorithm=request.algorithm,
                evidence=evidence,
                candidates=candidates,
                selected_index=selected_index,
            )
        except Exception as exc:
            result = _evolve_result(
                request=request,
                run_id=run_id,
                status="failed",
                candidate_version_ids=[],
                selected_version_id=None,
            )
            await self.repository.finish_evolution_without_version(
                project_id=project_id,
                operation_id=operation_id,
                result=result,
                failed=True,
                error_code=type(exc).__name__,
            )
            return _evolve_data(result)
        return _evolve_data(
            _evolve_result(
                request=request,
                run_id=run_id,
                status="succeeded" if versions else "no_change",
                candidate_version_ids=[version.version_id for version in versions],
                selected_version_id=versions[selected_index].version_id if versions else None,
            )
        )

    async def redispatch_queued_operations(self, *, project_id: str, limit: int = 100) -> int:
        """Operation-sweeper entry point for Kafka sends that failed after durable enqueue."""

        if self._producer is None:
            return 0
        dispatched = 0
        for operation in await self.repository.list_queued_operations(project_id, limit=limit):
            operation_type = operation["operation_type"]
            if operation_type == SkillRemoteOperationType.REPORT_TRAJECTORY.value:
                topic = SKILL_TRAJECTORY_INGEST_TOPIC
                dispatch_key = str(operation["operation_id"])
            elif operation_type == SkillRemoteOperationType.EVOLVE.value:
                topic = SKILL_EVOLUTION_TOPIC
                dispatch_key = str(operation.get("cloud_skill_id") or operation["operation_id"])
            else:
                continue
            await self._producer.send(
                topic,
                {"project_id": project_id, "operation_id": operation["operation_id"]},
                dispatch_key=dispatch_key,
                wait=True,
            )
            dispatched += 1
        return dispatched

    @traced("skill_service.trajectory_report")
    async def report_trajectories(
        self,
        auth: AuthContext,
        request: SkillTrajectoryReportRequest,
    ) -> SkillTrajectoryReportData:
        annotate_request_trace(auth)
        try:
            if request.mode == "async":
                result = await self.repository.enqueue_trajectories(project_id=auth.project_id, request=request)
                if self._producer is not None:
                    try:
                        await self._producer.send(
                            SKILL_TRAJECTORY_INGEST_TOPIC,
                            {
                                "project_id": auth.project_id,
                                "operation_id": request.operation_id,
                            },
                            dispatch_key=request.operation_id,
                            wait=True,
                        )
                    except Exception as exc:
                        # The operation is already durable. A sweeper can redispatch it.
                        logger.warning(
                            "queued Skill trajectory Kafka dispatch failed",
                            project_id=auth.project_id,
                            operation_id=request.operation_id,
                            error=str(exc),
                        )
                return result
            return await self.repository.ingest_trajectories(project_id=auth.project_id, request=request)
        except SkillVersionNotFoundError as exc:
            raise ResourceNotFoundError(str(exc), code="skill.version_not_found") from exc
        except SkillConflictError as exc:
            raise BadRequestError(str(exc), code="skill.trajectory_conflict", status_code=409) from exc
        except ValueError as exc:
            raise BadRequestError(str(exc), code="skill.trajectory_invalid", status_code=422) from exc

    @traced("skill_service.trajectory_list")
    async def list_trajectories(
        self,
        auth: AuthContext,
        *,
        cloud_skill_id: str,
        version_id: str | None,
        since: datetime | None,
        cursor: str | None,
        limit: int,
        status: str | None,
        min_score: float | None,
    ) -> SkillTrajectoryPageData:
        annotate_request_trace(auth)
        try:
            items, next_cursor, has_more = await self.repository.list_trajectories(
                project_id=auth.project_id,
                cloud_skill_id=cloud_skill_id,
                version_id=version_id,
                since=since,
                cursor=cursor,
                limit=limit,
                status=status,
                min_score=min_score,
            )
        except SkillVersionNotFoundError as exc:
            raise ResourceNotFoundError(str(exc), code="skill.version_not_found") from exc
        except (SkillConflictError, ValueError) as exc:
            raise BadRequestError(str(exc), code="skill.trajectory_invalid") from exc
        return SkillTrajectoryPageData(
            items=items,
            returned_count=len(items),
            next_cursor=next_cursor,
            has_more=has_more,
        )

    @traced("skill_service.trajectory_get")
    async def get_trajectory(self, auth: AuthContext, trajectory_id: str):
        annotate_request_trace(auth)
        try:
            return await self.repository.get_trajectory(auth.project_id, trajectory_id)
        except SkillNotFoundError as exc:
            raise ResourceNotFoundError(str(exc), code="skill.trajectory_not_found") from exc


_service: SkillService | None = None


def get_skill_service() -> SkillService:
    global _service
    if _service is None:
        _service = SkillService(producer=get_producer())
    return _service


def _evolve_result(
    *,
    request: SkillEvolveRequest,
    run_id: str,
    status: str,
    candidate_version_ids: list[str],
    selected_version_id: str | None,
) -> dict[str, Any]:
    return {
        "operation_id": request.operation_id,
        "evolution_run_id": run_id,
        "cloud_skill_id": request.cloud_skill_id,
        "base_version_id": request.base_version_id,
        "status": status,
        "candidate_version_ids": candidate_version_ids,
        "selected_version_id": selected_version_id,
    }


def _evolve_data(value: dict[str, Any]) -> SkillEvolveData:
    return SkillEvolveData.model_validate(
        {key: value.get(key) for key in SkillEvolveData.model_fields}
    )
