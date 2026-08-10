"""Unified lifecycle root for one local Skill application."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from functools import wraps
from typing import Any, Concatenate, ParamSpec, Self, TypeVar

from pydantic import BaseModel

from ..config import CompiledSkillApplicationConfig, SkillConfigCompiler, SkillConfigSource
from ..contracts import (
    SkillBundle,
    SkillTrajectoryBinding,
    SkillTrajectoryReportRequest,
    SkillTrajectoryUpload,
    SkillTrajectoryUploadItem,
    SkillVersionCore,
    canonical_request_hash,
    compute_trajectory_hash,
)
from ..errors import (
    SkillCapabilityUnavailableError,
    SkillConflictError,
    SkillNotFoundError,
    SkillRemoteOperationError,
    SkillRemoteRequestError,
    SkillServiceClosedError,
)
from ..infra.database import (
    DatabaseConfig,
    DatabaseRegistry,
    DatabaseScope,
    FilterGroup,
    Page,
    Predicate,
    RecordQuery,
    ScopedDatabase,
    bootstrap_database,
)
from ..management import (
    ExportSkillRequest,
    ExportSkillResult,
    LocalSkillManager,
    ManagedSkill,
    PendingSkillOperation,
    PendingSkillOperationStatus,
    PendingSkillOperationType,
    PublishSkillRequest,
    PublishSkillResult,
    PullResult,
    PushResult,
    RegisterSkillRequest,
    RegisterSkillResult,
    ResolvedSkillContext,
    SkillDetail,
    SkillDiffResult,
    SkillManagementDetail,
    SkillManagementOverview,
    SkillManagementSummary,
    SkillManagementSyncState,
    SkillRepository,
    frontmatter_value,
    parse_version_label,
    resolve_detected_contexts,
    serialize_files,
    snapshot_from_cloud_bundle,
    snapshot_from_record,
    snapshot_metadata,
)
from ..persistence import (
    ALGORITHM_LOG_TABLE,
    SKILL_REMOTE_OPERATION_TABLE,
    TRAJECTORY_TABLE,
    AlgorithmLogRecord,
    SkillRecord,
    SkillRemoteOperationRecord,
    TrajectoryRecord,
    build_persistence_tables,
    from_database_record,
    to_database_record,
)
from ..remote import (
    RemotePushRequest,
    RemotePushResult,
    RemoteSyncItem,
    RemoteSyncRequest,
    RemoteTrajectoryListRequest,
    RemoteTrajectoryPage,
    RemoteVersionSummary,
    SkillRemotePort,
)
from ..typing import (
    AgentExecutionRequest,
    AlgorithmLog,
    SkillAnalysisRequest,
    SkillAnalysisResult,
    SkillOptimizationRequest,
    SkillOptimizationResult,
    Trajectory,
)
from .components import RuntimeComponents, compose_runtime
from .enums import AlgorithmResultStatus, SkillApplicationCapability

_MANAGEMENT_CAPABILITIES = frozenset(
    capability.value
    for capability in (
        SkillApplicationCapability.DIFF,
        SkillApplicationCapability.EXPORT,
        SkillApplicationCapability.LIST,
        SkillApplicationCapability.PUBLISH,
        SkillApplicationCapability.REGISTER,
        SkillApplicationCapability.SHOW,
        SkillApplicationCapability.UNREGISTER,
    )
)

_Parameters = ParamSpec("_Parameters")
_ReturnT = TypeVar("_ReturnT")


def requires_ready(
    method: Callable[Concatenate[SkillApplication, _Parameters], Awaitable[_ReturnT]],
) -> Callable[Concatenate[SkillApplication, _Parameters], Awaitable[_ReturnT]]:
    """Require a ready application before invoking an async business method."""

    @wraps(method)
    async def wrapper(
        self: SkillApplication,
        *args: _Parameters.args,
        **kwargs: _Parameters.kwargs,
    ) -> _ReturnT:
        self._ensure_ready()
        return await method(self, *args, **kwargs)

    return wrapper


class SkillApplication:
    """Own local Skill state and optional algorithm resources on one event loop."""

    def __init__(
        self,
        *,
        config: CompiledSkillApplicationConfig,
        database: ScopedDatabase,
        manager: LocalSkillManager,
        runtime: RuntimeComponents,
        remote: SkillRemotePort | None,
        clock: Callable[[], datetime],
        id_generator: Callable[[], str],
    ) -> None:
        self._config = config
        self._database = database
        self._manager = manager
        self._runtime = runtime
        self._remote = remote
        self._clock = clock
        self._id_generator = id_generator
        self._owner_loop: asyncio.AbstractEventLoop | None = None
        self._started = False
        self._closed = False
        self._lifecycle_lock = asyncio.Lock()

    @classmethod
    async def from_config(
        cls,
        config: SkillConfigSource | CompiledSkillApplicationConfig,
        *,
        database_registry: DatabaseRegistry | None = None,
        remote: SkillRemotePort | None = None,
        clock: Callable[[], datetime] | None = None,
        id_generator: Callable[[], str] | None = None,
    ) -> Self:
        """Compile configuration and create a ready-to-use application."""

        compiled = (
            config
            if isinstance(config, CompiledSkillApplicationConfig)
            else SkillConfigCompiler(database_registry=database_registry).compile(config)
        )
        database_config = DatabaseConfig(
            provider=compiled.local.database.provider,
            options=compiled.local.database.options,
            required=compiled.local.database.required,
        )
        database = await bootstrap_database(
            database_config,
            build_persistence_tables(),
            registry=database_registry,
        )
        resolved_clock = clock or (lambda: datetime.now(UTC))
        resolved_id_generator = id_generator or (lambda: str(uuid.uuid4()))
        try:
            runtime = compose_runtime(compiled)
            manager = LocalSkillManager(
                SkillRepository(database),
                managed_root=compiled.local.root_dir,
                clock=resolved_clock,
                id_generator=resolved_id_generator,
            )
        except BaseException:
            await database.close()
            raise
        application = cls(
            config=compiled,
            database=database,
            manager=manager,
            runtime=runtime,
            remote=remote,
            clock=resolved_clock,
            id_generator=resolved_id_generator,
        )
        try:
            await application.start()
        except BaseException:
            try:
                await application.close()
            except BaseException:
                pass
            raise
        return application

    @property
    def config(self) -> CompiledSkillApplicationConfig:
        return self._config

    @property
    def capabilities(self) -> frozenset[str]:
        capabilities = set(_MANAGEMENT_CAPABILITIES)
        if self._runtime.agents:
            capabilities.add(SkillApplicationCapability.EXECUTE.value)
        if self._runtime.skill_algorithms is not None:
            capabilities.update(self._runtime.skill_algorithms.capabilities)
        if self._remote is not None:
            capabilities.add(SkillApplicationCapability.PULL.value)
            capabilities.add(SkillApplicationCapability.PUSH.value)
            capabilities.add(SkillApplicationCapability.SYNC.value)
        return frozenset(capabilities)

    async def start(self) -> None:
        async with self._lifecycle_lock:
            loop = asyncio.get_running_loop()
            if self._closed:
                raise SkillServiceClosedError("SkillApplication is closed")
            if self._owner_loop is not None and loop is not self._owner_loop:
                raise RuntimeError("SkillApplication cannot be used across event loops")
            if self._started:
                return
            self._owner_loop = loop
            await self._runtime.start()
            self._started = True

    async def close(self) -> None:
        async with self._lifecycle_lock:
            if self._closed:
                return
            self._ensure_owner_loop()
            self._closed = True
            error: BaseException | None = None
            try:
                await self._runtime.close()
            except BaseException as exc:
                error = exc
            try:
                await self._database.close()
            except BaseException as exc:
                if error is None:
                    error = exc
            self._started = False
            if error is not None:
                raise error

    @requires_ready
    async def register(self, request: RegisterSkillRequest) -> RegisterSkillResult:
        return await self._manager.register(request)

    @requires_ready
    async def publish(self, request: PublishSkillRequest) -> PublishSkillResult:
        return await self._manager.publish(request)

    @requires_ready
    async def list_skills(self) -> list[ManagedSkill]:
        return await self._manager.list_skills()

    @requires_ready
    async def unregister(self, skill_ref: str) -> ManagedSkill:
        return await self._manager.unregister(skill_ref)

    @requires_ready
    async def get_management_overview(self) -> SkillManagementOverview:
        """Return list rows and the durable outbox through one Application call."""

        skills = await self._manager.list_skills()
        operations: list[PendingSkillOperation] = []
        summaries: list[SkillManagementSummary] = []
        for skill in skills:
            family_operations = [
                PendingSkillOperation.from_record(record)
                for record in await self._manager.repository.list_operations(skill_id=skill.skill_id)
                if record.status in {"pending", "running", "failed"}
            ]
            operations.extend(family_operations)
            summaries.append(
                SkillManagementSummary(
                    skill=skill,
                    sync_state=_management_sync_state(
                        skill,
                        family_operations,
                        remote_configured=self._remote is not None,
                    ),
                )
            )
        return SkillManagementOverview(skills=summaries, pending_operations=operations)

    @requires_ready
    async def resolve_skill_context(
        self,
        messages: list[BaseModel | dict[str, Any]],
        *,
        ensure_remote: bool = True,
    ) -> list[ResolvedSkillContext]:
        """Resolve Memory Add context through the same local Application state."""

        skills = await self._manager.list_skills()
        contexts = resolve_detected_contexts(messages, skills=skills)
        if ensure_remote and self._remote is not None:
            by_latest_version = {skill.latest_version_id: skill.skill_id for skill in skills}
            for context in contexts:
                skill_id = by_latest_version[context.base_version_id]
                await self.push(skill_id, context.base_version_id)
        return contexts

    @requires_ready
    async def get_skill(self, skill_ref: str) -> SkillDetail:
        return await self._manager.get_skill(skill_ref)

    @requires_ready
    async def get_management_detail(self, skill_ref: str) -> SkillManagementDetail:
        """Return versions, derived latest version and durable outbox evidence."""

        detail = await self._manager.get_skill(skill_ref)
        versions = await self._manager.list_versions(detail.skill.skill_id)
        operations = [
            PendingSkillOperation.from_record(record)
            for record in await self._manager.repository.list_operations(skill_id=detail.skill.skill_id)
            if record.status in {"pending", "running", "failed"}
        ]
        return SkillManagementDetail(
            skill=detail.skill,
            versions=versions,
            latest_version=detail.latest_version,
            pending_operations=operations,
            sync_state=_management_sync_state(
                detail.skill,
                operations,
                remote_configured=self._remote is not None,
            ),
        )

    @requires_ready
    async def list_versions(self, skill_ref: str) -> list[SkillRecord]:
        return await self._manager.list_versions(skill_ref)

    @requires_ready
    async def get_version(self, skill_ref: str, version_id: str) -> SkillRecord:
        return await self._manager.get_version(skill_ref, version_id)

    @requires_ready
    async def export(self, request: ExportSkillRequest) -> ExportSkillResult:
        return await self._manager.export(request)

    @requires_ready
    async def diff(
        self,
        skill_ref: str,
        *,
        to_version_id: str,
        from_version_id: str | None = None,
    ) -> SkillDiffResult:
        return await self._manager.diff(
            skill_ref,
            to_version_id=to_version_id,
            from_version_id=from_version_id,
        )

    @requires_ready
    async def push(self, skill_ref: str, version_id: str | None = None) -> PushResult:
        """Idempotently upload one immutable version without changing local pointers."""

        remote = self._remote
        if remote is None:
            raise SkillCapabilityUnavailableError("Skill remote push is not configured")
        version, operation, cloud_skill_id = await self._manager.repository.begin_push(
            skill_ref,
            version_id=version_id,
            now=self._clock(),
        )
        try:
            bundle = SkillBundle.model_validate_json(version.bundle)
        except Exception as exc:
            await self._mark_push_failed(version.skill_id, operation.operation_id, "invalid_remote_content")
            raise SkillRemoteOperationError(operation.operation_id, "invalid_remote_content") from exc
        request = RemotePushRequest(
            operation_id=operation.operation_id,
            cloud_skill_id=cloud_skill_id,
            version=SkillVersionCore(
                version_id=version.version_id,
                cloud_skill_id=cloud_skill_id,
                parent_version_ids=version.parent_version_ids,
                name=version.name,
                content_hash=bundle.content_hash,
                version_label=version.version_label,
                commit_message=version.commit_message,
                status=version.status,
                version_revision=version.version_revision,
                origin=version.origin,
                metadata={key: value for key, value in version.metadata.items() if key != "cloud"},
                created_at=version.created_at,
                updated_at=version.updated_at,
            ),
            bundle=bundle,
        )
        try:
            result = await remote.push_version(request)
        except SkillRemoteRequestError as exc:
            await self._mark_push_failed(version.skill_id, operation.operation_id, exc.error_code)
            raise SkillRemoteOperationError(
                operation.operation_id,
                exc.error_code,
                retryable=exc.retryable,
                status_code=exc.status_code,
                request_id=exc.request_id,
            ) from exc
        mismatch = self._push_response_mismatch(request, result)
        if mismatch is not None:
            await self._mark_push_failed(version.skill_id, operation.operation_id, mismatch)
            raise SkillRemoteOperationError(operation.operation_id, mismatch)
        try:
            await self._manager.repository.complete_push(
                version.skill_id,
                version_id=version.version_id,
                operation_id=operation.operation_id,
                cloud_skill_id=result.cloud_skill_id,
                remote_content_hash=result.content_hash,
                remote_status=result.version.status,
                now=self._clock(),
                version_revision=result.version.version_revision,
                received_at=result.version.received_at,
            )
        except Exception as exc:
            await self._mark_push_failed(version.skill_id, operation.operation_id, "local_commit_failed")
            raise SkillRemoteOperationError(operation.operation_id, "local_commit_failed") from exc
        return PushResult(
            operation_id=operation.operation_id,
            skill_id=version.skill_id,
            version_id=version.version_id,
            cloud_skill_id=result.cloud_skill_id,
            remote_content_hash=result.content_hash,
            status=result.version.status,
        )

    @requires_ready
    async def pull(self, skill_ref: str) -> PullResult:
        """Import a complete cloud history without changing local family pointers."""

        remote = self._remote
        if remote is None:
            raise SkillCapabilityUnavailableError("Skill remote pull is not configured")
        detail = await self._manager.get_skill(skill_ref)
        cloud_skill_id = detail.skill.cloud_skill_id
        if cloud_skill_id is None:
            raise SkillConflictError(f"local Skill has no cloud mapping: {detail.skill.skill_id}")

        summaries = await _pull_remote_summaries(remote, cloud_skill_id)
        new_versions, existing_updates, matched = await self._prepare_remote_versions(
            detail,
            summaries,
        )
        imported, concurrently_matched = await self._manager.repository.import_cloud_versions(
            detail.skill.skill_id,
            cloud_skill_id=cloud_skill_id,
            new_versions=new_versions,
            existing_updates=existing_updates,
        )
        matched.extend(version_id for version_id in concurrently_matched if version_id not in matched)
        return PullResult(
            skill_id=detail.skill.skill_id,
            cloud_skill_id=cloud_skill_id,
            imported_version_ids=imported,
            matched_version_ids=matched,
        )

    @requires_ready
    async def sync(self, skill_ref: str) -> SkillDetail:
        """Push pending versions, pull changed facts and advance local sync time."""

        remote = self._remote
        if remote is None:
            raise SkillCapabilityUnavailableError("Skill remote sync is not configured")
        detail = await self._manager.get_skill(skill_ref)
        versions = await self._manager.repository.query_versions(skill_id=detail.skill.skill_id)
        pending_push_ids = {
            operation.version_id
            for operation in await self._manager.repository.list_operations(skill_id=detail.skill.skill_id)
            if operation.operation_type == PendingSkillOperationType.PUSH_VERSION.value
            and operation.status in {"pending", "failed", "running"}
            and operation.version_id is not None
        }
        for version in _order_pending_push_versions(versions, pending_push_ids):
            await self.push(detail.skill.skill_id, version.version_id)

        detail = await self._manager.get_skill(detail.skill.skill_id)
        cloud_skill_id = detail.skill.cloud_skill_id
        if cloud_skill_id is None:
            raise SkillConflictError(f"cloud mapping was not established for {detail.skill.skill_id}")
        versions = await self._manager.repository.query_versions(skill_id=detail.skill.skill_id)
        request = RemoteSyncRequest(
            items=[
                RemoteSyncItem(
                    cloud_skill_id=cloud_skill_id,
                    known_version_revisions={
                        version.version_id: int(
                            version.version_revision
                        )
                        for version in versions
                    },
                )
            ]
        )
        result = await remote.sync(request)
        matching = [item for item in result.items if item.cloud_skill_id == cloud_skill_id]
        if len(result.items) != 1 or len(matching) != 1:
            raise SkillConflictError(f"cloud sync returned an invalid family set for {cloud_skill_id}")
        item = matching[0]
        _ensure_unique_remote_summaries(item.versions)
        new_versions, existing_updates, _ = await self._prepare_remote_versions(
            detail,
            item.versions,
        )
        await self._manager.repository.import_cloud_versions(
            detail.skill.skill_id,
            cloud_skill_id=cloud_skill_id,
            new_versions=new_versions,
            existing_updates=existing_updates,
            now=self._clock(),
        )
        return await self._manager.get_skill(detail.skill.skill_id)

    @requires_ready
    async def analyze(self, request: SkillAnalysisRequest) -> SkillAnalysisResult:
        if self._runtime.skill_algorithms is None:
            raise SkillCapabilityUnavailableError("Skill analysis is not configured")
        result = await self._runtime.skill_algorithms.analyze(request)
        await self._record_algorithm_result(
            SkillApplicationCapability.ANALYZE,
            result.model_dump(mode="json"),
        )
        return result

    @requires_ready
    async def optimize(self, request: SkillOptimizationRequest) -> SkillOptimizationResult:
        if self._runtime.skill_algorithms is None:
            raise SkillCapabilityUnavailableError("Skill optimization is not configured")
        result = await self._runtime.skill_algorithms.optimize(request)
        if not result.changed:
            await self._record_algorithm_result(
                SkillApplicationCapability.OPTIMIZE,
                result.model_dump(mode="json"),
            )
            return result
        candidate = result.skill.model_copy(
            update={
                "metadata": {
                    **result.skill.metadata,
                    "skill_application": {"config_hash": self._config.config_hash},
                }
            }
        )
        persisted = await self._manager.persist_optimized_version(
            candidate,
            base_version_id=request.skill.version_id,
        )
        persisted_result = result.model_copy(update={"skill": persisted})
        await self._record_algorithm_result(
            SkillApplicationCapability.OPTIMIZE,
            persisted_result.model_dump(mode="json"),
        )
        return persisted_result

    @requires_ready
    async def execute(self, agent_name: str, request: AgentExecutionRequest) -> Trajectory:
        """Execute one configured Agent and durably append its physical attempt."""

        try:
            agent = self._runtime.agents[agent_name]
        except KeyError as exc:
            available = ", ".join(sorted(self._runtime.agents)) or "<none>"
            raise SkillCapabilityUnavailableError(
                f"Skill Agent {agent_name!r} is not configured; available Agents: {available}"
            ) from exc
        trajectory = await agent.execute(request)
        trajectory = trajectory.model_copy(
            update={
                "metadata": {
                    **trajectory.metadata,
                    "skill_application": {"config_hash": self._config.config_hash},
                }
            }
        )
        await self.record_trajectory(trajectory)
        return trajectory

    @requires_ready
    async def record_trajectory(self, trajectory: Trajectory) -> None:
        """Atomically append one attempt and its durable trajectory report outbox row."""

        record = trajectory.to_record()
        await self._append_record(TRAJECTORY_TABLE, record.trajectory_id, record)

    @requires_ready
    async def report_trajectory(self, trajectory_id: str) -> None:
        """Resolve exact cloud bindings and deliver one pending trajectory outbox item."""

        remote = self._remote
        if remote is None:
            raise SkillCapabilityUnavailableError("Skill trajectory remote report is not configured")
        records = await self._database.get_records(TRAJECTORY_TABLE, DatabaseScope(), [trajectory_id])
        if not records:
            raise SkillNotFoundError(f"trajectory not found: {trajectory_id}")
        record = from_database_record(records[0], TrajectoryRecord)
        bindings: list[SkillTrajectoryBinding] = []
        for raw in record.skill_bindings:
            version_id = raw.get("version_id")
            usage = raw.get("usage")
            if not isinstance(version_id, str) or not version_id or not isinstance(usage, str):
                raise SkillConflictError("trajectory report requires resolved version_id and usage on every binding")
            version = await self._manager.repository.get_version(version_id)
            if version.cloud_skill_id is None:
                await self.push(version.skill_id, version.version_id)
                version = await self._manager.repository.get_version(version_id)
            assert version.cloud_skill_id is not None
            bindings.append(
                SkillTrajectoryBinding(
                    name=str(raw.get("name") or version.name),
                    cloud_skill_id=version.cloud_skill_id,
                    version_id=version.version_id,
                    base_version_id=raw.get("base_version_id"),
                    content_hash=version.content_hash,
                    version_label=version.version_label,
                    usage=usage,
                    injection_mode=raw.get("injection_mode"),
                )
            )
        source = {
            "trajectory_id": record.trajectory_id,
            "task_id": record.task_id,
            "rollout_id": record.rollout_id,
            "attempt_no": record.attempt_no,
            "rollout_type": record.rollout_type,
            "task_instruction": record.task_instruction,
            "task_system_prompt": record.task_system_prompt,
            "task_tags": record.task_tags,
            "task_metadata": record.task_metadata,
            "env_metadata": record.env_metadata,
            "agent_type": record.agent_type,
            "agent_profile": record.agent_profile,
            "status": record.status,
            "trajectory": record.trajectory,
            "skill_bindings": bindings,
            "reward_score": record.reward_score,
            "reward_detail": record.reward_detail,
            "reward_metadata": record.reward_metadata,
            "started_at": record.started_at,
            "finished_at": record.finished_at,
            "n_turn": record.n_turn,
            "error_info": record.error_info,
            "metadata": record.metadata,
            "source": record.source,
            "source_add_record_id": record.source_add_record_id,
            "created_at": record.created_at,
        }
        source["trajectory_hash"] = compute_trajectory_hash(source)
        upload = SkillTrajectoryUpload.model_validate(source)
        operation_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"mindmemos:trajectory-report:{trajectory_id}"))
        report_request = SkillTrajectoryReportRequest(
            operation_id=operation_id,
            items=[SkillTrajectoryUploadItem(trajectory=upload)],
        )
        try:
            result = await remote.report_trajectories(report_request)
        except Exception as exc:
            await self._database.patch_record(
                SKILL_REMOTE_OPERATION_TABLE,
                DatabaseScope(),
                operation_id,
                {
                    "status": "failed",
                    "lease_expires_at": None,
                    "last_error_code": type(exc).__name__,
                    "updated_at": self._clock(),
                },
            )
            raise
        item = result.items[0]
        if item.status not in {"stored", "duplicate"}:
            raise SkillRemoteOperationError(operation_id, item.error_code or "trajectory_rejected")
        now = self._clock()
        async with self._database.transaction() as transaction:
            current = await transaction.get_records(TRAJECTORY_TABLE, DatabaseScope(), [trajectory_id])
            operation = await transaction.get_records(
                SKILL_REMOTE_OPERATION_TABLE,
                DatabaseScope(),
                [operation_id],
            )
            if not current or not operation:
                raise SkillConflictError("trajectory report commit evidence disappeared")
            await transaction.patch_record(
                TRAJECTORY_TABLE,
                DatabaseScope(),
                trajectory_id,
                {
                    "trajectory_hash": upload.trajectory_hash.removeprefix("sha256:"),
                    "skill_bindings": [binding.model_dump(mode="json") for binding in bindings],
                    "received_at": now,
                },
            )
            await transaction.patch_record(
                SKILL_REMOTE_OPERATION_TABLE,
                DatabaseScope(),
                operation_id,
                {
                    "request_hash": canonical_request_hash(report_request),
                    "status": "succeeded",
                    "lease_expires_at": None,
                    "last_error_code": None,
                    "remote_result": item.model_dump(mode="json"),
                    "updated_at": now,
                },
            )

    @requires_ready
    async def pull_trajectories(self, skill_ref: str, *, limit: int = 100) -> RemoteTrajectoryPage:
        """Pull the independent trajectory cursor stream without coupling it to version sync."""

        if self._remote is None:
            raise SkillCapabilityUnavailableError("Skill trajectory remote pull is not configured")
        detail = await self._manager.get_skill(skill_ref)
        if detail.skill.cloud_skill_id is None:
            raise SkillConflictError(f"local Skill has no cloud mapping: {detail.skill.skill_id}")
        page = await self._remote.list_trajectories(
            RemoteTrajectoryListRequest(
                cloud_skill_id=detail.skill.cloud_skill_id,
                cursor=detail.sync_state.trajectory_pull_cursor,
                limit=limit,
            )
        )
        for item in page.items:
            record = TrajectoryRecord.model_validate(
                {
                    **item.model_dump(mode="json"),
                    "running_dir": None,
                    "injected_skills": [],
                }
            )
            existing = await self._database.get_records(TRAJECTORY_TABLE, DatabaseScope(), [item.trajectory_id])
            if existing:
                current = from_database_record(existing[0], TrajectoryRecord)
                if current.trajectory_hash != record.trajectory_hash:
                    raise SkillConflictError(f"cloud trajectory conflicts locally: {item.trajectory_id}")
                continue
            await self._database.upsert_records(TRAJECTORY_TABLE, [to_database_record(record)])
        await self._manager.repository.update_trajectory_cursor(
            detail.skill.skill_id,
            cursor=page.next_cursor,
            now=self._clock(),
        )
        return page

    @requires_ready
    async def get_trajectory(self, trajectory_id: str) -> Trajectory:
        records = await self._database.get_records(TRAJECTORY_TABLE, DatabaseScope(), [trajectory_id])
        if not records:
            raise SkillNotFoundError(f"trajectory not found: {trajectory_id}")
        return Trajectory.from_record(from_database_record(records[0], TrajectoryRecord))

    @requires_ready
    async def record_algorithm_log(self, log: AlgorithmLog) -> None:
        """Append one algorithm step report without overwriting prior evidence."""

        record = log.to_record().model_copy(
            update={
                "payload": {
                    **log.step.payload,
                    "skill_application": {"config_hash": self._config.config_hash},
                }
            }
        )
        await self._append_record(ALGORITHM_LOG_TABLE, record.log_id, record)

    @requires_ready
    async def get_algorithm_log(self, log_id: str) -> AlgorithmLog:
        records = await self._database.get_records(ALGORITHM_LOG_TABLE, DatabaseScope(), [log_id])
        if not records:
            raise SkillNotFoundError(f"algorithm log not found: {log_id}")
        return AlgorithmLog.from_record(from_database_record(records[0], AlgorithmLogRecord))

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    def _ensure_ready(self) -> None:
        if self._closed:
            raise SkillServiceClosedError("SkillApplication is closed")
        self._ensure_owner_loop()
        if not self._started:
            raise RuntimeError("SkillApplication is not started")

    async def _append_record(
        self,
        table: str,
        record_id: str,
        record: TrajectoryRecord | AlgorithmLogRecord,
    ) -> None:
        async with self._database.transaction() as transaction:
            existing = await transaction.get_records(table, DatabaseScope(), [record_id])
            if existing:
                raise SkillConflictError(f"immutable persistence record already exists: {table}/{record_id}")
            if isinstance(record, TrajectoryRecord):
                attempts, _ = await transaction.query_records(
                    TRAJECTORY_TABLE,
                    RecordQuery(
                        filters=FilterGroup(
                            operator="and",
                            clauses=(
                                Predicate(field="rollout_id", op="eq", value=record.rollout_id),
                                Predicate(field="attempt_no", op="eq", value=record.attempt_no),
                            ),
                        ),
                        page=Page(limit=1),
                    ),
                )
                if attempts:
                    raise SkillConflictError(
                        f"trajectory rollout attempt already exists: {record.rollout_id}/{record.attempt_no}"
                    )
                operation_id = str(
                    uuid.uuid5(uuid.NAMESPACE_URL, f"mindmemos:trajectory-report:{record.trajectory_id}")
                )
                operation = SkillRemoteOperationRecord(
                    operation_id=operation_id,
                    operation_type=PendingSkillOperationType.REPORT_TRAJECTORY.value,
                    trajectory_id=record.trajectory_id,
                    request_hash=canonical_request_hash(record.model_dump(mode="json")),
                    status=PendingSkillOperationStatus.PENDING.value,
                    created_at=record.created_at,
                    updated_at=record.created_at,
                )
                await transaction.upsert_records(
                    SKILL_REMOTE_OPERATION_TABLE,
                    [to_database_record(operation)],
                )
            await transaction.upsert_records(table, [to_database_record(record)])

    async def _record_algorithm_result(
        self,
        capability: SkillApplicationCapability,
        payload: dict,
    ) -> None:
        capability_name = capability.value
        owner = self._runtime.algorithm_owners[capability_name]
        compiled = self._config.runtime.algorithms[owner]
        await self.record_algorithm_log(
            AlgorithmLog(
                log_id=self._id_generator(),
                algorithm={"name": compiled.type},
                step={
                    "component_name": owner,
                    "name": capability_name,
                    "status": AlgorithmResultStatus.SUCCEEDED.value,
                    "payload": {"result": payload},
                    "created_at": self._clock(),
                },
            )
        )

    async def _prepare_remote_versions(
        self,
        detail: SkillDetail,
        summaries: list[RemoteVersionSummary],
    ) -> tuple[
        list[SkillRecord],
        dict[str, tuple[SkillRecord, SkillRecord]],
        list[str],
    ]:
        remote = self._remote
        if remote is None:
            raise SkillCapabilityUnavailableError("Skill remote import is not configured")
        cloud_skill_id = detail.skill.cloud_skill_id
        if cloud_skill_id is None:
            raise SkillConflictError(f"local Skill has no cloud mapping: {detail.skill.skill_id}")
        _ensure_unique_remote_summaries(summaries)
        local_versions = await self._manager.repository.query_versions(skill_id=detail.skill.skill_id)
        local_by_id = {version.version_id: version for version in local_versions}
        existing_updates: dict[str, tuple[SkillRecord, SkillRecord]] = {}
        for summary in summaries:
            existing = local_by_id.get(summary.version_id)
            if existing is None:
                continue
            existing_updates[summary.version_id] = (
                existing,
                _matched_cloud_version(existing, summary, cloud_skill_id),
            )

        ordered_missing = _order_missing_remote_versions(
            summaries,
            known_version_ids=set(local_by_id),
        )
        snapshots = {}
        new_versions: list[SkillRecord] = []
        for summary in ordered_missing:
            content = await remote.pull_content(cloud_skill_id, summary.version_id)
            if content.version != summary:
                raise SkillConflictError(f"cloud version metadata changed while pulling: {summary.version_id}")
            actual_hash = content.bundle.content_hash
            if actual_hash != summary.content_hash:
                raise SkillConflictError(
                    f"cloud content hash mismatch for {summary.version_id}: "
                    f"expected {summary.content_hash}, got {actual_hash}"
                )
            try:
                remote_files = {item.path: item.content for item in content.bundle.files}
            except ValueError as exc:
                raise SkillConflictError(f"invalid cloud content for {summary.version_id}") from exc

            parent_ids = summary.parent_version_ids
            parent_snapshot = None
            if parent_ids:
                parent_id = parent_ids[0]
                parent_snapshot = snapshots.get(parent_id)
                if parent_snapshot is None:
                    parent = local_by_id.get(parent_id)
                    if parent is None:
                        raise SkillConflictError(f"cloud parent version is missing: {parent_id}")
                    parent_snapshot = snapshot_from_record(parent)
            snapshot = snapshot_from_cloud_bundle(remote_files, parent_snapshot)
            snapshots[summary.version_id] = snapshot
            version_label = summary.version_label or frontmatter_value(
                remote_files["SKILL.md"],
                "version",
            )
            if version_label is None:
                raise SkillConflictError(f"cloud version has no version label: {summary.version_id}")
            parse_version_label(version_label)
            record = SkillRecord(
                skill_id=detail.skill.skill_id,
                version_id=summary.version_id,
                cloud_skill_id=cloud_skill_id,
                parent_version_ids=parent_ids,
                name=detail.skill.name,
                description=frontmatter_value(remote_files["SKILL.md"], "description"),
                alias=detail.skill.alias,
                bundle=content.bundle.canonical_json(),
                resources=serialize_files(snapshot.resources),
                content_hash=snapshot.content_hash,
                local_snapshot_hash=snapshot.local_snapshot_hash,
                status=summary.status,
                version_revision=summary.version_revision,
                version_label=version_label,
                commit_message=summary.commit_message,
                metadata=summary.metadata,
                local_metadata={"snapshot": snapshot_metadata(snapshot)},
                created_at=summary.created_at,
                updated_at=summary.updated_at,
                received_at=summary.received_at,
                origin=summary.origin,
            )
            new_versions.append(record)
            local_by_id[summary.version_id] = record
        matched = [summary.version_id for summary in summaries if summary.version_id in existing_updates]
        return new_versions, existing_updates, matched

    async def _mark_push_failed(self, skill_id: str, operation_id: str, error_code: str) -> None:
        try:
            await self._manager.repository.mark_push_failed(
                skill_id,
                operation_id=operation_id,
                error_code=error_code,
                now=self._clock(),
            )
        except Exception:
            # Preserve the triggering error. A running operation remains recoverable
            # after its lease expires if this bookkeeping loses a concurrent CAS race.
            pass

    @staticmethod
    def _push_response_mismatch(request: RemotePushRequest, result: RemotePushResult) -> str | None:
        if result.version.version_id != request.version.version_id:
            return "immutable_version_id_mismatch"
        if result.version.content_hash != request.version.content_hash:
            return "immutable_content_hash_mismatch"
        if result.version.created_at != request.version.created_at:
            return "immutable_created_at_mismatch"
        if request.cloud_skill_id is not None and result.cloud_skill_id != request.cloud_skill_id:
            return "cloud_skill_id_mismatch"
        return None

    def _ensure_owner_loop(self) -> None:
        if self._owner_loop is not None and asyncio.get_running_loop() is not self._owner_loop:
            raise RuntimeError("SkillApplication cannot be used across event loops")


def _ensure_unique_remote_summaries(summaries: list[RemoteVersionSummary]) -> None:
    version_ids = [summary.version_id for summary in summaries]
    if len(version_ids) != len(set(version_ids)):
        duplicate = next(version_id for version_id in version_ids if version_ids.count(version_id) > 1)
        raise SkillConflictError(f"cloud pull returned duplicate version: {duplicate}")


def _order_pending_push_versions(
    versions: list[SkillRecord],
    pending_version_ids: set[str],
) -> list[SkillRecord]:
    by_id = {version.version_id: version for version in versions}
    missing = pending_version_ids - set(by_id)
    if missing:
        raise SkillConflictError(f"pending push versions do not exist: {', '.join(sorted(missing))}")
    for version_id in pending_version_ids:
        version = by_id[version_id]
        for parent_id in version.parent_version_ids:
            if parent_id in pending_version_ids:
                continue
            parent = by_id.get(parent_id)
            cloud = parent.metadata.get("cloud") if parent is not None else None
            if not isinstance(cloud, dict) or not isinstance(cloud.get("content_hash"), str):
                raise SkillConflictError(f"parent version is not confirmed by cloud: {parent_id}")
    remaining = set(pending_version_ids)
    available = set(by_id) - remaining
    ordered: list[SkillRecord] = []
    while remaining:
        ready = [
            version
            for version in versions
            if version.version_id in remaining
            and all(parent_id in available for parent_id in version.parent_version_ids)
        ]
        if not ready:
            raise SkillConflictError("pending push version graph has missing parents or a cycle")
        for version in ready:
            ordered.append(version)
            available.add(version.version_id)
            remaining.remove(version.version_id)
    return ordered


async def _pull_remote_summaries(
    remote: SkillRemotePort,
    cloud_skill_id: str,
) -> list[RemoteVersionSummary]:
    summaries: list[RemoteVersionSummary] = []
    version_ids: set[str] = set()
    seen_cursors: set[str | None] = set()
    cursor: str | None = None
    while True:
        if cursor in seen_cursors:
            raise SkillConflictError(f"cloud pull cursor repeated: {cursor!r}")
        seen_cursors.add(cursor)
        page = await remote.pull_versions(cloud_skill_id, cursor)
        for summary in page.versions:
            if summary.version_id in version_ids:
                raise SkillConflictError(f"cloud pull returned duplicate version: {summary.version_id}")
            version_ids.add(summary.version_id)
            summaries.append(summary)
        if page.next_cursor is None:
            return summaries
        cursor = page.next_cursor


def _order_missing_remote_versions(
    summaries: list[RemoteVersionSummary],
    *,
    known_version_ids: set[str],
) -> list[RemoteVersionSummary]:
    remaining = [summary for summary in summaries if summary.version_id not in known_version_ids]
    ordered: list[RemoteVersionSummary] = []
    available = set(known_version_ids)
    while remaining:
        ready = [
            summary
            for summary in remaining
            if all(parent_id in available for parent_id in summary.parent_version_ids)
        ]
        if not ready:
            unresolved = ", ".join(
                f"{summary.version_id}->{','.join(summary.parent_version_ids) or '<root>'}" for summary in remaining
            )
            raise SkillConflictError(f"cloud version graph has missing parents or a cycle: {unresolved}")
        for summary in ready:
            ordered.append(summary)
            available.add(summary.version_id)
            remaining.remove(summary)
    return ordered


def _matched_cloud_version(
    existing: SkillRecord,
    summary: RemoteVersionSummary,
    cloud_skill_id: str,
) -> SkillRecord:
    if existing.parent_version_ids != summary.parent_version_ids:
        raise SkillConflictError(f"cloud parent conflicts with local version: {summary.version_id}")
    if existing.version_label != summary.version_label:
        raise SkillConflictError(f"cloud version label conflicts with local version: {summary.version_id}")
    if existing.commit_message != summary.commit_message or existing.created_at != summary.created_at:
        raise SkillConflictError(f"cloud metadata conflicts with immutable local version: {summary.version_id}")
    if existing.cloud_skill_id is not None and existing.cloud_skill_id != cloud_skill_id:
        raise SkillConflictError(f"cloud Skill mapping conflicts with local version: {summary.version_id}")
    if existing.content_hash != summary.content_hash:
        raise SkillConflictError(f"cloud content conflicts with immutable local version: {summary.version_id}")
    return existing.model_copy(
        update={
            "cloud_skill_id": cloud_skill_id,
            "status": summary.status,
            "version_revision": summary.version_revision,
            "metadata": summary.metadata,
            "updated_at": summary.updated_at,
            "received_at": summary.received_at,
        }
    )


def _management_sync_state(
    skill: ManagedSkill,
    operations: list[PendingSkillOperation],
    *,
    remote_configured: bool,
) -> SkillManagementSyncState:
    if not remote_configured:
        return SkillManagementSyncState.LOCAL_ONLY
    if any(operation.status is PendingSkillOperationStatus.FAILED for operation in operations):
        return SkillManagementSyncState.FAILED
    if any(operation.status in {PendingSkillOperationStatus.PENDING, PendingSkillOperationStatus.RUNNING} for operation in operations):
        return SkillManagementSyncState.PENDING
    if skill.cloud_skill_id is None:
        return SkillManagementSyncState.LOCAL_ONLY
    return SkillManagementSyncState.SYNCED


__all__ = ["SkillApplication"]
