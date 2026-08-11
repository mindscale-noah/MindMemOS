"""Synchronous compatibility facade over ``mindmemos_skill.SkillApplication``.

The SDK owns only the async-to-sync bridge and DTO projection. All Skill state,
version and synchronization behavior executes inside the standalone Application.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
from collections.abc import Coroutine
from datetime import datetime
from typing import Any, TypeVar

from mindmemos_skill.management import (
    DuplicateAction,
    PendingSkillOperation,
    PendingSkillOperationStatus,
    SkillManagementDetail,
    SkillManagementOverview,
    SkillSnapshot,
    SnapshotFileRole,
    snapshot_from_record,
)
from mindmemos_skill.management import (
    ExportSkillRequest as ApplicationExportSkillRequest,
)
from mindmemos_skill.management import (
    PublishSkillRequest as ApplicationPublishSkillRequest,
)
from mindmemos_skill.management import (
    RegisterSkillRequest as ApplicationRegisterSkillRequest,
)
from mindmemos_skill.persistence import (
    SkillRecord as ApplicationSkillRecord,
)
from mindmemos_skill.persistence import (
    SkillVersionOrigin as ApplicationSkillOrigin,
)
from mindmemos_skill.persistence import (
    SkillVersionStatus as ApplicationSkillStatus,
)

from mindmemos_skill import (
    EvolveRunRequest,
    MindMemOSSkillError,
    RemotePushRequest,
    RemotePushResult,
    RemoteSyncRequest,
    RemoteSyncResult,
    RemoteSyncResultItem,
    RemoteVersionContent,
    RemoteVersionsPage,
    RemoteVersionSummary,
    SkillAlgorithmRunResult,
    SkillApplication,
    SkillBundle,
    Trace2SkillRunRequest,
)

from ..config import CompiledSDKProfileV2, ConfigManager, SDKConfigCompilerV2
from ..errors import SkillRegistryError, translate_skill_error
from .bundle import serialize_bundle
from .http_adapter import _translate_remote_errors
from .models import (
    DuplicateSkillAction,
    DuplicateSkillMatch,
    ExportSkillRequest,
    ExportSkillResult,
    LocalSkillFileEntry,
    LocalSkillFileRole,
    LocalSkillManifest,
    LocalSkillOperationStatus,
    LocalSkillOperationType,
    LocalSkillSnapshot,
    LocalSkillSyncState,
    LocalSkillVersionMetadata,
    PublishLocalRequest,
    PublishLocalResult,
    PullVersionContent,
    PullVersionsPage,
    PullVersionSummary,
    PushVersionRequest,
    PushVersionResult,
    RegisterLocalRequest,
    RegisterLocalResult,
    SkillContext,
    SkillDiffResult,
    SkillOrigin,
    SkillUsage,
    SkillVersionStatus,
    SyncCloudItem,
    SyncCloudRequest,
    SyncCloudResult,
)

_ResultT = TypeVar("_ResultT")


class SkillRegisterPlan:
    """Deprecated plain compatibility value retained for import stability."""

    def __init__(
        self,
        *,
        path: str,
        skill_name: str,
        version_label: str | None,
        content_hash: str,
        base_version_id: str,
    ) -> None:
        self.path = path
        self.skill_name = skill_name
        self.version_label = version_label
        self.content_hash = content_hash
        self.base_version_id = base_version_id


class _ApplicationRunner:
    """Keep Application start/call/close on one dedicated owner event loop."""

    def __init__(self, config: Any, *, remote: Any = None, connection: Any = None) -> None:
        self._loop_ready: concurrent.futures.Future[asyncio.AbstractEventLoop] = concurrent.futures.Future()
        self._thread = threading.Thread(target=self._run_loop, name="mindmemos-skill-application", daemon=True)
        self._thread.start()
        self._loop = self._loop_ready.result()
        try:
            self._application, self._connection = asyncio.run_coroutine_threadsafe(
                self._start_application(config, remote=remote, connection=connection),
                self._loop,
            ).result()
        except BaseException:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join()
            raise
        self._closed = False

    @staticmethod
    async def _start_application(config: Any, *, remote: Any, connection: Any) -> tuple[SkillApplication, Any]:
        if connection is not None:
            from .http_adapter import HttpSkillRemoteAdapter

            await connection.open()
            remote = HttpSkillRemoteAdapter(connection)
        try:
            application = await SkillApplication.from_config(config, remote=remote)
        except BaseException:
            if connection is not None:
                await connection.aclose()
            raise
        return application, connection

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop_ready.set_result(loop)
        try:
            loop.run_forever()
        finally:
            loop.close()

    def call(self, coroutine: Coroutine[Any, Any, _ResultT]) -> _ResultT:
        if self._closed:
            coroutine.close()
            raise SkillRegistryError("SkillManager is closed")
        return asyncio.run_coroutine_threadsafe(coroutine, self._loop).result()

    @property
    def application(self) -> SkillApplication:
        return self._application

    async def _close_application(self) -> None:
        error: BaseException | None = None
        try:
            await self._application.close()
        except BaseException as exc:
            error = exc
        if self._connection is not None:
            try:
                await self._connection.aclose()
            except BaseException as exc:
                if error is None:
                    error = exc
        if error is not None:
            raise error

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            asyncio.run_coroutine_threadsafe(self._close_application(), self._loop).result()
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join()


class SkillManager:
    """Synchronous SDK facade whose only business dependency is SkillApplication."""

    def __init__(self, *, runner: _ApplicationRunner) -> None:
        self._runner = runner

    @classmethod
    def from_config_manager(
        cls,
        config_manager: ConfigManager,
        cloud: object | None = None,
        *,
        shared_connection_name: str | None = None,
    ) -> SkillManager:
        """Create the sync facade from the active portal v2 profile."""

        if config_manager.portal_exists() or config_manager.legacy_exists():
            profile = config_manager.compile_portal().profile
        else:
            profile = SDKConfigCompilerV2().compile(config_manager.default_portal()).profile
        if cloud is not None and shared_connection_name is None:
            shared_connection_name = profile.skill_connection
        return cls.from_portal_profile(
            profile,
            cloud=cloud,
            shared_connection_name=shared_connection_name,
        )

    @classmethod
    def from_portal_profile(
        cls,
        profile: CompiledSDKProfileV2,
        *,
        cloud: object | None = None,
        shared_connection_name: str | None = None,
    ) -> SkillManager:
        """Create the facade from one already-compiled portal profile."""

        if profile.skill_connection is None:
            return cls(runner=_ApplicationRunner(profile.skill_application))
        if shared_connection_name == profile.skill_connection and cloud is not None and _is_remote_capable(cloud):
            return cls(
                runner=_ApplicationRunner(
                    profile.skill_application,
                    remote=_SyncCloudRemoteAdapter(cloud),
                )
            )
        from ..connections import HttpConnection

        connection = HttpConnection(profile.connections[profile.skill_connection])
        return cls(runner=_ApplicationRunner(profile.skill_application, connection=connection))

    def close(self) -> None:
        self._runner.close()

    def __enter__(self) -> SkillManager:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        runner = getattr(self, "_runner", None)
        if runner is not None:
            try:
                runner.close()
            except BaseException:
                pass

    def management_overview(self) -> SkillManagementOverview:
        return self._call(self._runner.application.get_management_overview())

    def management_detail(self, skill_ref: str) -> SkillManagementDetail:
        return self._call(self._runner.application.get_management_detail(skill_ref))

    def register_local(self, request: RegisterLocalRequest) -> RegisterLocalResult:
        result = self._call(
            self._runner.application.register(
                ApplicationRegisterSkillRequest(
                    source_path=request.source_path,
                    name=request.name,
                    alias=request.alias,
                    version_label=request.version_label,
                    commit_message=request.commit_message,
                    duplicate_action=_duplicate_action(request.duplicate_action),
                )
            )
        )
        summary = None
        if result.action == "reused":
            manifest = self.show_local(result.skill_id)
            version = self.get_local_version(result.skill_id, result.version_id)
            summary = DuplicateSkillMatch(
                local_snapshot_hash=version.local_snapshot_hash,
                skill_id=manifest.skill_id,
                name=manifest.name,
                latest_version_id=manifest.latest_version_id,
                matched_version_id=result.version_id,
                cloud_skill_id=manifest.cloud_skill_id,
                last_sync_at=manifest.last_sync_at,
            )
        return RegisterLocalResult(
            action=result.action,
            skill_id=result.skill_id,
            version_id=result.version_id,
            latest_version_id=result.version_id,
            summary=summary,
        )

    def publish_local(self, request: PublishLocalRequest) -> PublishLocalResult:
        result = self._call(
            self._runner.application.publish(
                ApplicationPublishSkillRequest(
                    skill_ref=request.skill_id,
                    base_version_id=request.base_version_id,
                    source_path=request.source_path,
                    content=request.content,
                    files=request.files,
                    version_label=request.version_label,
                    commit_message=request.commit_message,
                )
            )
        )
        return PublishLocalResult(
            skill_id=result.skill_id,
            version_id=result.version_id,
            latest_version_id=result.version_id,
            local_snapshot_hash=result.local_snapshot_hash,
        )

    def list_local(self) -> list[LocalSkillManifest]:
        overview = self.management_overview()
        return [self._manifest(summary.skill.skill_id) for summary in overview.skills]

    def show_local(self, skill_ref: str) -> LocalSkillManifest:
        return self._manifest(skill_ref)

    def local_history(self, skill_ref: str) -> list[LocalSkillVersionMetadata]:
        detail = self.management_detail(skill_ref)
        return [self._version_metadata(record, detail.pending_operations) for record in detail.versions]

    def get_local_version(self, skill_ref: str, version_id: str) -> LocalSkillVersionMetadata:
        detail = self.management_detail(skill_ref)
        try:
            record = next(version for version in detail.versions if version.version_id == version_id)
        except StopIteration as exc:
            raise SkillRegistryError(f"Skill version not found: {version_id}") from exc
        return self._version_metadata(record, detail.pending_operations)

    def pending_local_operations(self, skill_ref: str | None = None) -> list[Any]:
        if skill_ref is None:
            operations = self.management_overview().pending_operations
        else:
            operations = self.management_detail(skill_ref).pending_operations
        return [_local_operation(operation) for operation in operations]

    def get_local_snapshot(self, skill_ref: str, *, version_id: str | None = None) -> LocalSkillSnapshot:
        detail = self.management_detail(skill_ref)
        resolved = version_id or detail.latest_version.version_id
        try:
            record = next(version for version in detail.versions if version.version_id == resolved)
        except StopIteration as exc:
            raise SkillRegistryError(f"Skill version not found: {resolved}") from exc
        return _local_snapshot(snapshot_from_record(record))

    def export_local(self, request: ExportSkillRequest) -> ExportSkillResult:
        result = self._call(
            self._runner.application.export(
                ApplicationExportSkillRequest(
                    skill_ref=request.skill_id,
                    target_path=request.target_path,
                    version_id=request.version_id,
                    replace=request.replace,
                )
            )
        )
        return ExportSkillResult.model_validate(result.model_dump(mode="json"))

    def diff_local(
        self,
        skill_ref: str,
        *,
        from_version_id: str | None = None,
        to_version_id: str,
    ) -> SkillDiffResult:
        result = self._call(
            self._runner.application.diff(
                skill_ref,
                from_version_id=from_version_id,
                to_version_id=to_version_id,
            )
        )
        return SkillDiffResult(
            skill_id=result.skill_id,
            from_version_id=result.from_version_id,
            to_version_id=result.to_version_id,
            diff=result.diff,
        )

    def latest_skill_context(self, skill_ref: str, *, usage: SkillUsage | str | None = None) -> SkillContext:
        detail = self.management_detail(skill_ref)
        return SkillContext(
            name=detail.skill.name,
            content_hash=detail.latest_version.content_hash,
            version_id=detail.latest_version.version_id,
            version_label=detail.latest_version.version_label,
            usage=usage,
        )

    def ensure_skill_context(self, skill_id: str, *, usage: SkillUsage | str | None = None) -> SkillContext:
        return self.latest_skill_context(skill_id, usage=usage)

    def ensure_latest_cloud_version(self, skill_ref: str) -> PushVersionResult:
        detail = self.management_detail(skill_ref)
        return self.push_local(detail.skill.skill_id, version_id=detail.latest_version.version_id)

    def resolve_skill_context(self, messages: list[object]) -> list[SkillContext]:
        contexts = self._call(self._runner.application.resolve_skill_context(messages))
        return [SkillContext.model_validate(context.model_dump(mode="json")) for context in contexts]

    def skill_id_for_context(self, context: SkillContext) -> str | None:
        return next(
            (item.skill.skill_id for item in self.management_overview().skills if item.skill.name == context.name), None
        )

    def push_local(self, skill_ref: str, *, version_id: str | None = None) -> PushVersionResult:
        result = self._call(self._runner.application.push(skill_ref, version_id))
        record = self._call(self._runner.application.get_version(result.skill_id, result.version_id))
        return PushVersionResult(
            cloud_skill_id=result.cloud_skill_id,
            version_id=result.version_id,
            content_hash=result.remote_content_hash,
            status=SkillVersionStatus(result.status.value),
            created_at=_iso(record.created_at),
            received_at=_iso(record.created_at),
        )

    def pull_local(self, skill_ref: str) -> list[LocalSkillVersionMetadata]:
        self._call(self._runner.application.pull(skill_ref))
        return self.local_history(skill_ref)

    def sync_local(self, skill_ref: str, *, direction: str = "both") -> LocalSkillManifest:
        if direction == "both":
            self._call(self._runner.application.sync(skill_ref))
        elif direction == "pull":
            self._call(self._runner.application.pull(skill_ref))
        elif direction == "push":
            detail = self.management_detail(skill_ref)
            version_ids = [
                operation.version_id
                for operation in detail.pending_operations
                if operation.operation_type.value == "push_version" and operation.version_id is not None
            ]
            for version_id in version_ids:
                self._call(self._runner.application.push(detail.skill.skill_id, version_id))
        else:
            raise SkillRegistryError("sync direction must be 'push', 'pull', or 'both'")
        return self.show_local(skill_ref)

    def evolve_local(self, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise SkillRegistryError("cloud evolve is not part of the SkillApplication remote port")

    def run_trace2skill_local(self, request: Trace2SkillRunRequest) -> SkillAlgorithmRunResult:
        return self._call(self._runner.application.run_trace2skill(request))

    def run_evolve_local(self, request: EvolveRunRequest) -> SkillAlgorithmRunResult:
        return self._call(self._runner.application.run_evolve(request))

    def unregister(self, skill_ref: str) -> LocalSkillManifest:
        manifest = self.show_local(skill_ref)
        self._call(self._runner.application.unregister(skill_ref))
        return manifest

    def _manifest(self, skill_ref: str) -> LocalSkillManifest:
        detail = self.management_detail(skill_ref)
        skill = detail.skill
        return LocalSkillManifest(
            skill_id=skill.skill_id,
            name=skill.name,
            alias=skill.alias,
            cloud_skill_id=skill.cloud_skill_id,
            latest_version_id=detail.latest_version.version_id,
            version_ids=[version.version_id for version in detail.versions],
            last_sync_at=_optional_iso(skill.last_version_sync_at),
            created_at=_iso(skill.created_at),
            updated_at=_iso(skill.updated_at),
        )

    @staticmethod
    def _version_metadata(
        record: ApplicationSkillRecord,
        operations: list[PendingSkillOperation],
    ) -> LocalSkillVersionMetadata:
        snapshot = snapshot_from_record(record)
        operation = next((item for item in operations if item.version_id == record.version_id), None)
        if operation is not None and operation.status is PendingSkillOperationStatus.FAILED:
            sync_state = LocalSkillSyncState.FAILED
        elif operation is not None:
            sync_state = LocalSkillSyncState.PENDING
        elif isinstance(record.metadata.get("cloud"), dict) or record.origin is ApplicationSkillOrigin.CLOUD:
            sync_state = LocalSkillSyncState.SYNCED
        else:
            sync_state = LocalSkillSyncState.LOCAL_ONLY
        return LocalSkillVersionMetadata(
            version_id=record.version_id,
            skill_id=record.skill_id,
            parent_version_ids=list(record.parent_version_ids),
            skill_name=record.name,
            content_hash=record.content_hash,
            local_snapshot_hash=snapshot.local_snapshot_hash,
            version_label=record.version_label,
            commit_message=record.commit_message,
            origin=SkillOrigin(record.origin.value),
            cloud_status=SkillVersionStatus(record.status.value),
            sync_state=sync_state,
            created_at=_iso(record.created_at),
        )

    def _call(self, coroutine: Coroutine[Any, Any, _ResultT]) -> _ResultT:
        try:
            return self._runner.call(coroutine)
        except SkillRegistryError:
            raise
        except MindMemOSSkillError as exc:
            raise translate_skill_error(exc) from exc


class _SyncCloudRemoteAdapter:
    """Temporary transport adapter for the legacy synchronous SDK connection."""

    def __init__(self, cloud: object) -> None:
        self._cloud = cloud

    @_translate_remote_errors
    async def push_version(self, request: RemotePushRequest) -> RemotePushResult:
        version = request.version
        result = self._cloud.push_version(
            PushVersionRequest(
                operation_id=request.operation_id,
                cloud_skill_id=request.cloud_skill_id,
                name=version.name,
                version_id=version.version_id,
                parent_version_ids=list(version.parent_version_ids),
                content=request.bundle.canonical_json(),
                expected_content_hash=version.content_hash,
                version_label=version.version_label,
                commit_message=version.commit_message,
                status=SkillVersionStatus(version.status.value),
                origin=SkillOrigin(version.origin.value),
                version_revision=version.version_revision,
                metadata=version.metadata,
                created_at=_iso(version.created_at),
            )
        )
        received_at = datetime.fromisoformat(result.received_at.replace("Z", "+00:00"))
        return RemotePushResult(
            version=version.model_copy(
                update={
                    "cloud_skill_id": result.cloud_skill_id,
                    "status": _application_status(result.status.value),
                    "received_at": received_at,
                }
            )
        )

    @_translate_remote_errors
    async def pull_versions(self, cloud_skill_id: str, cursor: str | None = None) -> RemoteVersionsPage:
        result: PullVersionsPage = self._cloud.pull_versions(cloud_skill_id, cursor=cursor)
        return RemoteVersionsPage(
            versions=[_remote_version(version) for version in result.versions],
            next_cursor=result.next_cursor,
        )

    @_translate_remote_errors
    async def pull_content(self, cloud_skill_id: str, version_id: str) -> RemoteVersionContent:
        result: PullVersionContent = self._cloud.pull_content(cloud_skill_id, version_id)
        return RemoteVersionContent(
            version=_remote_version(result.version),
            bundle=SkillBundle.model_validate_json(result.content),
        )

    @_translate_remote_errors
    async def sync(self, request: RemoteSyncRequest) -> RemoteSyncResult:
        wire = SyncCloudRequest(
            items=[
                SyncCloudItem(
                    cloud_skill_id=item.cloud_skill_id,
                    known_version_revisions=item.known_version_revisions,
                )
                for item in request.items
            ]
        )
        result: SyncCloudResult = self._cloud.sync_cloud(wire)
        return RemoteSyncResult(
            items=[
                RemoteSyncResultItem(
                    cloud_skill_id=item.cloud_skill_id,
                    versions=[_remote_version(version) for version in item.versions],
                )
                for item in result.items
            ]
        )

def _is_remote_capable(cloud: object) -> bool:
    return any(
        callable(getattr(cloud, name, None))
        for name in ("push_version", "pull_versions", "pull_content", "sync_cloud")
    )


def _duplicate_action(value: DuplicateSkillAction | None) -> DuplicateAction | None:
    return DuplicateAction(value.value) if value is not None else None


def _application_status(value: str) -> ApplicationSkillStatus:
    return {
        "rejected": ApplicationSkillStatus.REJECTED,
        "published": ApplicationSkillStatus.PUBLISHED,
        "archived": ApplicationSkillStatus.ARCHIVED,
    }.get(value, ApplicationSkillStatus.DRAFT)


def _application_origin(value: SkillOrigin) -> ApplicationSkillOrigin:
    return ApplicationSkillOrigin(value.value)


def _remote_version(version: PullVersionSummary) -> RemoteVersionSummary:
    return RemoteVersionSummary(
        version_id=version.version_id,
        cloud_skill_id=version.cloud_skill_id,
        parent_version_ids=version.parent_version_ids,
        name=version.name,
        content_hash=version.content_hash,
        version_label=version.version_label,
        commit_message=version.commit_message,
        origin=_application_origin(version.origin),
        status=_application_status(version.status.value),
        version_revision=version.version_revision,
        metadata=version.metadata,
        created_at=version.created_at,
        updated_at=version.updated_at or version.created_at,
        received_at=version.received_at,
    )


def _local_operation(operation: PendingSkillOperation) -> Any:
    from .models import LocalSyncOperation

    return LocalSyncOperation(
        operation_id=operation.operation_id,
        operation_type=LocalSkillOperationType(operation.operation_type.value),
        skill_id=operation.skill_id,
        version_id=operation.version_id,
        status=LocalSkillOperationStatus(operation.status.value),
        attempt_count=operation.attempt_count,
        next_retry_at=_optional_iso(operation.next_retry_at),
        last_error_code=operation.last_error_code,
        created_at=_iso(operation.created_at),
        updated_at=_iso(operation.updated_at),
    )


def _local_snapshot(snapshot: SkillSnapshot) -> LocalSkillSnapshot:
    files = [
        LocalSkillFileEntry(
            path=item.path,
            blob_hash=item.content_hash,
            byte_size=item.byte_size,
            media_type=item.media_type,
            mode=item.mode,
            role=_local_file_role(item.role),
        )
        for item in snapshot.files
    ]
    return LocalSkillSnapshot(
        content=serialize_bundle(snapshot.blob),
        content_hash=snapshot.content_hash,
        local_snapshot_hash=snapshot.local_snapshot_hash,
        files=files,
        file_contents=snapshot.file_contents,
    )


def _local_file_role(role: SnapshotFileRole) -> LocalSkillFileRole:
    return {
        SnapshotFileRole.ALGORITHM: LocalSkillFileRole.ALGORITHM,
        SnapshotFileRole.SCRIPT: LocalSkillFileRole.SCRIPT,
        SnapshotFileRole.REFERENCE: LocalSkillFileRole.REFERENCE,
        SnapshotFileRole.RESOURCE: LocalSkillFileRole.ASSET,
    }[role]


def _iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _optional_iso(value: datetime | None) -> str | None:
    return _iso(value) if value is not None else None


__all__ = ["SkillManager", "SkillRegisterPlan"]
