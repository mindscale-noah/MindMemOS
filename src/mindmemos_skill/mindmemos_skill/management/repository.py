"""Transactional local Skill DAG, sync state and flat remote outbox."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import timedelta

from ..contracts import canonical_request_hash
from ..errors import SkillConflictError, SkillNotFoundError
from ..infra.database import DatabaseScope, FilterGroup, Page, Predicate, RecordQuery, Sort
from ..persistence import (
    SKILL_REMOTE_OPERATION_TABLE,
    SKILL_SYNC_STATE_TABLE,
    SKILL_TABLE,
    SkillRecord,
    SkillRemoteOperationRecord,
    SkillSyncStateRecord,
    SkillVersionStatus,
    from_database_record,
    to_database_record,
)
from .models import (
    PendingSkillOperation,
    PendingSkillOperationStatus,
    PendingSkillOperationType,
    push_operation_id,
)


class SkillRepository:
    def __init__(self, database) -> None:
        self._database = database

    @property
    def database(self):
        return self._database

    async def create_version(
        self,
        record: SkillRecord,
        *,
        now,
        pending_operation: PendingSkillOperation | None = None,
    ) -> SkillSyncStateRecord:
        async with self._database.transaction() as transaction:
            if await self._get_version(transaction, record.version_id) is not None:
                raise SkillConflictError(f"Skill version already exists: {record.version_id}")
            versions = await self._query_versions(transaction, skill_id=record.skill_id)
            await self._validate_new_version(transaction, record, versions)
            await transaction.upsert_records(SKILL_TABLE, [to_database_record(record)])
            state = await self._get_sync_state(transaction, record.skill_id)
            if state is None:
                state = SkillSyncStateRecord(skill_id=record.skill_id, created_at=now, updated_at=now)
                await transaction.upsert_records(SKILL_SYNC_STATE_TABLE, [to_database_record(state)])
            if pending_operation is not None:
                await self._insert_operation(transaction, _operation_record(pending_operation))
            return state

    async def get_version(self, version_id: str) -> SkillRecord:
        record = await self._get_version(self._database, version_id)
        if record is None:
            raise SkillNotFoundError(f"Skill version not found: {version_id}")
        return record

    async def get_latest_available_version(self, skill_ref: str) -> SkillRecord:
        skill_id = await self.resolve_skill_id(skill_ref)
        records, _ = await self._database.query_records(
            SKILL_TABLE,
            RecordQuery(
                filters=FilterGroup(
                    clauses=(
                        Predicate(field="skill_id", op="eq", value=skill_id),
                        Predicate(field="status", op="in", value=["draft", "published"]),
                    )
                ),
                sort=(Sort(field="created_at", direction="desc"), Sort(field="version_id", direction="desc")),
                page=Page(limit=1),
            ),
        )
        if not records:
            raise SkillNotFoundError(f"Skill has no available version: {skill_ref}")
        return from_database_record(records[0], SkillRecord)

    async def list_versions(self, skill_ref: str) -> list[SkillRecord]:
        return await self.query_versions(skill_id=await self.resolve_skill_id(skill_ref))

    async def query_versions(
        self,
        *,
        skill_id: str | None = None,
        alias: str | None = None,
        version_id: str | None = None,
        content_hash: str | None = None,
    ) -> list[SkillRecord]:
        return await self._query_versions(
            self._database,
            skill_id=skill_id,
            alias=alias,
            version_id=version_id,
            content_hash=content_hash,
        )

    async def find_snapshot_matches(self, local_snapshot_hash: str) -> list[SkillRecord]:
        records, _ = await self._database.query_records(
            SKILL_TABLE,
            RecordQuery(
                filters=Predicate(field="local_snapshot_hash", op="eq", value=local_snapshot_hash),
                sort=(Sort(field="created_at", direction="desc"),),
                page=Page(limit=100),
            ),
        )
        return [from_database_record(record, SkillRecord) for record in records]

    async def get_sync_state(self, skill_id: str) -> SkillSyncStateRecord:
        state = await self._get_sync_state(self._database, skill_id)
        if state is None:
            raise SkillNotFoundError(f"Skill family not found: {skill_id}")
        return state

    async def get_family_state(self, skill_id: str) -> SkillSyncStateRecord:
        return await self.get_sync_state(skill_id)

    async def list_sync_states(self) -> list[SkillSyncStateRecord]:
        records, _ = await self._database.query_records(
            SKILL_SYNC_STATE_TABLE,
            RecordQuery(sort=(Sort(field="created_at"),), page=Page(limit=10_000)),
        )
        return [from_database_record(record, SkillSyncStateRecord) for record in records]

    async def list_family_states(self) -> list[SkillSyncStateRecord]:
        return await self.list_sync_states()

    async def list_operations(
        self,
        *,
        skill_id: str | None = None,
        trajectory_id: str | None = None,
    ) -> list[SkillRemoteOperationRecord]:
        filters = []
        if skill_id is not None:
            filters.append(Predicate(field="skill_id", op="eq", value=skill_id))
        if trajectory_id is not None:
            filters.append(Predicate(field="trajectory_id", op="eq", value=trajectory_id))
        records, _ = await self._database.query_records(
            SKILL_REMOTE_OPERATION_TABLE,
            RecordQuery(
                filters=FilterGroup(clauses=tuple(filters)) if filters else None,
                sort=(Sort(field="created_at"), Sort(field="operation_id")),
                page=Page(limit=10_000),
            ),
        )
        return [from_database_record(record, SkillRemoteOperationRecord) for record in records]

    async def resolve_skill_id(self, skill_ref: str) -> str:
        direct = await self._get_sync_state(self._database, skill_ref)
        if direct is not None:
            return direct.skill_id
        matches = await self._query_versions(self._database, alias=skill_ref)
        skill_ids = {item.skill_id for item in matches}
        if len(skill_ids) == 1:
            return next(iter(skill_ids))
        if len(skill_ids) > 1:
            raise SkillConflictError(f"Skill alias is ambiguous: {skill_ref}")
        raise SkillNotFoundError(f"Skill family not found: {skill_ref}")

    async def get_cloud_skill_id(self, skill_id: str) -> str | None:
        versions = await self._query_versions(self._database, skill_id=skill_id)
        ids = {item.cloud_skill_id for item in versions if item.cloud_skill_id is not None}
        if len(ids) > 1:
            raise SkillConflictError(f"Skill family has conflicting cloud mappings: {skill_id}")
        return next(iter(ids), None)

    async def delete_family(self, skill_id: str) -> None:
        versions = await self._query_versions(self._database, skill_id=skill_id)
        operations = await self.list_operations(skill_id=skill_id)
        async with self._database.transaction() as transaction:
            await transaction.delete_records(SKILL_TABLE, DatabaseScope(), [item.version_id for item in versions])
            await transaction.delete_records(SKILL_SYNC_STATE_TABLE, DatabaseScope(), [skill_id])
            await transaction.delete_records(
                SKILL_REMOTE_OPERATION_TABLE,
                DatabaseScope(),
                [item.operation_id for item in operations],
            )

    async def begin_push(self, skill_ref: str, *, version_id: str | None, now):
        skill_id = await self.resolve_skill_id(skill_ref)
        version = (
            await self.get_version(version_id) if version_id else await self.get_latest_available_version(skill_id)
        )
        if version.skill_id != skill_id:
            raise SkillConflictError(f"version {version.version_id} does not belong to Skill {skill_id}")
        operation_id = push_operation_id(skill_id, version.version_id)
        request_hash = canonical_request_hash(
            {
                "skill_id": skill_id,
                "version_id": version.version_id,
                "content_hash": version.content_hash,
                "parent_version_ids": version.parent_version_ids,
                "runtime_type": version.runtime_type,
                "runtime_schema_version": version.runtime_schema_version,
                "runtime_metadata": version.runtime_metadata,
            }
        )
        async with self._database.transaction() as transaction:
            existing = await self._get_operation(transaction, operation_id)
            if existing is None:
                existing = SkillRemoteOperationRecord(
                    operation_id=operation_id,
                    operation_type=PendingSkillOperationType.PUSH_VERSION.value,
                    skill_id=skill_id,
                    cloud_skill_id=version.cloud_skill_id,
                    version_id=version.version_id,
                    request_hash=request_hash,
                    status=PendingSkillOperationStatus.PENDING.value,
                    created_at=now,
                    updated_at=now,
                )
                await transaction.upsert_records(SKILL_REMOTE_OPERATION_TABLE, [to_database_record(existing)])
            elif existing.request_hash != request_hash:
                raise SkillConflictError(f"push operation evidence changed: {operation_id}")
            if existing.status not in {"pending", "failed", "running"}:
                raise SkillConflictError(f"push operation is not retryable: {operation_id}")
            leased = existing.model_copy(
                update={
                    "status": "running",
                    "attempt_count": existing.attempt_count + 1,
                    "lease_expires_at": now + timedelta(seconds=60),
                    "next_retry_at": None,
                    "updated_at": now,
                }
            )
            await transaction.upsert_records(SKILL_REMOTE_OPERATION_TABLE, [to_database_record(leased)])
        return version, PendingSkillOperation.from_record(leased), version.cloud_skill_id

    async def mark_push_failed(self, skill_id: str, *, operation_id: str, error_code: str, now) -> None:
        operation = await self._get_operation(self._database, operation_id)
        if operation is None or operation.skill_id != skill_id:
            return
        await self._database.patch_record(
            SKILL_REMOTE_OPERATION_TABLE,
            DatabaseScope(),
            operation_id,
            {
                "status": "failed",
                "lease_expires_at": None,
                "last_error_code": error_code,
                "updated_at": now,
            },
        )

    async def complete_push(
        self,
        skill_id: str,
        *,
        version_id: str,
        operation_id: str,
        cloud_skill_id: str,
        remote_content_hash: str,
        remote_status: SkillVersionStatus,
        now,
        version_revision: int = 0,
        received_at=None,
    ) -> SkillSyncStateRecord:
        async with self._database.transaction() as transaction:
            version = await self._get_version(transaction, version_id)
            operation = await self._get_operation(transaction, operation_id)
            state = await self._get_sync_state(transaction, skill_id)
            if version is None or operation is None or state is None:
                raise SkillConflictError("local push commit evidence disappeared")
            if version.skill_id != skill_id or version.content_hash != remote_content_hash:
                raise SkillConflictError("remote push response conflicts with local immutable version")
            await transaction.patch_record(
                SKILL_TABLE,
                DatabaseScope(),
                version_id,
                {
                    "cloud_skill_id": cloud_skill_id,
                    "status": remote_status.value,
                    "version_revision": version_revision,
                    "updated_at": now,
                    "received_at": received_at or now,
                },
            )
            await transaction.patch_record(
                SKILL_REMOTE_OPERATION_TABLE,
                DatabaseScope(),
                operation_id,
                {
                    "cloud_skill_id": cloud_skill_id,
                    "status": "succeeded",
                    "lease_expires_at": None,
                    "last_error_code": None,
                    "remote_result": {
                        "cloud_skill_id": cloud_skill_id,
                        "version_id": version_id,
                        "content_hash": remote_content_hash,
                        "version_revision": version_revision,
                    },
                    "updated_at": now,
                },
            )
            return state

    async def import_cloud_versions(
        self,
        skill_id: str,
        *,
        cloud_skill_id: str,
        new_versions: list[SkillRecord],
        existing_updates: Mapping[str, tuple[SkillRecord, SkillRecord]],
        now=None,
    ) -> tuple[list[str], list[str]]:
        async with self._database.transaction() as transaction:
            current = await self._query_versions(transaction, skill_id=skill_id)
            current_by_id = {item.version_id: item for item in current}
            writes: list[SkillRecord] = []
            for version_id, (expected, updated) in existing_updates.items():
                if current_by_id.get(version_id) != expected:
                    raise SkillConflictError(f"local version changed while syncing: {version_id}")
                writes.append(updated)
                current_by_id[version_id] = updated
            inserted: list[str] = []
            matched: list[str] = []
            for version in new_versions:
                existing = current_by_id.get(version.version_id)
                if existing is not None:
                    if existing != version:
                        raise SkillConflictError(f"cloud version conflicts locally: {version.version_id}")
                    matched.append(version.version_id)
                    continue
                if version.skill_id != skill_id or version.cloud_skill_id != cloud_skill_id:
                    raise SkillConflictError("cloud version family identity is invalid")
                await self._validate_new_version(transaction, version, list(current_by_id.values()))
                writes.append(version)
                current_by_id[version.version_id] = version
                inserted.append(version.version_id)
            if writes:
                await transaction.upsert_records(SKILL_TABLE, [to_database_record(item) for item in writes])
            if now is not None:
                state = await self._get_sync_state(transaction, skill_id)
                if state is None:
                    raise SkillNotFoundError(f"Skill family not found: {skill_id}")
                await transaction.patch_record(
                    SKILL_SYNC_STATE_TABLE,
                    DatabaseScope(),
                    skill_id,
                    {"last_version_sync_at": now, "updated_at": now},
                )
            return inserted, matched

    async def update_trajectory_cursor(self, skill_id: str, *, cursor: str | None, now) -> None:
        state = await self.get_sync_state(skill_id)
        changed = await self._database.compare_and_swap_record(
            SKILL_SYNC_STATE_TABLE,
            DatabaseScope(),
            skill_id,
            expected={"trajectory_pull_cursor": state.trajectory_pull_cursor},
            changes={
                "trajectory_pull_cursor": cursor,
                "last_trajectory_pull_at": now,
                "updated_at": now,
            },
        )
        if not changed:
            raise SkillConflictError("trajectory pull cursor changed concurrently")

    async def _validate_new_version(self, transaction, record: SkillRecord, versions: list[SkillRecord]) -> None:
        alias_matches = await self._query_versions(transaction, alias=record.alias) if record.alias else []
        if any(item.skill_id != record.skill_id for item in alias_matches):
            raise SkillConflictError(f"Skill alias already exists: {record.alias}")
        if any(item.version_label == record.version_label for item in versions):
            raise SkillConflictError(f"version label already exists: {record.version_label}")
        parents = []
        for parent_id in record.parent_version_ids:
            parent = await self._get_version(transaction, parent_id)
            if parent is None:
                raise SkillConflictError(f"parent version not found: {parent_id}")
            if parent.skill_id != record.skill_id:
                raise SkillConflictError("all parents must belong to the same Skill family")
            parents.append(parent)
        if versions and not record.parent_version_ids:
            raise SkillConflictError("an existing Skill family cannot add a second root")

    async def _insert_operation(self, transaction, operation: SkillRemoteOperationRecord) -> None:
        existing = await self._get_operation(transaction, operation.operation_id)
        if existing is not None and existing != operation:
            raise SkillConflictError(f"remote operation already exists: {operation.operation_id}")
        await transaction.upsert_records(SKILL_REMOTE_OPERATION_TABLE, [to_database_record(operation)])

    @staticmethod
    async def _get_version(database, version_id: str) -> SkillRecord | None:
        records = await database.get_records(SKILL_TABLE, DatabaseScope(), [version_id])
        return from_database_record(records[0], SkillRecord) if records else None

    @staticmethod
    async def _get_sync_state(database, skill_id: str) -> SkillSyncStateRecord | None:
        records = await database.get_records(SKILL_SYNC_STATE_TABLE, DatabaseScope(), [skill_id])
        return from_database_record(records[0], SkillSyncStateRecord) if records else None

    @staticmethod
    async def _get_operation(database, operation_id: str) -> SkillRemoteOperationRecord | None:
        records = await database.get_records(SKILL_REMOTE_OPERATION_TABLE, DatabaseScope(), [operation_id])
        return from_database_record(records[0], SkillRemoteOperationRecord) if records else None

    @staticmethod
    async def _query_versions(
        database,
        *,
        skill_id=None,
        alias=None,
        version_id=None,
        content_hash=None,
    ) -> list[SkillRecord]:
        clauses = []
        for field, value in (
            ("skill_id", skill_id),
            ("alias", alias),
            ("version_id", version_id),
            ("content_hash", content_hash),
        ):
            if value is not None:
                clauses.append(Predicate(field=field, op="eq", value=value))
        records, cursor = await database.query_records(
            SKILL_TABLE,
            RecordQuery(
                filters=FilterGroup(clauses=tuple(clauses)) if clauses else None,
                sort=(Sort(field="created_at"), Sort(field="version_id")),
                page=Page(limit=10_000),
            ),
        )
        if cursor is not None:
            raise RuntimeError("local Skill version query exceeded the safety limit")
        return [from_database_record(record, SkillRecord) for record in records]


def _operation_record(operation: PendingSkillOperation) -> SkillRemoteOperationRecord:
    return SkillRemoteOperationRecord.model_validate(operation.model_dump(mode="json"))


__all__ = ["SkillRepository"]
