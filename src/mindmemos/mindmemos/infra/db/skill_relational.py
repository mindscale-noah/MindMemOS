"""Relational Skill v2 repository.

The repository owns the cloud projections defined by the edge-cloud protocol.
It uses the backend-neutral structured database capability from
``mindmemos_skill``; Qdrant payloads are intentionally absent from this module.
"""

from __future__ import annotations

import base64
import json
import uuid
from datetime import UTC, datetime
from typing import Any

from mindmemos_skill.contracts import (
    SkillBundle,
    SkillRemoteOperationStatus,
    SkillRemoteOperationType,
    SkillTrajectory,
    SkillTrajectoryReportRequest,
    SkillTrajectoryReportResult,
    SkillTrajectoryReportResultItem,
    SkillVersionCore,
    SkillVersionOrigin,
    SkillVersionStatus,
    TrajectorySanitizer,
    canonical_request_hash,
    parse_skill_bundle,
)
from mindmemos_skill.infra.database import (
    DatabaseScope,
    FieldSpec,
    FieldType,
    FilterGroup,
    IndexSpec,
    Page,
    Predicate,
    Record,
    RecordQuery,
    SchemaMigration,
    ScopedDatabase,
    Sort,
    TableRegistry,
    TableSpec,
)
from mindmemos_skill.skill_runtime import SkillRuntimeRegistry, build_default_skill_runtime_registry

from ...errors import SkillConflictError, SkillNotFoundError, SkillVersionNotFoundError

SKILL_VERSION_TABLE = "skill_versions"
SKILL_OPERATION_TABLE = "skill_operations"
SKILL_TRAJECTORY_TABLE = "skill_trajectories"
SKILL_TRAJECTORY_BINDING_TABLE = "skill_trajectory_bindings"


def build_cloud_skill_tables() -> TableRegistry:
    """Build the four relation-table projections from the data dictionary."""

    registry = TableRegistry(
        (
            TableSpec(
                name=SKILL_VERSION_TABLE,
                primary_key="version_id",
                fields=(
                    _text("project_id", nullable=False),
                    _text("cloud_skill_id", nullable=False),
                    _text("version_id", nullable=False),
                    _json("parent_version_ids", nullable=False, default=[]),
                    _text("name", nullable=False),
                    _text("bundle", nullable=False),
                    _text("content_hash", nullable=False),
                    _text("runtime_type", nullable=False, default="static"),
                    _integer("runtime_schema_version", nullable=False, default=1),
                    _json("runtime_metadata", nullable=False, default={}),
                    _text("version_label", nullable=False),
                    _text("commit_message"),
                    _text("status", nullable=False, default=SkillVersionStatus.DRAFT.value),
                    _integer("version_revision", nullable=False, default=0),
                    _text("origin", nullable=False),
                    _json("metadata", nullable=False, default={}),
                    _datetime("created_at", nullable=False),
                    _datetime("updated_at", nullable=False),
                    _datetime("received_at", nullable=False),
                ),
                indexes=(
                    IndexSpec(
                        name="skill_versions_label_uq",
                        fields=("cloud_skill_id", "version_label"),
                        unique=True,
                    ),
                    IndexSpec(
                        name="skill_versions_latest_idx",
                        fields=("cloud_skill_id", "status", "created_at", "version_id"),
                    ),
                    IndexSpec(name="skill_versions_hash_idx", fields=("content_hash",)),
                ),
            ),
            TableSpec(
                name=SKILL_OPERATION_TABLE,
                primary_key="operation_id",
                fields=(
                    _text("project_id", nullable=False),
                    _text("operation_id", nullable=False),
                    _text("cloud_skill_id"),
                    _text("operation_type", nullable=False),
                    _text("request_hash", nullable=False),
                    _json("request_payload"),
                    _text("status", nullable=False),
                    _json("result"),
                    _text("error_code"),
                    _datetime("created_at", nullable=False),
                    _datetime("updated_at", nullable=False),
                ),
                indexes=(
                    IndexSpec(
                        name="skill_operations_status_idx",
                        fields=("operation_type", "status", "updated_at"),
                    ),
                ),
            ),
            TableSpec(
                name=SKILL_TRAJECTORY_TABLE,
                primary_key="trajectory_id",
                fields=(
                    _text("project_id", nullable=False),
                    _text("trajectory_id", nullable=False),
                    _text("trajectory_hash", nullable=False),
                    _text("task_id", nullable=False),
                    _text("rollout_id", nullable=False),
                    _integer("attempt_no", nullable=False, default=0),
                    _text("rollout_type", nullable=False),
                    _text("task_instruction", nullable=False),
                    _text("task_system_prompt"),
                    _json("task_tags", nullable=False, default=[]),
                    _json("task_metadata", nullable=False, default={}),
                    _text("env_ref", nullable=False, default="unknown"),
                    _json("env_metadata", nullable=False, default={}),
                    _text("agent_type", nullable=False),
                    _json("agent_profile", nullable=False, default={}),
                    _text("status", nullable=False),
                    _json("trajectory", nullable=False, default=[]),
                    _json("skill_bindings", nullable=False, default=[]),
                    _float("reward_score"),
                    _text("reward_detail"),
                    _json("reward_metadata", nullable=False, default={}),
                    _datetime("started_at", nullable=False),
                    _datetime("finished_at", nullable=False),
                    _integer("n_turn", nullable=False, default=0),
                    _text("error_info"),
                    _json("metadata", nullable=False, default={}),
                    _integer("metadata_revision", nullable=False, default=0),
                    _datetime("metadata_updated_at"),
                    _text("source", nullable=False),
                    _text("source_add_record_id"),
                    _datetime("created_at", nullable=False),
                    _datetime("received_at", nullable=False),
                ),
                indexes=(
                    IndexSpec(
                        name="skill_trajectories_rollout_attempt_uq",
                        fields=("rollout_id", "attempt_no"),
                        unique=True,
                    ),
                    IndexSpec(name="skill_trajectories_task_idx", fields=("task_id",)),
                    IndexSpec(
                        name="skill_trajectories_incremental_idx",
                        fields=("metadata_updated_at", "received_at", "trajectory_id"),
                    ),
                ),
            ),
            TableSpec(
                name=SKILL_TRAJECTORY_BINDING_TABLE,
                primary_key="binding_id",
                fields=(
                    _text("project_id", nullable=False),
                    _text("binding_id", nullable=False),
                    _text("trajectory_id", nullable=False),
                    _integer("binding_no", nullable=False),
                    _text("cloud_skill_id", nullable=False),
                    _text("version_id", nullable=False),
                    _text("base_version_id"),
                    _text("name", nullable=False),
                    _text("content_hash", nullable=False),
                    _text("version_label"),
                    _text("usage", nullable=False),
                    _text("injection_mode"),
                ),
                indexes=(
                    IndexSpec(
                        name="skill_trajectory_bindings_no_uq",
                        fields=("trajectory_id", "binding_no"),
                        unique=True,
                    ),
                    IndexSpec(
                        name="skill_trajectory_bindings_usage_uq",
                        fields=("trajectory_id", "cloud_skill_id", "version_id", "usage"),
                        unique=True,
                    ),
                    IndexSpec(
                        name="skill_trajectory_bindings_version_idx",
                        fields=("cloud_skill_id", "version_id", "trajectory_id"),
                    ),
                ),
            ),
        ),
        migrations=(
            SchemaMigration(
                namespace="cloud-skill-v2",
                version=1,
                name="create_unified_skill_relations",
                tables=(
                    SKILL_VERSION_TABLE,
                    SKILL_OPERATION_TABLE,
                    SKILL_TRAJECTORY_TABLE,
                    SKILL_TRAJECTORY_BINDING_TABLE,
                ),
            ),
        ),
    )
    registry.freeze()
    return registry


class SkillRelationalRepository:
    """Transactional project-scoped cloud Skill facts and operation ledger."""

    def __init__(
        self,
        database: ScopedDatabase,
        *,
        clock=None,
        id_generator=None,
        sanitizer: TrajectorySanitizer | None = None,
        runtime_registry: SkillRuntimeRegistry | None = None,
    ) -> None:
        self._database = database
        self._clock = clock or (lambda: datetime.now(UTC))
        self._id_generator = id_generator or (lambda: str(uuid.uuid4()))
        self._sanitizer = sanitizer or TrajectorySanitizer()
        self._runtime_registry = runtime_registry or build_default_skill_runtime_registry()

    async def ensure_schema(self) -> None:
        await self._database.ensure_schema(build_cloud_skill_tables())

    async def close(self) -> None:
        await self._database.close()

    async def create_version(
        self,
        *,
        project_id: str,
        operation_id: str,
        version: SkillVersionCore,
        bundle: SkillBundle | str | dict[str, Any],
        operation_type: SkillRemoteOperationType = SkillRemoteOperationType.PUSH_VERSION,
    ) -> SkillVersionCore:
        canonical_bundle = parse_skill_bundle(bundle)
        self._runtime_registry.validate_spec(
            runtime_type=version.runtime_type,
            schema_version=version.runtime_schema_version,
            metadata=version.runtime_metadata,
        )
        if canonical_bundle.content_hash != version.content_hash:
            raise SkillConflictError("Skill bundle hash does not match version content_hash")
        request_payload = {
            "operation_type": operation_type.value,
            "version": version.model_dump(mode="json"),
            "bundle": canonical_bundle.canonical_dict(),
        }
        request_hash = canonical_request_hash(request_payload)
        now = self._clock()
        scope = _scope(project_id)
        async with self._database.transaction() as transaction:
            replay = await _begin_operation(
                transaction,
                scope=scope,
                project_id=project_id,
                operation_id=operation_id,
                operation_type=operation_type,
                request_hash=request_hash,
                request_payload=request_payload,
                cloud_skill_id=version.cloud_skill_id,
                now=now,
            )
            if replay is not None:
                result = replay.get("result") or {}
                result_version_id = result.get("version_id")
                if isinstance(result_version_id, str):
                    stored = await transaction.get_records(SKILL_VERSION_TABLE, scope, [result_version_id])
                    if stored:
                        return _version_from_payload(stored[0].payload)

            resolved_cloud_skill_id = version.cloud_skill_id
            parents: list[SkillVersionCore] = []
            if version.parent_version_ids:
                parent_records = await transaction.get_records(
                    SKILL_VERSION_TABLE,
                    scope,
                    version.parent_version_ids,
                )
                if len(parent_records) != len(version.parent_version_ids):
                    found = {record.record_id for record in parent_records}
                    missing = [item for item in version.parent_version_ids if item not in found]
                    raise SkillVersionNotFoundError(f"parent versions not found: {', '.join(missing)}")
                by_id = {record.record_id: _version_from_payload(record.payload) for record in parent_records}
                parents = [by_id[parent_id] for parent_id in version.parent_version_ids]
                family_ids = {parent.cloud_skill_id for parent in parents}
                if len(family_ids) != 1:
                    raise SkillConflictError("all parent versions must belong to one cloud Skill family")
                parent_family = next(iter(family_ids))
                if resolved_cloud_skill_id is not None and resolved_cloud_skill_id != parent_family:
                    raise SkillConflictError("version cloud_skill_id differs from its parent family")
                resolved_cloud_skill_id = parent_family
            elif resolved_cloud_skill_id is None:
                resolved_cloud_skill_id = self._id_generator()
            assert resolved_cloud_skill_id is not None

            if operation_type is SkillRemoteOperationType.MERGE:
                if len(version.parent_version_ids) < 2 or version.origin is not SkillVersionOrigin.MERGE:
                    raise SkillConflictError("merge requires at least two parents and origin=merge")
            elif len(version.parent_version_ids) > 1:
                raise SkillConflictError("multi-parent versions must use the merge operation")

            existing = await transaction.get_records(SKILL_VERSION_TABLE, scope, [version.version_id])
            received_at = version.received_at or now
            stored_version = version.model_copy(
                update={
                    "cloud_skill_id": resolved_cloud_skill_id,
                    "received_at": received_at,
                }
            )
            if existing:
                current = _version_from_payload(existing[0].payload)
                current_bundle = existing[0].payload["bundle"]
                if current != stored_version or current_bundle != canonical_bundle.canonical_json():
                    raise SkillConflictError(f"version_id already has different immutable facts: {version.version_id}")
            else:
                label_records, _ = await transaction.query_records(
                    SKILL_VERSION_TABLE,
                    RecordQuery(
                        scope=scope,
                        filters=FilterGroup(
                            clauses=(
                                Predicate(field="cloud_skill_id", op="eq", value=resolved_cloud_skill_id),
                                Predicate(field="version_label", op="eq", value=version.version_label),
                            )
                        ),
                        page=Page(limit=1),
                    ),
                )
                if label_records:
                    raise SkillConflictError(
                        f"version label already exists in cloud Skill family: {version.version_label}"
                    )
                await transaction.upsert_records(
                    SKILL_VERSION_TABLE,
                    [_version_record(project_id, stored_version, canonical_bundle.canonical_json())],
                )
            await _complete_operation(
                transaction,
                scope=scope,
                operation_id=operation_id,
                cloud_skill_id=resolved_cloud_skill_id,
                result={
                    "cloud_skill_id": resolved_cloud_skill_id,
                    "version_id": version.version_id,
                    "content_hash": version.content_hash,
                },
                now=now,
            )
            return stored_version

    async def get_version(self, project_id: str, version_id: str) -> SkillVersionCore:
        records = await self._database.get_records(SKILL_VERSION_TABLE, _scope(project_id), [version_id])
        if not records:
            raise SkillVersionNotFoundError(f"Skill version not found: {version_id}")
        return _version_from_payload(records[0].payload)

    async def get_bundle(self, project_id: str, version_id: str) -> SkillBundle:
        records = await self._database.get_records(SKILL_VERSION_TABLE, _scope(project_id), [version_id])
        if not records:
            raise SkillVersionNotFoundError(f"Skill version not found: {version_id}")
        return parse_skill_bundle(str(records[0].payload["bundle"]))

    async def list_versions(self, project_id: str, cloud_skill_id: str) -> list[SkillVersionCore]:
        records, cursor = await self._database.query_records(
            SKILL_VERSION_TABLE,
            RecordQuery(
                scope=_scope(project_id),
                filters=Predicate(field="cloud_skill_id", op="eq", value=cloud_skill_id),
                sort=(Sort(field="created_at"), Sort(field="version_id")),
                page=Page(limit=500),
            ),
        )
        if cursor is not None:
            raise RuntimeError("cloud Skill version families larger than 500 require paged repository reads")
        if not records:
            raise SkillNotFoundError(f"cloud Skill not found: {cloud_skill_id}")
        return [_version_from_payload(record.payload) for record in records]

    async def latest_available_version(self, project_id: str, cloud_skill_id: str) -> SkillVersionCore:
        records, _ = await self._database.query_records(
            SKILL_VERSION_TABLE,
            RecordQuery(
                scope=_scope(project_id),
                filters=FilterGroup(
                    clauses=(
                        Predicate(field="cloud_skill_id", op="eq", value=cloud_skill_id),
                        Predicate(
                            field="status",
                            op="in",
                            value=[SkillVersionStatus.DRAFT.value, SkillVersionStatus.PUBLISHED.value],
                        ),
                    )
                ),
                sort=(Sort(field="created_at", direction="desc"), Sort(field="version_id", direction="desc")),
                page=Page(limit=1),
            ),
        )
        if not records:
            raise SkillNotFoundError(f"cloud Skill has no available version: {cloud_skill_id}")
        return _version_from_payload(records[0].payload)

    async def list_latest_available_versions(self, project_id: str) -> list[SkillVersionCore]:
        records, cursor = await self._database.query_records(
            SKILL_VERSION_TABLE,
            RecordQuery(
                scope=_scope(project_id),
                filters=Predicate(
                    field="status",
                    op="in",
                    value=[SkillVersionStatus.DRAFT.value, SkillVersionStatus.PUBLISHED.value],
                ),
                sort=(Sort(field="created_at", direction="desc"), Sort(field="version_id", direction="desc")),
                page=Page(limit=10_000),
            ),
        )
        if cursor is not None:
            raise RuntimeError("project Skill version query exceeded the repository safety limit")
        latest: dict[str, SkillVersionCore] = {}
        for record in records:
            version = _version_from_payload(record.payload)
            assert version.cloud_skill_id is not None
            latest.setdefault(version.cloud_skill_id, version)
        return list(latest.values())

    async def update_status(
        self,
        *,
        project_id: str,
        version_id: str,
        status: SkillVersionStatus,
        expected_revision: int,
    ) -> SkillVersionCore:
        current = await self.get_version(project_id, version_id)
        allowed = {
            SkillVersionStatus.DRAFT: {
                SkillVersionStatus.PUBLISHED,
                SkillVersionStatus.REJECTED,
                SkillVersionStatus.ARCHIVED,
            },
            SkillVersionStatus.PUBLISHED: {SkillVersionStatus.ARCHIVED},
        }
        if current.version_revision != expected_revision:
            raise SkillConflictError("Skill version revision conflict")
        if status not in allowed.get(current.status, set()):
            raise SkillConflictError(f"invalid Skill version status transition: {current.status} -> {status}")
        now = self._clock()
        changed = await self._database.compare_and_swap_record(
            SKILL_VERSION_TABLE,
            _scope(project_id),
            version_id,
            expected={"version_revision": expected_revision, "status": current.status.value},
            changes={
                "status": status.value,
                "version_revision": expected_revision + 1,
                "updated_at": now,
            },
        )
        if not changed:
            raise SkillConflictError("Skill version revision conflict")
        return current.model_copy(
            update={"status": status, "version_revision": expected_revision + 1, "updated_at": now}
        )

    async def sync_versions(
        self,
        *,
        project_id: str,
        cloud_skill_id: str,
        known_version_revisions: dict[str, int],
    ) -> list[SkillVersionCore]:
        versions = await self.list_versions(project_id, cloud_skill_id)
        return [
            version
            for version in versions
            if known_version_revisions.get(version.version_id) != version.version_revision
        ]

    async def ingest_trajectories(
        self,
        *,
        project_id: str,
        request: SkillTrajectoryReportRequest,
    ) -> SkillTrajectoryReportResult:
        request_hash = canonical_request_hash(request)
        request_payload = request.model_dump(mode="json")
        now = self._clock()
        scope = _scope(project_id)
        async with self._database.transaction() as transaction:
            replay = await _begin_operation(
                transaction,
                scope=scope,
                project_id=project_id,
                operation_id=request.operation_id,
                operation_type=SkillRemoteOperationType.REPORT_TRAJECTORY,
                request_hash=request_hash,
                request_payload=request_payload,
                cloud_skill_id=None,
                now=now,
            )
            if replay is not None and replay.get("status") == SkillRemoteOperationStatus.SUCCEEDED.value:
                return SkillTrajectoryReportResult.model_validate(replay.get("result") or {})

            results: list[SkillTrajectoryReportResultItem] = []
            stored_trajectories: list[Record] = []
            stored_bindings: list[Record] = []
            for item in request.items:
                upload = self._sanitizer.validate(item.trajectory)
                existing = await transaction.get_records(
                    SKILL_TRAJECTORY_TABLE,
                    scope,
                    [upload.trajectory_id],
                )
                if existing:
                    if existing[0].payload["trajectory_hash"] != upload.trajectory_hash.removeprefix("sha256:"):
                        raise SkillConflictError(
                            f"trajectory_id already has different immutable facts: {upload.trajectory_id}"
                        )
                    results.append(
                        SkillTrajectoryReportResultItem(trajectory_id=upload.trajectory_id, status="duplicate")
                    )
                    continue

                version_records = await transaction.get_records(
                    SKILL_VERSION_TABLE,
                    scope,
                    [binding.version_id for binding in upload.skill_bindings],
                )
                versions = {record.record_id: _version_from_payload(record.payload) for record in version_records}
                for binding in upload.skill_bindings:
                    version = versions.get(binding.version_id)
                    if version is None:
                        raise SkillVersionNotFoundError(f"binding version not found: {binding.version_id}")
                    if version.cloud_skill_id != binding.cloud_skill_id:
                        raise SkillConflictError(
                            f"binding version does not belong to family: {binding.version_id}/{binding.cloud_skill_id}"
                        )
                    if version.content_hash != binding.content_hash:
                        raise SkillConflictError(f"binding content hash differs from version: {binding.version_id}")

                trajectory = SkillTrajectory.model_validate(
                    {
                        **upload.model_dump(mode="json"),
                        "metadata_revision": 0,
                        "metadata_updated_at": None,
                        "received_at": now,
                    }
                )
                stored_trajectories.append(_trajectory_record(project_id, trajectory))
                stored_bindings.extend(_binding_records(project_id, trajectory))
                results.append(SkillTrajectoryReportResultItem(trajectory_id=upload.trajectory_id, status="stored"))
            await transaction.upsert_records(SKILL_TRAJECTORY_TABLE, stored_trajectories)
            await transaction.upsert_records(SKILL_TRAJECTORY_BINDING_TABLE, stored_bindings)
            result = SkillTrajectoryReportResult(items=results)
            await _complete_operation(
                transaction,
                scope=scope,
                operation_id=request.operation_id,
                cloud_skill_id=None,
                result=result.model_dump(mode="json"),
                now=now,
            )
            return result

    async def enqueue_trajectories(
        self,
        *,
        project_id: str,
        request: SkillTrajectoryReportRequest,
    ) -> SkillTrajectoryReportResult:
        """Persist a recoverable async ingest operation before queue dispatch."""

        if request.mode != "async":
            raise ValueError("only async trajectory reports may be queued")
        for item in request.items:
            self._sanitizer.validate(item.trajectory)
        request_hash = canonical_request_hash(request)
        request_payload = request.model_dump(mode="json")
        now = self._clock()
        scope = _scope(project_id)
        async with self._database.transaction() as transaction:
            existing = await transaction.get_records(SKILL_OPERATION_TABLE, scope, [request.operation_id])
            if existing:
                operation = dict(existing[0].payload)
                if (
                    operation["request_hash"] != request_hash
                    or operation["operation_type"] != SkillRemoteOperationType.REPORT_TRAJECTORY.value
                ):
                    raise SkillConflictError(
                        f"operation_id already belongs to different inputs: {request.operation_id}"
                    )
                return SkillTrajectoryReportResult.model_validate(operation.get("result") or {})
            result = SkillTrajectoryReportResult(
                items=[
                    SkillTrajectoryReportResultItem(
                        trajectory_id=item.trajectory.trajectory_id,
                        status="queued",
                    )
                    for item in request.items
                ]
            )
            await transaction.upsert_records(
                SKILL_OPERATION_TABLE,
                [
                    Record(
                        table=SKILL_OPERATION_TABLE,
                        record_id=request.operation_id,
                        scope=scope,
                        payload={
                            "project_id": project_id,
                            "operation_id": request.operation_id,
                            "cloud_skill_id": None,
                            "operation_type": SkillRemoteOperationType.REPORT_TRAJECTORY.value,
                            "request_hash": request_hash,
                            "request_payload": request_payload,
                            "status": SkillRemoteOperationStatus.QUEUED.value,
                            "result": result.model_dump(mode="json"),
                            "error_code": None,
                            "created_at": now,
                            "updated_at": now,
                        },
                    )
                ],
            )
            return result

    async def resume_trajectory_ingest(
        self,
        *,
        project_id: str,
        operation_id: str,
    ) -> SkillTrajectoryReportResult:
        """Replay one durable queued trajectory operation from its frozen payload."""

        operation = await self.get_operation_payload(project_id, operation_id)
        if operation["operation_type"] != SkillRemoteOperationType.REPORT_TRAJECTORY.value:
            raise SkillConflictError(f"operation is not a trajectory report: {operation_id}")
        request = SkillTrajectoryReportRequest.model_validate(operation.get("request_payload") or {})
        return await self.ingest_trajectories(project_id=project_id, request=request)

    async def get_operation_payload(self, project_id: str, operation_id: str) -> dict[str, Any]:
        records = await self._database.get_records(
            SKILL_OPERATION_TABLE,
            _scope(project_id),
            [operation_id],
        )
        if not records:
            raise SkillNotFoundError(f"Skill operation not found: {operation_id}")
        return dict(records[0].payload)

    async def list_queued_operations(self, project_id: str, *, limit: int = 100) -> list[dict[str, Any]]:
        """Return durable queue facts for an operation sweeper to redispatch."""

        records, _ = await self._database.query_records(
            SKILL_OPERATION_TABLE,
            RecordQuery(
                scope=_scope(project_id),
                filters=Predicate(field="status", op="eq", value=SkillRemoteOperationStatus.QUEUED.value),
                sort=(Sort(field="updated_at"), Sort(field="operation_id")),
                page=Page(limit=limit),
            ),
        )
        return [dict(record.payload) for record in records]

    async def prepare_evolution(
        self,
        *,
        project_id: str,
        operation_id: str,
        cloud_skill_id: str,
        base_version_id: str,
        algorithm: str,
        mode: str,
        reuse_evidence: bool,
        trajectory_ids: list[str] | None,
    ) -> tuple[str, SkillVersionCore, SkillBundle, list[SkillTrajectory], dict[str, Any] | None]:
        """Validate and freeze an evolution run before algorithm execution or enqueue."""

        base = await self.get_version(project_id, base_version_id)
        if base.cloud_skill_id != cloud_skill_id:
            raise SkillConflictError("evolution base version does not belong to cloud Skill family")
        bundle = await self.get_bundle(project_id, base_version_id)
        if trajectory_ids is None:
            evidence, _, _ = await self.list_trajectories(
                project_id=project_id,
                cloud_skill_id=cloud_skill_id,
                version_id=base_version_id,
                limit=500,
            )
        else:
            evidence = [await self.get_trajectory(project_id, item) for item in trajectory_ids]
            for trajectory in evidence:
                if not any(
                    binding.cloud_skill_id == cloud_skill_id and binding.version_id == base_version_id
                    for binding in trajectory.skill_bindings
                ):
                    raise SkillConflictError(
                        f"trajectory is not bound to the requested base version: {trajectory.trajectory_id}"
                    )
        frozen_ids = [item.trajectory_id for item in evidence]
        request_facts = {
            "operation_id": operation_id,
            "cloud_skill_id": cloud_skill_id,
            "base_version_id": base_version_id,
            "algorithm": algorithm,
            "mode": mode,
            "reuse_evidence": reuse_evidence,
            "trajectory_ids": trajectory_ids,
        }
        request_hash = canonical_request_hash(request_facts)
        run_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"mindmemos:evolution:{project_id}:{operation_id}"))
        now = self._clock()
        scope = _scope(project_id)
        async with self._database.transaction() as transaction:
            existing = await transaction.get_records(SKILL_OPERATION_TABLE, scope, [operation_id])
            if existing:
                payload = dict(existing[0].payload)
                if (
                    payload["operation_type"] != SkillRemoteOperationType.EVOLVE.value
                    or payload["request_hash"] != request_hash
                ):
                    raise SkillConflictError(f"operation_id already belongs to different inputs: {operation_id}")
                return run_id, base, bundle, evidence, payload.get("result")
            frozen_payload = {
                **request_facts,
                "evolution_run_id": run_id,
                "selected_trajectory_ids": frozen_ids,
                "submitted_at": now.isoformat(),
            }
            await transaction.upsert_records(
                SKILL_OPERATION_TABLE,
                [
                    Record(
                        table=SKILL_OPERATION_TABLE,
                        record_id=operation_id,
                        scope=scope,
                        payload={
                            "project_id": project_id,
                            "operation_id": operation_id,
                            "cloud_skill_id": cloud_skill_id,
                            "operation_type": SkillRemoteOperationType.EVOLVE.value,
                            "request_hash": request_hash,
                            "request_payload": frozen_payload,
                            "status": (
                                SkillRemoteOperationStatus.QUEUED.value
                                if mode == "async"
                                else SkillRemoteOperationStatus.RUNNING.value
                            ),
                            "result": {
                                "operation_id": operation_id,
                                "evolution_run_id": run_id,
                                "cloud_skill_id": cloud_skill_id,
                                "base_version_id": base_version_id,
                                "status": "queued" if mode == "async" else "running",
                                "candidate_version_ids": [],
                                "selected_version_id": None,
                                "evidence": frozen_ids,
                            },
                            "error_code": None,
                            "created_at": now,
                            "updated_at": now,
                        },
                    )
                ],
            )
        return run_id, base, bundle, evidence, None

    async def finish_evolution_without_version(
        self,
        *,
        project_id: str,
        operation_id: str,
        result: dict[str, Any],
        failed: bool = False,
        error_code: str | None = None,
    ) -> None:
        await self._database.patch_record(
            SKILL_OPERATION_TABLE,
            _scope(project_id),
            operation_id,
            {
                "status": (
                    SkillRemoteOperationStatus.FAILED.value if failed else SkillRemoteOperationStatus.NO_CHANGE.value
                ),
                "result": result,
                "error_code": error_code,
                "updated_at": self._clock(),
            },
        )

    async def complete_evolution_candidates(
        self,
        *,
        project_id: str,
        operation_id: str,
        evolution_run_id: str,
        base: SkillVersionCore,
        algorithm: str,
        evidence: list[SkillTrajectory],
        candidates: list[SkillBundle],
        selected_index: int = 0,
    ) -> list[SkillVersionCore]:
        """Atomically persist candidate versions, immutable provenance and operation result."""

        unique_candidates: list[SkillBundle] = []
        seen_hashes = {base.content_hash}
        for candidate in candidates:
            if candidate.content_hash not in seen_hashes:
                unique_candidates.append(candidate)
                seen_hashes.add(candidate.content_hash)
        if not unique_candidates:
            result = {
                "operation_id": operation_id,
                "evolution_run_id": evolution_run_id,
                "cloud_skill_id": base.cloud_skill_id,
                "base_version_id": base.version_id,
                "status": "no_change",
                "candidate_version_ids": [],
                "selected_version_id": None,
                "evidence": [item.trajectory_id for item in evidence],
            }
            await self.finish_evolution_without_version(
                project_id=project_id,
                operation_id=operation_id,
                result=result,
            )
            return []
        if not 0 <= selected_index < len(unique_candidates):
            raise ValueError("selected evolution candidate index is out of range")
        major, minor, patch = _parse_version_label(base.version_label)
        now = self._clock()
        stored: list[SkillVersionCore] = []
        for index, candidate in enumerate(unique_candidates):
            version_id = self._id_generator()
            evidence_facts = [
                {
                    "trajectory_id": item.trajectory_id,
                    "trajectory_hash": item.trajectory_hash,
                    "source_version_id": base.version_id,
                }
                for item in evidence
            ]
            stored.append(
                SkillVersionCore(
                    version_id=version_id,
                    cloud_skill_id=base.cloud_skill_id,
                    parent_version_ids=[base.version_id],
                    name=base.name,
                    content_hash=candidate.content_hash,
                    version_label=f"{major}.{minor}.{patch + index + 1}",
                    commit_message=f"evolve {algorithm} from {base.version_id}",
                    status=SkillVersionStatus.DRAFT,
                    origin=SkillVersionOrigin.EVOLUTION,
                    runtime_type=base.runtime_type,
                    runtime_schema_version=base.runtime_schema_version,
                    runtime_metadata=base.runtime_metadata,
                    metadata={
                        "evolution": {
                            "operation_id": operation_id,
                            "evolution_run_id": evolution_run_id,
                            "algorithm": algorithm,
                            "base_version_id": base.version_id,
                            "evidence": evidence_facts,
                            "candidate_index": index,
                            "selected": index == selected_index,
                            "status": "succeeded",
                        }
                    },
                    created_at=now,
                    updated_at=now,
                    received_at=now,
                )
            )
        scope = _scope(project_id)
        result = {
            "operation_id": operation_id,
            "evolution_run_id": evolution_run_id,
            "cloud_skill_id": base.cloud_skill_id,
            "base_version_id": base.version_id,
            "status": "succeeded",
            "candidate_version_ids": [item.version_id for item in stored],
            "selected_version_id": stored[selected_index].version_id,
            "evidence": [item.trajectory_id for item in evidence],
        }
        async with self._database.transaction() as transaction:
            operation = await transaction.get_records(SKILL_OPERATION_TABLE, scope, [operation_id])
            if not operation or operation[0].payload["operation_type"] != SkillRemoteOperationType.EVOLVE.value:
                raise SkillConflictError("evolution operation ledger entry is missing")
            existing_records, _ = await transaction.query_records(
                SKILL_VERSION_TABLE,
                RecordQuery(
                    scope=scope,
                    filters=Predicate(field="cloud_skill_id", op="eq", value=base.cloud_skill_id),
                    page=Page(limit=500),
                ),
            )
            existing_labels = {_version_from_payload(record.payload).version_label for record in existing_records}
            if existing_labels & {item.version_label for item in stored}:
                raise SkillConflictError("evolution candidate version label already exists")
            await transaction.upsert_records(
                SKILL_VERSION_TABLE,
                [
                    _version_record(project_id, version, candidate.canonical_json())
                    for version, candidate in zip(stored, unique_candidates, strict=True)
                ],
            )
            await transaction.patch_record(
                SKILL_OPERATION_TABLE,
                scope,
                operation_id,
                {
                    "status": SkillRemoteOperationStatus.SUCCEEDED.value,
                    "result": result,
                    "error_code": None,
                    "updated_at": now,
                },
            )
        return stored

    async def get_trajectory(self, project_id: str, trajectory_id: str) -> SkillTrajectory:
        records = await self._database.get_records(
            SKILL_TRAJECTORY_TABLE,
            _scope(project_id),
            [trajectory_id],
        )
        if not records:
            raise SkillNotFoundError(f"Skill trajectory not found: {trajectory_id}")
        return _trajectory_from_payload(records[0].payload)

    async def list_trajectories(
        self,
        *,
        project_id: str,
        cloud_skill_id: str,
        version_id: str | None = None,
        since: datetime | None = None,
        cursor: str | None = None,
        limit: int = 100,
        status: str | None = None,
        min_score: float | None = None,
    ) -> tuple[list[SkillTrajectory], str | None, bool]:
        if not 1 <= limit <= 500:
            raise ValueError("trajectory page limit must be between 1 and 500")
        query_identity = {
            "project_id": project_id,
            "cloud_skill_id": cloud_skill_id,
            "version_id": version_id,
            "since": since.isoformat() if since else None,
            "status": status,
            "min_score": min_score,
        }
        fingerprint = canonical_request_hash(query_identity)
        after = _decode_trajectory_cursor(cursor, fingerprint)
        binding_filters: list[Predicate] = [Predicate(field="cloud_skill_id", op="eq", value=cloud_skill_id)]
        if version_id is not None:
            version = await self.get_version(project_id, version_id)
            if version.cloud_skill_id != cloud_skill_id:
                raise SkillConflictError("trajectory version filter does not belong to family")
            binding_filters.append(Predicate(field="version_id", op="eq", value=version_id))
        bindings, cursor_more = await self._database.query_records(
            SKILL_TRAJECTORY_BINDING_TABLE,
            RecordQuery(
                scope=_scope(project_id),
                filters=FilterGroup(clauses=tuple(binding_filters)),
                page=Page(limit=10_000),
            ),
        )
        if cursor_more is not None:
            raise RuntimeError("trajectory binding query exceeded the repository safety limit")
        trajectory_ids = sorted({str(record.payload["trajectory_id"]) for record in bindings})
        records = await self._database.get_records(SKILL_TRAJECTORY_TABLE, _scope(project_id), trajectory_ids)
        trajectories = [_trajectory_from_payload(record.payload) for record in records]
        filtered: list[SkillTrajectory] = []
        for trajectory in trajectories:
            effective_time = trajectory.metadata_updated_at or trajectory.received_at
            if effective_time is None:
                continue
            if since is not None and effective_time < since:
                continue
            if status is not None and trajectory.status.value != status:
                continue
            if min_score is not None and (trajectory.reward_score is None or trajectory.reward_score < min_score):
                continue
            key = (effective_time.isoformat(), trajectory.trajectory_id)
            if after is not None and key <= after:
                continue
            filtered.append(trajectory)
        filtered.sort(key=lambda item: ((item.metadata_updated_at or item.received_at), item.trajectory_id))
        has_more = len(filtered) > limit
        items = filtered[:limit]
        next_cursor = None
        if has_more and items:
            last = items[-1]
            last_time = last.metadata_updated_at or last.received_at
            assert last_time is not None
            next_cursor = _encode_trajectory_cursor(fingerprint, last_time.isoformat(), last.trajectory_id)
        return items, next_cursor, has_more


def _scope(project_id: str) -> DatabaseScope:
    if not project_id:
        raise ValueError("project_id must not be empty")
    return DatabaseScope(project_id=project_id)


def _parse_version_label(value: str) -> tuple[int, int, int]:
    parts = value.split(".")
    if len(parts) != 3 or any(not part.isdigit() for part in parts):
        raise SkillConflictError(f"invalid semantic version label: {value}")
    return tuple(int(part) for part in parts)  # type: ignore[return-value]


def _version_record(project_id: str, version: SkillVersionCore, bundle: str) -> Record:
    payload = version.model_dump(mode="json")
    payload["project_id"] = project_id
    payload["bundle"] = bundle
    return Record(
        table=SKILL_VERSION_TABLE,
        record_id=version.version_id,
        scope=_scope(project_id),
        payload=payload,
    )


def _version_from_payload(payload: Any) -> SkillVersionCore:
    value = dict(payload)
    value.pop("project_id", None)
    value.pop("bundle", None)
    return SkillVersionCore.model_validate(value)


def _trajectory_record(project_id: str, trajectory: SkillTrajectory) -> Record:
    payload = trajectory.model_dump(mode="json")
    payload["project_id"] = project_id
    return Record(
        table=SKILL_TRAJECTORY_TABLE,
        record_id=trajectory.trajectory_id,
        scope=_scope(project_id),
        payload=payload,
    )


def _trajectory_from_payload(payload: Any) -> SkillTrajectory:
    value = dict(payload)
    value.pop("project_id", None)
    return SkillTrajectory.model_validate(value)


def _binding_records(project_id: str, trajectory: SkillTrajectory) -> list[Record]:
    records: list[Record] = []
    for index, binding in enumerate(trajectory.skill_bindings):
        binding_id = f"{trajectory.trajectory_id}:{index}"
        payload = binding.model_dump(mode="json")
        payload.update(
            {
                "project_id": project_id,
                "binding_id": binding_id,
                "trajectory_id": trajectory.trajectory_id,
                "binding_no": index,
            }
        )
        records.append(
            Record(
                table=SKILL_TRAJECTORY_BINDING_TABLE,
                record_id=binding_id,
                scope=_scope(project_id),
                payload=payload,
            )
        )
    return records


async def _begin_operation(
    transaction,
    *,
    scope: DatabaseScope,
    project_id: str,
    operation_id: str,
    operation_type: SkillRemoteOperationType,
    request_hash: str,
    request_payload: dict[str, Any],
    cloud_skill_id: str | None,
    now: datetime,
) -> dict[str, Any] | None:
    existing = await transaction.get_records(SKILL_OPERATION_TABLE, scope, [operation_id])
    if existing:
        operation = dict(existing[0].payload)
        if operation["request_hash"] != request_hash or operation["operation_type"] != operation_type.value:
            raise SkillConflictError(f"operation_id already belongs to different inputs: {operation_id}")
        return operation
    await transaction.upsert_records(
        SKILL_OPERATION_TABLE,
        [
            Record(
                table=SKILL_OPERATION_TABLE,
                record_id=operation_id,
                scope=scope,
                payload={
                    "project_id": project_id,
                    "operation_id": operation_id,
                    "cloud_skill_id": cloud_skill_id,
                    "operation_type": operation_type.value,
                    "request_hash": request_hash,
                    "request_payload": request_payload,
                    "status": SkillRemoteOperationStatus.RUNNING.value,
                    "result": None,
                    "error_code": None,
                    "created_at": now,
                    "updated_at": now,
                },
            )
        ],
    )
    return None


async def _complete_operation(
    transaction,
    *,
    scope: DatabaseScope,
    operation_id: str,
    cloud_skill_id: str | None,
    result: dict[str, Any],
    now: datetime,
) -> None:
    await transaction.patch_record(
        SKILL_OPERATION_TABLE,
        scope,
        operation_id,
        {
            "cloud_skill_id": cloud_skill_id,
            "status": SkillRemoteOperationStatus.SUCCEEDED.value,
            "result": result,
            "error_code": None,
            "updated_at": now,
        },
    )


def _encode_trajectory_cursor(fingerprint: str, timestamp: str, trajectory_id: str) -> str:
    payload = json.dumps(
        {"fingerprint": fingerprint, "timestamp": timestamp, "trajectory_id": trajectory_id},
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return base64.urlsafe_b64encode(payload).decode().rstrip("=")


def _decode_trajectory_cursor(cursor: str | None, fingerprint: str) -> tuple[str, str] | None:
    if cursor is None:
        return None
    try:
        padding = "=" * (-len(cursor) % 4)
        payload = json.loads(base64.urlsafe_b64decode(cursor + padding))
        if payload["fingerprint"] != fingerprint:
            raise ValueError("trajectory cursor does not match the query")
        return str(payload["timestamp"]), str(payload["trajectory_id"])
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        if isinstance(exc, ValueError) and str(exc) == "trajectory cursor does not match the query":
            raise
        raise ValueError("invalid trajectory cursor") from exc


def _text(name: str, *, nullable: bool = True, default=None) -> FieldSpec:
    return FieldSpec(name=name, field_type=FieldType.TEXT, nullable=nullable, default=default)


def _integer(name: str, *, nullable: bool = True, default=None) -> FieldSpec:
    return FieldSpec(name=name, field_type=FieldType.INTEGER, nullable=nullable, default=default)


def _float(name: str, *, nullable: bool = True, default=None) -> FieldSpec:
    return FieldSpec(name=name, field_type=FieldType.FLOAT, nullable=nullable, default=default)


def _datetime(name: str, *, nullable: bool = True, default=None) -> FieldSpec:
    return FieldSpec(name=name, field_type=FieldType.DATETIME, nullable=nullable, default=default)


def _json(name: str, *, nullable: bool = True, default=None) -> FieldSpec:
    return FieldSpec(name=name, field_type=FieldType.JSON, nullable=nullable, default=default)


__all__ = [
    "SKILL_OPERATION_TABLE",
    "SKILL_TRAJECTORY_BINDING_TABLE",
    "SKILL_TRAJECTORY_TABLE",
    "SKILL_VERSION_TABLE",
    "SkillRelationalRepository",
    "build_cloud_skill_tables",
]
