"""Standalone local Skill management over immutable version facts."""

from __future__ import annotations

import re
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from difflib import unified_diff
from pathlib import Path

from ..contracts import SkillBundle, canonical_request_hash
from ..errors import SkillConflictError
from ..persistence import (
    DEFAULT_SKILL_DATABASE_PATH,
    SkillRecord,
    SkillSyncStateRecord,
    SkillVersionOrigin,
    SkillVersionStatus,
    bootstrap_skill_database,
)
from ..typing import Skill, SkillCandidate, compute_skill_content_hash
from .bundle import frontmatter_value, next_version_label, parse_version_label, serialize_files
from .installer import SkillInstaller
from .models import (
    DuplicateAction,
    ExportSkillRequest,
    ExportSkillResult,
    ManagedSkill,
    PendingSkillOperation,
    PendingSkillOperationStatus,
    PendingSkillOperationType,
    PublishSkillRequest,
    PublishSkillResult,
    RegisterSkillRequest,
    RegisterSkillResult,
    SkillDetail,
    SkillDiffResult,
    SkillSnapshot,
    push_operation_id,
)
from .repository import SkillRepository
from .snapshot import (
    read_skill_snapshot,
    snapshot_from_editor,
    snapshot_from_editor_files,
    snapshot_from_record,
    snapshot_metadata,
)

_ALIAS_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


class LocalSkillManager:
    """Manage a local Skill DAG without mutable head pointers."""

    def __init__(
        self,
        repository: SkillRepository,
        *,
        managed_root: str | Path | None = None,
        clock: Callable[[], datetime] | None = None,
        id_generator: Callable[[], str] | None = None,
        owns_database: bool = False,
    ) -> None:
        self.repository = repository
        self._installer = SkillInstaller(managed_root=managed_root)
        self._clock = clock or (lambda: datetime.now(UTC))
        self._id_generator = id_generator or (lambda: str(uuid.uuid4()))
        self._owns_database = owns_database

    @classmethod
    async def open(cls, database_path: str | Path | None = None) -> LocalSkillManager:
        path = DEFAULT_SKILL_DATABASE_PATH if database_path is None else Path(database_path).expanduser()
        database = await bootstrap_skill_database(path)
        return cls(SkillRepository(database), managed_root=path.parent, owns_database=True)

    async def close(self) -> None:
        if self._owns_database:
            await self.repository.database.close()
            self._owns_database = False

    async def register(self, request: RegisterSkillRequest) -> RegisterSkillResult:
        snapshot = read_skill_snapshot(request.source_path)
        matches = await self.repository.find_snapshot_matches(snapshot.local_snapshot_hash)
        if matches and request.duplicate_action is None:
            raise SkillConflictError(
                "identical local Skill snapshot already exists; choose duplicate_action='reuse' or 'create_new'"
            )
        if matches and request.duplicate_action == DuplicateAction.REUSE:
            matched = matches[0]
            return RegisterSkillResult(action="reused", skill_id=matched.skill_id, version_id=matched.version_id)

        now = self._clock()
        skill_id = self._id_generator()
        version_id = self._id_generator()
        alias = _normalize_alias(request.alias)
        name = request.name or frontmatter_value(snapshot.blob["SKILL.md"], "name") or Path(request.source_path).stem
        version_label = request.version_label or frontmatter_value(snapshot.blob["SKILL.md"], "version") or "0.1.0"
        parse_version_label(version_label)
        record = self._record_from_snapshot(
            snapshot,
            skill_id=skill_id,
            version_id=version_id,
            name=name,
            alias=alias,
            version_label=version_label,
            commit_message=_normalized_message(request.commit_message),
            parent_version_ids=[],
            created_at=now,
        )
        await self.repository.create_version(record, now=now, pending_operation=_push_operation(record, now))
        return RegisterSkillResult(action="created", skill_id=skill_id, version_id=version_id)

    async def publish(self, request: PublishSkillRequest) -> PublishSkillResult:
        if sum(value is not None for value in (request.source_path, request.content, request.files)) != 1:
            raise SkillConflictError("publish requires exactly one of source_path, content or files")
        skill_id = await self.repository.resolve_skill_id(request.skill_ref)
        base = (
            await self.get_version(skill_id, request.base_version_id)
            if request.base_version_id
            else await self.repository.get_latest_available_version(skill_id)
        )
        inherited = snapshot_from_record(base)
        if request.source_path is not None:
            snapshot = read_skill_snapshot(request.source_path)
        elif request.files is not None:
            snapshot = snapshot_from_editor_files(request.files, inherited)
        else:
            snapshot = snapshot_from_editor(request.content or "", inherited)
        versions = await self.repository.list_versions(skill_id)
        version_label = (
            request.version_label
            or frontmatter_value(snapshot.blob["SKILL.md"], "version")
            or next_version_label([item.version_label for item in versions])
        )
        parse_version_label(version_label)
        now = self._clock()
        record = self._record_from_snapshot(
            snapshot,
            skill_id=skill_id,
            version_id=self._id_generator(),
            name=base.name,
            alias=base.alias,
            version_label=version_label,
            commit_message=_normalized_message(request.commit_message),
            parent_version_ids=[base.version_id],
            created_at=now,
            cloud_skill_id=await self.repository.get_cloud_skill_id(skill_id),
        )
        await self.repository.create_version(record, now=now, pending_operation=_push_operation(record, now))
        return PublishSkillResult(
            skill_id=skill_id,
            version_id=record.version_id,
            local_snapshot_hash=snapshot.local_snapshot_hash,
        )

    async def list_skills(self) -> list[ManagedSkill]:
        summaries = [await self._summary(state) for state in await self.repository.list_sync_states()]
        return sorted(summaries, key=lambda item: (item.name.lower(), item.skill_id))

    async def unregister(self, skill_ref: str) -> ManagedSkill:
        detail = await self.get_skill(skill_ref)
        await self.repository.delete_family(detail.skill.skill_id)
        return detail.skill

    async def get_skill(self, skill_ref: str) -> SkillDetail:
        skill_id = await self.repository.resolve_skill_id(skill_ref)
        state = await self.repository.get_sync_state(skill_id)
        latest = await self.repository.get_latest_available_version(skill_id)
        return SkillDetail(skill=await self._summary(state), latest_version=latest, sync_state=state)

    async def list_versions(self, skill_ref: str) -> list[SkillRecord]:
        return await self.repository.list_versions(skill_ref)

    async def get_version(self, skill_ref: str, version_id: str) -> SkillRecord:
        skill_id = await self.repository.resolve_skill_id(skill_ref)
        record = await self.repository.get_version(version_id)
        if record.skill_id != skill_id:
            raise SkillConflictError(f"version {version_id} does not belong to Skill {skill_id}")
        return record

    async def export(self, request: ExportSkillRequest) -> ExportSkillResult:
        detail = await self.get_skill(request.skill_ref)
        version_id = request.version_id or detail.latest_version.version_id
        record = await self.get_version(detail.skill.skill_id, version_id)
        snapshot = snapshot_from_record(record)
        target = self._installer.export(snapshot, request.target_path, replace=request.replace)
        return ExportSkillResult(
            skill_id=record.skill_id,
            version_id=record.version_id,
            target_path=str(target),
            exported_files=[item.path for item in snapshot.files],
            local_snapshot_hash=snapshot.local_snapshot_hash,
        )

    async def diff(
        self,
        skill_ref: str,
        *,
        to_version_id: str,
        from_version_id: str | None = None,
    ) -> SkillDiffResult:
        detail = await self.get_skill(skill_ref)
        resolved_from = from_version_id or detail.latest_version.version_id
        before = snapshot_from_record(await self.get_version(detail.skill.skill_id, resolved_from))
        after = snapshot_from_record(await self.get_version(detail.skill.skill_id, to_version_id))
        chunks: list[str] = []
        changed_files: list[str] = []
        for path in sorted(set(before.file_contents) | set(after.file_contents)):
            if before.file_contents.get(path) == after.file_contents.get(path):
                continue
            changed_files.append(path)
            chunks.extend(
                unified_diff(
                    before.file_contents.get(path, "").splitlines(keepends=True),
                    after.file_contents.get(path, "").splitlines(keepends=True),
                    fromfile=f"{resolved_from}/{path}",
                    tofile=f"{to_version_id}/{path}",
                )
            )
        return SkillDiffResult(
            skill_id=detail.skill.skill_id,
            from_version_id=resolved_from,
            to_version_id=to_version_id,
            diff="".join(chunks),
            changed_files=changed_files,
        )

    async def persist_algorithm_candidate(self, candidate: SkillCandidate, *, base_version_id: str) -> Skill:
        """Create one canonical immutable version from an unpersisted algorithm candidate."""

        base = Skill.from_record(await self.repository.get_version(base_version_id))
        versions = await self.repository.list_versions(base.skill_id)
        now = self._clock()
        evolved = base.model_copy(
            update={
                "version_id": self._id_generator(),
                "cloud_skill_id": await self.repository.get_cloud_skill_id(base.skill_id),
                "parent_version_ids": [base.version_id],
                "blob": candidate.blob,
                "resources": candidate.resources,
                "content_hash": compute_skill_content_hash(candidate.blob),
                "status": SkillVersionStatus.DRAFT,
                "version_label": next_version_label([version.version_label for version in versions]),
                "commit_message": candidate.commit_message,
                "metadata": {**base.metadata, **candidate.metadata},
                "created_at": now,
                "updated_at": now,
                "origin": SkillVersionOrigin.EVOLUTION,
            }
        )
        record = evolved.to_record()
        await self.repository.create_version(record, now=now, pending_operation=_push_operation(record, now))
        return Skill.from_record(record)

    async def persist_optimized_version(self, candidate: SkillCandidate, *, base_version_id: str) -> Skill:
        """Compatibility alias for callers migrating to ``persist_algorithm_candidate``."""

        return await self.persist_algorithm_candidate(candidate, base_version_id=base_version_id)

    async def _summary(self, state: SkillSyncStateRecord) -> ManagedSkill:
        versions = await self.repository.query_versions(skill_id=state.skill_id)
        latest = await self.repository.get_latest_available_version(state.skill_id)
        operations = await self.repository.list_operations(skill_id=state.skill_id)
        pending_count = sum(item.status in {"pending", "running", "failed"} for item in operations)
        return ManagedSkill(
            skill_id=state.skill_id,
            name=latest.name,
            description=latest.description,
            alias=latest.alias,
            cloud_skill_id=await self.repository.get_cloud_skill_id(state.skill_id),
            latest_version_id=latest.version_id,
            latest_version_label=latest.version_label,
            last_version_sync_at=state.last_version_sync_at,
            last_trajectory_pull_at=state.last_trajectory_pull_at,
            version_count=len(versions),
            pending_count=pending_count,
            created_at=state.created_at,
            updated_at=max([state.updated_at, *(item.updated_at for item in versions)]),
        )

    @staticmethod
    def _record_from_snapshot(
        snapshot: SkillSnapshot,
        *,
        skill_id: str,
        version_id: str,
        name: str,
        alias: str | None,
        version_label: str,
        commit_message: str | None,
        parent_version_ids: list[str],
        created_at: datetime,
        cloud_skill_id: str | None = None,
    ) -> SkillRecord:
        bundle = SkillBundle.from_files(snapshot.blob)
        return SkillRecord(
            skill_id=skill_id,
            version_id=version_id,
            cloud_skill_id=cloud_skill_id,
            parent_version_ids=parent_version_ids,
            name=name,
            description=frontmatter_value(snapshot.blob["SKILL.md"], "description"),
            alias=alias,
            bundle=bundle.canonical_json(),
            resources=serialize_files(snapshot.resources),
            content_hash=bundle.content_hash,
            local_snapshot_hash=snapshot.local_snapshot_hash,
            status=SkillVersionStatus.DRAFT,
            version_label=version_label,
            commit_message=commit_message,
            metadata={},
            local_metadata={"snapshot": snapshot_metadata(snapshot)},
            created_at=created_at,
            updated_at=created_at,
            origin=SkillVersionOrigin.LOCAL,
        )


def _normalize_alias(alias: str | None) -> str | None:
    if alias is None or not alias.strip():
        return None
    normalized = alias.strip()
    if _ALIAS_PATTERN.fullmatch(normalized) is None:
        raise SkillConflictError(
            "Skill alias must be 1-64 characters and contain only letters, numbers, '.', '_', or '-'"
        )
    return normalized


def _normalized_message(message: str | None) -> str | None:
    normalized = message.strip() if message is not None else ""
    return normalized or None


def _push_operation(record: SkillRecord, now: datetime) -> PendingSkillOperation:
    request_hash = canonical_request_hash(
        {
            "skill_id": record.skill_id,
            "version_id": record.version_id,
            "content_hash": record.content_hash,
            "parent_version_ids": record.parent_version_ids,
        }
    )
    return PendingSkillOperation(
        operation_id=push_operation_id(record.skill_id, record.version_id),
        operation_type=PendingSkillOperationType.PUSH_VERSION,
        skill_id=record.skill_id,
        version_id=record.version_id,
        request_hash=request_hash,
        status=PendingSkillOperationStatus.PENDING,
        created_at=now,
        updated_at=now,
    )


__all__ = ["LocalSkillManager"]
