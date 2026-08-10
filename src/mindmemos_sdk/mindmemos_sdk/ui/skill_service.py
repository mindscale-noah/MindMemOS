"""UI-facing application service for centralized local Skill management."""

from __future__ import annotations

from mindmemos_skill.management import (
    PendingSkillOperation,
    SkillManagementDetail,
    SkillManagementSummary,
    snapshot_from_record,
)
from mindmemos_skill.persistence import SkillRecord
from pydantic import BaseModel, ConfigDict, Field

from ..skills import (
    EvolveCloudResult,
    ExportSkillRequest,
    ExportSkillResult,
    PublishLocalRequest,
    PublishLocalResult,
    RegisterLocalRequest,
    RegisterLocalResult,
    SkillEvolveMode,
    SkillManager,
)
from ..skills.models import LocalSkillFileRole, LocalSkillSyncState


class SkillListItemView(BaseModel):
    """One centralized Skill family row rendered by the local UI."""

    model_config = ConfigDict(extra="forbid")

    skill_id: str
    name: str
    description: str | None = None
    alias: str | None = None
    cloud_skill_id: str | None = None
    latest_version_id: str
    latest_version_label: str
    version_count: int
    pending_count: int
    sync_state: str
    last_sync_at: str | None = None


class SkillVersionView(BaseModel):
    """One immutable version row rendered by the local UI."""

    model_config = ConfigDict(extra="forbid")

    version_id: str
    parent_version_ids: list[str] = Field(default_factory=list)
    version_label: str | None = None
    commit_message: str | None = None
    content_hash: str
    local_snapshot_hash: str
    origin: str
    status: str
    is_latest: bool
    is_published: bool
    has_linked_files: bool
    sync_state: str
    created_at: str


class SkillDetailView(BaseModel):
    """Complete local UI detail aggregate for one Skill family."""

    model_config = ConfigDict(extra="forbid")

    skill: SkillListItemView
    versions: list[SkillVersionView] = Field(default_factory=list)
    latest_version: SkillVersionView
    outbox_operations: list[PendingSkillOperation] = Field(default_factory=list)


class SkillContentView(BaseModel):
    """Human-editable files for one immutable version."""

    model_config = ConfigDict(extra="forbid")

    skill_id: str
    version_id: str
    content: str
    files: dict[str, str] = Field(default_factory=dict)


class SkillCompareView(BaseModel):
    """Local-only comparison including private linked-file path changes."""

    model_config = ConfigDict(extra="forbid")

    from_version_id: str
    to_version_id: str
    content_diff: str
    linked_file_changes: list[str] = Field(default_factory=list)


class SkillDeleteResultView(BaseModel):
    """Explicit scope report for one destructive local unregister operation."""

    model_config = ConfigDict(extra="forbid")

    skill_id: str
    name: str
    alias: str | None = None
    deleted_version_count: int = Field(ge=0)
    deleted_pending_count: int = Field(ge=0)
    source_files_deleted: bool = False
    cloud_skill_deleted: bool = False


class LocalSkillUIService:
    """Aggregate UI DTOs while delegating every state change to ``SkillManager``."""

    def __init__(self, manager: SkillManager) -> None:
        self._manager = manager

    def list_skills(self) -> list[SkillListItemView]:
        """Project the one-shot Application overview without querying SDK stores."""

        skills, _operations = self.overview()
        return skills

    def overview(self) -> tuple[list[SkillListItemView], list[PendingSkillOperation]]:
        """Return list rows and outbox projected from one Application result DTO."""

        overview = self._manager.management_overview()
        return [self._list_item(summary) for summary in overview.skills], overview.pending_operations

    def detail(self, skill_ref: str) -> SkillDetailView:
        """Build one Skill detail aggregate from immutable local state."""

        aggregate = self._manager.management_detail(skill_ref)
        versions = [self._version_view(aggregate, record) for record in aggregate.versions]
        by_id = {version.version_id: version for version in versions}
        return SkillDetailView(
            skill=self._list_item(SkillManagementSummary(skill=aggregate.skill, sync_state=aggregate.sync_state)),
            versions=versions,
            latest_version=by_id[aggregate.latest_version.version_id],
            outbox_operations=aggregate.pending_operations,
        )

    def content(self, skill_ref: str, version_id: str | None = None) -> SkillContentView:
        """Return only the algorithm-managed ``SKILL.md`` content."""

        aggregate = self._manager.management_detail(skill_ref)
        resolved_version_id = version_id or aggregate.latest_version.version_id
        record = next(item for item in aggregate.versions if item.version_id == resolved_version_id)
        snapshot = snapshot_from_record(record)
        return SkillContentView(
            skill_id=aggregate.skill.skill_id,
            version_id=resolved_version_id,
            content=snapshot.blob["SKILL.md"],
            files=snapshot.file_contents,
        )

    def register(self, request: RegisterLocalRequest) -> RegisterLocalResult:
        """Import a one-time source snapshot through the shared manager."""

        return self._manager.register_local(request)

    def publish(self, request: PublishLocalRequest) -> tuple[PublishLocalResult, SkillDetailView]:
        """Create an immutable editor or directory version and return refreshed detail."""

        result = self._manager.publish_local(request)
        return result, self.detail(result.skill_id)

    def export(self, request: ExportSkillRequest) -> ExportSkillResult:
        """Export a complete selected snapshot through the shared manager."""

        return self._manager.export_local(request)

    def unregister(self, skill_ref: str) -> SkillDeleteResultView:
        """Delete one local registration while preserving source and cloud data."""

        detail = self.detail(skill_ref)
        removed = self._manager.unregister(detail.skill.skill_id)
        return SkillDeleteResultView(
            skill_id=removed.skill_id,
            name=removed.name,
            alias=removed.alias,
            deleted_version_count=len(detail.versions),
            deleted_pending_count=len(detail.outbox_operations),
        )

    def sync(self, skill_ref: str, *, direction: str = "both") -> SkillDetailView:
        """Run one explicit cloud direction and return refreshed local state."""

        manifest = self._manager.sync_local(skill_ref, direction=direction)
        return self.detail(manifest.skill_id)

    def evolve(
        self,
        skill_ref: str,
        *,
        base_version_id: str | None = None,
        algorithm: str | None = None,
        mode: SkillEvolveMode = "sync",
        operation_id: str | None = None,
    ) -> EvolveCloudResult:
        """Request cloud evolution through the shared manager."""

        return self._manager.evolve_local(
            skill_ref,
            base_version_id=base_version_id,
            algorithm=algorithm,
            mode=mode,
            operation_id=operation_id,
        )

    def compare(self, skill_ref: str, from_version_id: str, to_version_id: str) -> SkillCompareView:
        """Compare algorithm content and local-only linked file manifests."""

        result = self._manager.diff_local(
            skill_ref,
            from_version_id=from_version_id,
            to_version_id=to_version_id,
        )
        aggregate = self._manager.management_detail(skill_ref)
        by_id = {record.version_id: record for record in aggregate.versions}
        from_snapshot = snapshot_from_record(by_id[from_version_id])
        to_snapshot = snapshot_from_record(by_id[to_version_id])
        from_files = {
            item.path: item.content_hash
            for item in from_snapshot.files
            if item.role.value != LocalSkillFileRole.ALGORITHM.value
        }
        to_files = {
            item.path: item.content_hash
            for item in to_snapshot.files
            if item.role.value != LocalSkillFileRole.ALGORITHM.value
        }
        changed = [
            path for path in sorted(set(from_files) | set(to_files)) if from_files.get(path) != to_files.get(path)
        ]
        return SkillCompareView(
            from_version_id=from_version_id,
            to_version_id=to_version_id,
            content_diff=result.diff,
            linked_file_changes=changed,
        )

    def _list_item(
        self,
        summary: SkillManagementSummary,
    ) -> SkillListItemView:
        skill = summary.skill
        return SkillListItemView(
            skill_id=skill.skill_id,
            name=skill.name,
            description=skill.description,
            alias=skill.alias,
            cloud_skill_id=skill.cloud_skill_id,
            latest_version_id=skill.latest_version_id,
            latest_version_label=skill.latest_version_label,
            version_count=skill.version_count,
            pending_count=skill.pending_count,
            sync_state=summary.sync_state.value,
            last_sync_at=skill.last_version_sync_at.isoformat() if skill.last_version_sync_at else None,
        )

    def _version_view(
        self,
        aggregate: SkillManagementDetail,
        record: SkillRecord,
    ) -> SkillVersionView:
        snapshot = snapshot_from_record(record)
        operation = next((item for item in aggregate.pending_operations if item.version_id == record.version_id), None)
        if operation is not None and operation.status.value == "failed":
            sync_state = LocalSkillSyncState.FAILED.value
        elif operation is not None:
            sync_state = LocalSkillSyncState.PENDING.value
        elif isinstance(record.metadata.get("cloud"), dict) or record.origin.value == "cloud":
            sync_state = LocalSkillSyncState.SYNCED.value
        else:
            sync_state = LocalSkillSyncState.LOCAL_ONLY.value
        return SkillVersionView(
            version_id=record.version_id,
            parent_version_ids=list(record.parent_version_ids),
            version_label=record.version_label,
            commit_message=record.commit_message,
            content_hash=record.content_hash,
            local_snapshot_hash=snapshot.local_snapshot_hash,
            origin=record.origin.value,
            status=record.status.value,
            is_latest=record.version_id == aggregate.latest_version.version_id,
            is_published=record.status.value == "published",
            has_linked_files=bool(snapshot.resources),
            sync_state=sync_state,
            created_at=record.created_at.isoformat(),
        )
