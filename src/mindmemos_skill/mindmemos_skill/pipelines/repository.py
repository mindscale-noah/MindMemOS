"""Repository business operations over one injected persistence backend."""

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path

from ..errors import SkillConflictError
from ..infra.database import Scope
from ..persistence import SkillRecord, SkillVersionRecord
from ..persistence.db_services.skills.typing import VersionNaming
from ..persistence.store import SkillPersistence
from ..skill import Skill, SkillSnapshot
from .base import Pipeline
from .config import RepositoryPipelineConfig
from .content import compare_versions, snapshot_from_version
from .graphs import GraphQueryPipeline
from .typing import (
    CommitRequest,
    DiffRequest,
    DiffResult,
    ExportRequest,
    HistoryRequest,
    ListRequest,
    RegisterRequest,
    SetMetaRequest,
    ShowRequest,
    SkillDetail,
    SkillRequest,
    SwitchRequest,
)


@Pipeline.register("repository")
class RepositoryPipeline(Pipeline[RepositoryPipelineConfig]):
    """Implement Skill repository operations without owning the database.

    Args:
        config: Repository operation settings.
        persistence: Shared Skill and trajectory data services.
    """

    config_type = RepositoryPipelineConfig
    """Strict configuration used by the existing backend factory."""

    def __init__(self, config: RepositoryPipelineConfig, *, persistence: SkillPersistence):
        super().__init__(config)
        if not isinstance(persistence, SkillPersistence):
            raise TypeError("Repository pipeline requires SkillPersistence")
        self.persistence = persistence

    async def initialize_storage(self) -> None:
        """Create missing tables and migrate historical version display labels."""
        await asyncio.to_thread(self.persistence.initialize_storage)

    async def graph_queries(self, project_id: str, authorize: Callable[[], Awaitable[Scope]]) -> GraphQueryPipeline:
        """Bind graph reads to an explicitly resolved project.

        Args:
            project_id: Required project identity.
            authorize: Service callback that rechecks access before each query.
        """
        return GraphQueryPipeline(self.persistence, self._scope(project_id), authorize)

    @staticmethod
    def _scope(project_id: str | None) -> str:
        """Require a resolved scope when invoked without the public service."""
        if project_id is None:
            raise ValueError("project_id is required")
        return project_id

    async def register_skill(self, request: RegisterRequest) -> SkillDetail:
        """Register a complete source under a new central identity.

        Args:
            request: Project, source content, display metadata, and provenance.
        """
        request = RegisterRequest.model_validate(request)
        snapshot = await asyncio.to_thread(self._source, request.source)
        source = request.registration_source
        if source is None and isinstance(request.source, str):
            source = str(Path(request.source).expanduser().resolve())
        store = self.persistence.skills
        record = await asyncio.to_thread(
            store.register,
            project_id=self._scope(request.project_id),
            name=request.name,
            alias=request.alias,
            source=source,
            message=request.message,
            snapshot=snapshot,
            naming=VersionNaming(
                version_label=request.version_label, version_bump=request.version_bump, algorithm=request.algorithm
            ),
        )
        return SkillDetail(skill=record, version=await asyncio.to_thread(store.version, record))

    async def list_skills(self, request: ListRequest) -> list[SkillRecord]:
        """List central records in one project.

        Args:
            request: Project and pagination settings.
        """
        request = ListRequest.model_validate(request)
        return await asyncio.to_thread(
            self.persistence.skills.list_skills,
            self._scope(request.project_id),
            limit=request.limit,
            offset=request.offset,
        )

    async def show_skill(self, request: ShowRequest) -> SkillDetail:
        """Read central metadata and an owned version.

        Args:
            request: Project, Skill reference, and optional version.
        """
        request = ShowRequest.model_validate(request)
        store = self.persistence.skills
        record = await asyncio.to_thread(store.resolve, self._scope(request.project_id), request.skill_ref)
        return SkillDetail(skill=record, version=await asyncio.to_thread(store.version, record, request.version_id))

    async def export_skill(self, request: ExportRequest) -> str:
        """Export a complete stored snapshot without tracking the directory.

        Args:
            request: Skill/version selection and explicit destination replacement policy.

        Returns:
            Absolute path of the exported application directory.
        """
        request = ExportRequest.model_validate(request)
        store = self.persistence.skills
        target = Path(request.directory).expanduser().resolve()
        database_path = self.persistence.local_database_path()
        if database_path is not None:
            if Path(database_path).resolve().is_relative_to(target):
                raise ValueError("Export destination must not contain the repository database")
        record = await asyncio.to_thread(store.resolve, self._scope(request.project_id), request.skill_ref)
        snapshot = snapshot_from_version(await asyncio.to_thread(store.version, record, request.version_id))
        skill = Skill.from_snapshot(snapshot)
        return str(await asyncio.to_thread(skill.export, request.directory, overwrite=request.overwrite))

    async def commit_skill(self, request: CommitRequest) -> SkillDetail:
        """Save a new version and select it only if the captured parent is still current.

        Args:
            request: Target Skill, source, optional expected parent, and commit message.
        """
        request = CommitRequest.model_validate(request)
        store = self.persistence.skills
        record = await asyncio.to_thread(store.resolve, self._scope(request.project_id), request.skill_ref)
        if request.base_version_id is not None and request.base_version_id != record.current_version_id:
            raise SkillConflictError("base_version_id is not the current version")
        snapshot = await asyncio.to_thread(
            self._source,
            request.source,
            template=snapshot_from_version(await asyncio.to_thread(store.version, record)),
        )
        updated = await asyncio.to_thread(
            store.commit,
            record,
            snapshot,
            message=request.message,
            naming=VersionNaming(
                version_label=request.version_label, version_bump=request.version_bump, algorithm=request.algorithm
            ),
        )
        return SkillDetail(skill=updated, version=await asyncio.to_thread(store.version, updated))

    async def skill_history(self, request: HistoryRequest) -> list[SkillVersionRecord]:
        """Read a page of versions without changing the current selection.

        Args:
            request: Project, Skill reference, and pagination.
        """
        request = HistoryRequest.model_validate(request)
        store = self.persistence.skills
        return await asyncio.to_thread(
            store.history,
            await asyncio.to_thread(store.resolve, self._scope(request.project_id), request.skill_ref),
            limit=request.limit,
            offset=request.offset,
        )

    async def skill_cloud_status(self, request: SkillRequest) -> str:
        """Summarize confirmed local cloud mappings without contacting a remote.

        Args:
            request: Project-scoped Skill reference.
        """
        skill = await asyncio.to_thread(
            self.persistence.skills.resolve, self._scope(request.project_id), request.skill_ref
        )
        remote = self.persistence.remote
        if remote is None:
            return "disabled"
        if await asyncio.to_thread(remote.has_version, skill.current_version_id):
            return "synced"
        offset = 0
        while True:
            page = await asyncio.to_thread(self.persistence.skills.history, skill, limit=100, offset=offset)
            for version in page:
                if await asyncio.to_thread(remote.has_version, version.version_id):
                    return "history_only"
            if len(page) < 100:
                return "not_synced"
            offset += len(page)

    async def diff_skill(self, request: DiffRequest) -> DiffResult:
        """Compare complete snapshots from the same Skill.

        Args:
            request: Skill and version pair; the starting version defaults to current.
        """
        request = DiffRequest.model_validate(request)
        store = self.persistence.skills
        record = await asyncio.to_thread(store.resolve, self._scope(request.project_id), request.skill_ref)
        return compare_versions(
            await asyncio.to_thread(store.version, record, request.from_version_id),
            await asyncio.to_thread(store.version, record, request.to_version_id),
        )

    async def switch_skill(self, request: SwitchRequest) -> SkillRecord:
        """Change the current pointer without editing any application directory.

        Args:
            request: Skill and existing owned version to select.
        """
        request = SwitchRequest.model_validate(request)
        store = self.persistence.skills
        return await asyncio.to_thread(
            store.switch,
            await asyncio.to_thread(store.resolve, self._scope(request.project_id), request.skill_ref),
            request.version_id,
        )

    async def set_skill_meta(self, request: SetMetaRequest) -> SkillRecord:
        """Update name or alias without creating a version.

        Args:
            request: Skill reference and explicit metadata changes.
        """
        request = SetMetaRequest.model_validate(request)
        store = self.persistence.skills
        return await asyncio.to_thread(
            store.set_meta,
            await asyncio.to_thread(store.resolve, self._scope(request.project_id), request.skill_ref),
            name=request.name,
            alias=request.alias,
            clear_alias=request.clear_alias,
        )

    async def delete_skill(self, request: SkillRequest) -> None:
        """Delete central data only when no retained evidence references it.

        Args:
            request: Project and Skill identity, alias, or unique prefix.
        """
        request = SkillRequest.model_validate(request)
        store = self.persistence.skills
        await asyncio.to_thread(
            store.delete,
            await asyncio.to_thread(store.resolve, self._scope(request.project_id), request.skill_ref),
        )

    @staticmethod
    def _source(source: SkillSnapshot | str, *, template: SkillSnapshot | None = None) -> SkillSnapshot:
        """Capture supplied source without retaining any directory binding."""
        if isinstance(source, str):
            captured = Skill.from_path(source).snapshot()
            if template is not None:
                captured.implementation = template.implementation
                captured.config_schema_version = template.config_schema_version
                captured.config = template.config
            return captured
        return SkillSnapshot.model_validate(source.model_dump())
