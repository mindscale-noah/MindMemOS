"""Repository business operations over one injected persistence backend."""

import asyncio
from datetime import datetime
from pathlib import Path

from ....api.typing import (
    ActiveRequest,
    CommitFromLocalRequest,
    CommitRequest,
    DiffRequest,
    ExportRequest,
    FileSkillInfo,
    HistoryRequest,
    ListRequest,
    ListTrajectoriesRequest,
    RegisterFromLocalRequest,
    RegisterRequest,
    RegisterResult,
    ReportTrajectoryRequest,
    SetMetaRequest,
    ShowRequest,
    ShowTrajectoryRequest,
    SkillDetail,
    SkillRequest,
    StructuredSkillInfo,
    VersionResolution,
    VersionResolveRequest,
)
from ....common.skill_versions import compare_versions, snapshot_from_version
from ....common.time_utils import utcnow
from ....errors import SkillNotFoundError, SkillVersionNotFoundError
from ....persistence import SkillRecord, SkillVersionRecord, SkillVersionSummary, TrajectoryRecord, TrajectorySummary
from ....persistence.constants import SKILL_VERSIONS_TABLE, SKILLS_TABLE
from ....persistence.db_services.skills.typing import VersionNaming
from ....persistence.store import SkillPersistence
from ....skill import Skill, SkillContent, SkillSnapshot, decode_resources, encode_resources
from ....skill.io import read_skill
from ....skill.typing import DiffResult
from ...base import Pipeline
from .config import RepositoryPipelineConfig


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

    @staticmethod
    def _scope(project_id: str | None) -> str:
        """Require a resolved scope when invoked without the public service."""
        if project_id is None:
            raise ValueError("project_id is required")
        return project_id

    async def register_skill(self, request: RegisterRequest) -> RegisterResult:
        """Register structured source content and return its persisted identity.

        Args:
            request: Authorized project, source content, and optional family alias.

        Returns:
            Family ID, initial version ID, and both source hashes.
        """
        request = RegisterRequest.model_validate(request)
        snapshot = self._snapshot(request.skill)
        record = await asyncio.to_thread(
            self.persistence.skills.register,
            project_id=self._scope(request.project_id),
            name=snapshot.name,
            alias=request.alias,
            source=request.skill.commit_source,
            message=request.skill.commit_message,
            snapshot=snapshot,
            naming=VersionNaming(version_label=request.skill.commit_version, version_bump=request.skill.version_bump),
            metadata=request.skill.metadata,
            file_meta=request.skill.metadata,
        )
        return RegisterResult(
            skill_id=record.skill_id,
            skill_version_id=record.active_version_id,
            content_hash=snapshot.content_hash,
            skill_hash=snapshot.skill_hash,
        )

    async def register_from_local(self, request: RegisterFromLocalRequest) -> RegisterResult:
        """Read a standard local source before entering the shared registration path.

        Args:
            request: Authorized project, local source path, and family alias.

        Returns:
            Persisted registration identity and source hashes.
        """
        request = RegisterFromLocalRequest.model_validate(request)
        source = await asyncio.to_thread(self._local_source, request.skill)
        return await self.register_skill(
            RegisterRequest(
                project_id=request.project_id,
                user_id=request.user_id,
                agent_id=request.agent_id,
                alias=request.alias,
                skill=source,
            )
        )

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
            offset=request.cursor.offset,
            snapshot_at=request.cursor.snapshot_at,
            after_created_at=request.cursor.after_created_at,
            after_skill_id=request.cursor.after_skill_id,
        )

    async def report_trajectory(self, request: ReportTrajectoryRequest) -> TrajectoryRecord:
        """Persist a sparse report through the shared trajectory store.

        Args:
            request: Authorized project, loaded version, and Harness execution.

        Returns:
            Stored physical attempt with server receipt time.
        """
        request = ReportTrajectoryRequest.model_validate(request)
        return await asyncio.to_thread(
            self.persistence.trajectories.report,
            project_id=self._scope(request.project_id),
            skill_version_id=request.skill_version_id,
            user_id=request.user_id,
            agent_id=request.agent_id,
            messages=request.messages,
            session_id=request.session_id,
            status=request.status,
            started_at=request.started_at,
            finished_at=request.finished_at,
        )

    async def show_trajectory(self, request: ShowTrajectoryRequest) -> TrajectoryRecord:
        """Read one physical attempt inside its authorized project.

        Args:
            request: Project and exact trajectory identity.

        Returns:
            Stored local or reported trajectory.
        """
        request = ShowTrajectoryRequest.model_validate(request)
        record = await asyncio.to_thread(self.persistence.trajectories.get, request.trajectory_id)
        if record is None or record.project_id != self._scope(request.project_id):
            raise SkillNotFoundError("Trajectory not found in this project")
        return record

    async def list_trajectories(self, request: ListTrajectoriesRequest) -> list[TrajectorySummary]:
        """Page reported attempts for one exact project-owned Skill family.

        Args:
            request: Family, filters, and stable receipt cursor.

        Returns:
            Reported trajectories with an optional lookahead row.
        """
        request = ListTrajectoriesRequest.model_validate(request)
        project_id = self._scope(request.project_id)
        family = await asyncio.to_thread(
            self.persistence.database.select,
            SKILLS_TABLE,
            filters={"project_id": project_id, "skill_id": request.skill_id},
            fields=["skill_id"],
            limit=1,
        )
        if not family:
            raise SkillNotFoundError("Skill not found in this project")
        if request.skill_version_id is not None:
            version = await asyncio.to_thread(
                self.persistence.database.select,
                SKILL_VERSIONS_TABLE,
                filters={"project_id": project_id, "version_id": request.skill_version_id},
                fields=["skill_id"],
                limit=1,
            )
            if not version:
                raise SkillVersionNotFoundError("Skill version not found in this project")
            if version[0]["skill_id"] != request.skill_id:
                raise SkillVersionNotFoundError("Skill version does not belong to this Skill")
        return await asyncio.to_thread(
            self.persistence.trajectories.list_reported,
            request.skill_id,
            skill_version_id=request.skill_version_id,
            status=request.status,
            received_at_from=request.cursor.received_at_from,
            received_at_to=request.cursor.received_at_to,
            snapshot_at=request.cursor.snapshot_at or utcnow(),
            after_received_at=request.cursor.after_received_at,
            after_trajectory_id=request.cursor.after_trajectory_id,
            limit=request.cursor.limit,
        )

    async def skill_has_reported_trajectory(
        self,
        project_id: str,
        skill_id: str,
        *,
        user_id: str | None,
        agent_id: str | None,
        snapshot_at: datetime,
    ) -> bool:
        """Check one Skill family against same-execution actor filters.

        Args:
            project_id: Authorized project identity.
            skill_id: Exact family identity.
            user_id: Optional business user identity.
            agent_id: Optional Agent identity.
            snapshot_at: First-page receipt cutoff.

        Returns:
            Whether one report satisfies all supplied filters.
        """
        return await asyncio.to_thread(
            self.persistence.trajectories.has_reported,
            skill_id,
            user_id=user_id,
            agent_id=agent_id,
            snapshot_at=snapshot_at,
        )

    async def reported_skill_ids(
        self,
        skill_ids: list[str],
        *,
        user_id: str | None,
        agent_id: str | None,
        snapshot_at: datetime,
    ) -> set[str]:
        """Find candidate Skill IDs with matching reports in the active project.

        Args:
            skill_ids: Candidate family identities from one list batch.
            user_id: Optional exact business user identity.
            agent_id: Optional exact Agent identity on the same report.
            snapshot_at: First-page receipt cutoff.

        Returns:
            Candidate IDs with same-report actor evidence.
        """
        return await asyncio.to_thread(
            self.persistence.trajectories.reported_skill_ids,
            skill_ids,
            user_id=user_id,
            agent_id=agent_id,
            snapshot_at=snapshot_at,
        )

    async def show_skill(self, request: ShowRequest) -> SkillDetail:
        """Read central metadata and an owned version.

        Args:
            request: Project, Skill reference, and optional version.
        """
        request = ShowRequest.model_validate(request)
        store = self.persistence.skills
        if request.skill_id is None:
            version = await asyncio.to_thread(store.version_by_id, self._scope(request.project_id), request.version_id)
            record = await asyncio.to_thread(store.resolve_family, self._scope(request.project_id), version.skill_id)
        else:
            record = await asyncio.to_thread(store.resolve_family, self._scope(request.project_id), request.skill_id)
            version = await asyncio.to_thread(store.version, record, request.version_id)
        return SkillDetail(skill=record, version=version)

    async def resolve_skill_reference(self, project_id: str, reference: str) -> SkillDetail:
        """Resolve a project-owned Skill alias, family, or immutable version.

        Args:
            project_id: Authorized project identity.
            reference: Alias or unique prefixed Skill identity.

        Returns:
            Owning family and selected version.
        """
        skill, version = await asyncio.to_thread(self.persistence.skills.resolve, self._scope(project_id), reference)
        return SkillDetail(skill=skill, version=version)

    async def resolve_skill_version(self, request: VersionResolveRequest) -> VersionResolution:
        """Resolve all supplied immutable constraints inside the project boundary.

        Args:
            request: Exact identity, hashes, and optional owning family.

        Returns:
            Unique version identity, hashes, metadata, and active state.
        """
        request = VersionResolveRequest.model_validate(request)
        store = self.persistence.skills
        version = await asyncio.to_thread(
            store.resolve_version_hash,
            request.project_id,
            request.content_hash,
            skill_hash=request.skill_hash,
            skill_id=request.skill_id,
            version_id=request.skill_version_id,
        )
        family = await asyncio.to_thread(store.resolve_family, request.project_id, version.skill_id)
        return VersionResolution(
            skill_id=version.skill_id,
            skill_version_id=version.version_id,
            content_hash=version.content_hash,
            skill_hash=version.skill_hash,
            version=version.version_label,
            alias=family.alias,
            commit_message=version.commit_message,
            is_active=family.active_version_id == version.version_id,
        )

    async def get_skill_version(self, project_id: str, version_id: str) -> SkillVersionRecord:
        """Read one exact Skill version in the authorized project.

        Args:
            project_id: Authorized project identity.
            version_id: Exact immutable Skill version identity.

        Returns:
            The project-owned version record.
        """
        return await asyncio.to_thread(self.persistence.skills.version_by_id, self._scope(project_id), version_id)

    async def list_skill_versions(self, project_id: str, skill_id: str) -> list[SkillVersionRecord]:
        """Read the complete history of an exact project-owned Skill family.

        Args:
            project_id: Authorized project identity.
            skill_id: Exact Skill family identity.

        Returns:
            Versions in creation order, or an empty list for an absent family.
        """
        if self.persistence.database.scope.require("project_id") != self._scope(project_id):
            raise PermissionError("Skill version project differs from the active scope")
        versions = []
        while True:
            page = await asyncio.to_thread(
                self.persistence.skills.list_versions, skill_id, limit=100, offset=len(versions)
            )
            versions.extend(page)
            if len(page) < 100:
                return sorted(versions, key=lambda version: (version.created_at, version.version_id))

    async def export_skill(self, request: ExportRequest) -> str:
        """Export a complete stored snapshot without tracking the directory.

        Args:
            request: Skill/version selection and explicit destination replacement policy.

        Returns:
            Absolute path of the exported application directory.
        """
        request = ExportRequest.model_validate(request)
        target = Path(request.directory).expanduser().resolve()
        database_path = self.persistence.local_database_path()
        if database_path is not None:
            if Path(database_path).resolve().is_relative_to(target):
                raise ValueError("Export destination must not contain the repository database")
        detail = await self.show_skill(
            ShowRequest(project_id=request.project_id, skill_id=request.skill_id, version_id=request.version_id)
        )
        snapshot = snapshot_from_version(detail.version)
        skill = Skill.from_snapshot(snapshot)
        return str(await asyncio.to_thread(skill.export, request.directory, overwrite=request.overwrite))

    async def commit_skill(self, request: CommitRequest) -> SkillDetail:
        """Publish a child snapshot and optionally activate it atomically.

        Args:
            request: Authorized family, source, parent, and publication options.

        Returns:
            Family metadata and the exact newly persisted version.
        """
        request = CommitRequest.model_validate(request)
        store = self.persistence.skills
        record = await asyncio.to_thread(store.resolve_family, self._scope(request.project_id), request.skill_id)
        parent = await asyncio.to_thread(store.version, record, request.base_version_id)
        snapshot = self._snapshot(
            request.skill,
            template=snapshot_from_version(parent),
            inherited_resources=parent.resources if request.inherit_resources else None,
        )
        version_id = store.new_version_id()
        updated = await asyncio.to_thread(
            store.commit,
            record,
            snapshot,
            message=request.skill.commit_message,
            naming=VersionNaming(version_label=request.skill.commit_version, version_bump=request.skill.version_bump),
            base_version_id=parent.version_id,
            active=request.active,
            version_id=version_id,
            file_meta=request.skill.metadata,
            metadata=request.skill.metadata,
            commit_source=request.skill.commit_source,
        )
        version = await asyncio.to_thread(store.version_by_id, request.project_id, version_id)
        return SkillDetail(skill=updated, version=version)

    async def commit_from_local(self, request: CommitFromLocalRequest) -> SkillDetail:
        """Read a standard local source before entering the shared commit path.

        Args:
            request: Authorized family, local source, and publication options.

        Returns:
            Family metadata and the exact newly persisted version.
        """
        request = CommitFromLocalRequest.model_validate(request)
        source = await asyncio.to_thread(self._local_source, request.skill)
        return await self.commit_skill(
            CommitRequest(
                project_id=request.project_id,
                user_id=request.user_id,
                agent_id=request.agent_id,
                skill_id=request.skill_id,
                skill=source,
                base_version_id=request.base_version_id,
                active=request.active,
                inherit_resources=request.inherit_resources,
            )
        )

    async def skill_history(self, request: HistoryRequest) -> list[SkillVersionSummary]:
        """Read a page of versions without changing the active selection.

        Args:
            request: Project, Skill reference, and pagination.
        """
        request = HistoryRequest.model_validate(request)
        store = self.persistence.skills
        return await asyncio.to_thread(
            store.history,
            await asyncio.to_thread(store.resolve_family, self._scope(request.project_id), request.skill_id),
            limit=request.limit,
            offset=request.cursor.offset,
            snapshot_at=request.cursor.snapshot_at,
            after_created_at=request.cursor.after_created_at,
            after_version_id=request.cursor.after_version_id,
        )

    async def diff_skill(self, request: DiffRequest) -> DiffResult:
        """Compare complete snapshots from the same Skill.

        Args:
            request: Skill and version pair; the starting version defaults to active.
        """
        request = DiffRequest.model_validate(request)
        store = self.persistence.skills
        target = await asyncio.to_thread(store.version_by_id, request.project_id, request.to_version_id)
        if request.from_version_id is None:
            family = await asyncio.to_thread(store.resolve_family, request.project_id, target.skill_id)
            source = await asyncio.to_thread(store.version, family)
        else:
            source = await asyncio.to_thread(store.version_by_id, request.project_id, request.from_version_id)
        return compare_versions(source, target)

    async def active_skill(self, request: ActiveRequest) -> SkillRecord:
        """Change the current pointer without editing any application directory.

        Args:
            request: Skill and existing owned version to select.
        """
        request = ActiveRequest.model_validate(request)
        store = self.persistence.skills
        version = await asyncio.to_thread(store.version_by_id, request.project_id, request.version_id)
        family = await asyncio.to_thread(store.resolve_family, request.project_id, version.skill_id)
        if request.expected_active_version_id is not None:
            family = family.model_copy(update={"active_version_id": request.expected_active_version_id})
        return await asyncio.to_thread(store.activate, family, version.version_id)

    async def set_skill_meta(self, request: SetMetaRequest) -> SkillRecord:
        """Update name or alias without creating a version.

        Args:
            request: Skill reference and explicit metadata changes.
        """
        request = SetMetaRequest.model_validate(request)
        store = self.persistence.skills
        return await asyncio.to_thread(
            store.set_meta,
            await asyncio.to_thread(store.resolve_family, self._scope(request.project_id), request.skill_id),
            name=request.name,
            alias=request.alias,
            clear_alias=request.clear_alias,
            metadata=request.metadata,
        )

    async def delete_skill(self, request: SkillRequest) -> None:
        """Soft-delete a central Skill while retaining stored references.

        Args:
            request: Project and Skill identity, alias, or unique prefix.
        """
        request = SkillRequest.model_validate(request)
        store = self.persistence.skills
        await asyncio.to_thread(
            store.delete,
            await asyncio.to_thread(store.resolve_family, self._scope(request.project_id), request.skill_ref),
        )

    @staticmethod
    def _local_source(info: FileSkillInfo) -> StructuredSkillInfo:
        """Read every local file before constructing the shared structured input."""
        source = read_skill(info.path)
        return StructuredSkillInfo(
            content=source.content,
            resources=encode_resources(source.resources),
            name=source.name,
            description=source.description,
            standard_format=True,
            metadata=info.metadata,
            commit_message=info.commit_message,
            commit_version=info.commit_version,
            version_bump=info.version_bump,
            commit_source=info.commit_source,
        )

    @staticmethod
    def _snapshot(
        info: StructuredSkillInfo,
        *,
        template: SkillSnapshot | None = None,
        inherited_resources: dict[str, str] | None = None,
    ) -> SkillSnapshot:
        """Capture validated source while preserving parent runtime configuration."""
        source = SkillContent(
            content=info.content,
            resources=decode_resources(inherited_resources if inherited_resources is not None else info.resources),
            name=info.name,
            description=info.description,
            standard_format=info.standard_format,
        )
        return SkillSnapshot.from_content(
            source,
            implementation=template.implementation if template else "static",
            config_schema_version=template.config_schema_version if template else 1,
            config=template.config if template else None,
        )
