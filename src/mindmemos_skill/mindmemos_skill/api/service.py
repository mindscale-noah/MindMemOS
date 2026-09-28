"""Transport-independent facade delegating business operations to pipelines."""

from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack, contextmanager
from contextvars import ContextVar
from inspect import isawaitable
from typing import Any, TypeVar

from ..common.asyncio import run_sync
from ..infra.database import Database, Scope
from ..persistence import SkillRecord, SkillVersionRecord
from ..persistence.observability import AlgorithmAuditPersistence, AuditLog, AuditRun, AuditSpan
from ..persistence.store import SkillPersistence, validate_skill_scope
from ..pipelines import EvolutionPipeline, Pipeline, TrainingPipeline
from ..pipelines.graphs import GraphQueryPipeline
from ..pipelines.repository import RepositoryPipeline
from ..pipelines.train import training_run_id
from ..pipelines.typing import (
    CommitRequest,
    DiffRequest,
    DiffResult,
    ExportRequest,
    HistoryRequest,
    ListRequest,
    ProjectRequest,
    RegisterRequest,
    SetMetaRequest,
    ShowRequest,
    SkillDetail,
    SkillPipelineRequest,
    SkillPipelineResult,
    SkillRequest,
    SwitchRequest,
    TrainingRequest,
)
from .base import MindMemOSSkill

RequestT = TypeVar("RequestT", bound=ProjectRequest)
PipelineT = TypeVar("PipelineT", bound=Pipeline)


class _PrincipalLease:
    """Invalidate inherited request identities when their request exits."""

    def __init__(self, principal: object):
        self.principal = principal
        self.active = True


def _select_pipeline(pipelines: dict[str, PipelineT], name: str | None) -> PipelineT:
    """Resolve an explicit request name or the sole available pipeline."""
    if name is not None:
        if name not in pipelines:
            raise ValueError(f"Unknown or unavailable pipeline {name!r}; available: {list(pipelines)}")
        return pipelines[name]
    if len(pipelines) != 1:
        raise ValueError(f"pipeline is required; available: {list(pipelines)}")
    return next(iter(pipelines.values()))


class SkillService(MindMemOSSkill):
    """Expose injected pipelines and own the assembled backend lifecycle.

    Args:
        backends: Already constructed dependencies, ordered by construction.
        repository: Selected repository operations, or None when unconfigured.
        training_pipelines: Named training instances available to requests.
        evolution_pipelines: Named evolution instances available to requests.
        local_project_id: Scope bound only by explicit local assembly.
    """

    def __init__(
        self,
        *,
        backends: dict[str, Any],
        observability: AlgorithmAuditPersistence,
        repository: RepositoryPipeline | None = None,
        training_pipelines: dict[str, TrainingPipeline] | None = None,
        evolution_pipelines: dict[str, EvolutionPipeline] | None = None,
        local_project_id: str | None = None,
        persistence: SkillPersistence | None = None,
        project_authorizer: Callable[[object, str], Scope | Awaitable[Scope]] | None = None,
    ) -> None:
        self._backends = backends.copy()
        self._repository = repository
        self._training_pipelines = dict(training_pipelines or {})
        self._evolution_pipelines = dict(evolution_pipelines or {})
        self._observability = observability
        self.local_project_id = local_project_id
        self._persistence = persistence
        self._project_authorizer = project_authorizer
        self._principal: ContextVar[_PrincipalLease | None] = ContextVar("skill_service_principal", default=None)

    @property
    def backends(self) -> dict[str, Any]:
        """Return assembled backend instances for diagnostics and lifecycle inspection."""
        return self._backends.copy()

    async def close(self) -> None:
        """Close pipelines before their borrowed database in reverse construction order."""
        async with AsyncExitStack() as cleanup:
            if self._observability.database is None:
                cleanup.callback(self._observability.close)
            for backend in self._backends.values():
                if isinstance(backend, Pipeline):
                    cleanup.push_async_callback(backend.close)
                elif isinstance(backend, Database):
                    cleanup.callback(backend.close)
                elif isinstance(backend, AlgorithmAuditPersistence):
                    cleanup.callback(backend.close)

    @property
    def repository(self) -> RepositoryPipeline:
        """Return the selected repository pipeline, never a raw database."""
        if self._repository is None:
            raise NotImplementedError("Repository pipeline is not configured")
        return self._repository

    async def initialize_storage(self) -> None:
        """Idempotently initialize configured stores without enabling repository APIs."""
        for backend in self._backends.values():
            if isinstance(backend, (SkillPersistence, AlgorithmAuditPersistence)):
                await run_sync(backend.initialize_storage)

    async def algorithm_run(self, project_id: str | None, run_id: str) -> AuditRun | None:
        """Read one project-scoped algorithm run.

        Args:
            project_id: Explicit project or locally bound scope.
            run_id: Exact stable training run identity.

        Returns:
            Current run state, or None when absent.
        """
        if self._observability.database is None:
            raise NotImplementedError("Observability is not configured")
        scope = await self._authorized_scope(project_id)
        return await run_sync(self._observability.get_run, scope.require("project_id"), run_id)

    async def algorithm_spans(
        self, project_id: str | None, run_id: str, *, limit: int = 500, offset: int = 0
    ) -> list[AuditSpan]:
        """Read a project-scoped page of completed algorithm spans.

        Args:
            project_id: Explicit project or locally bound scope.
            run_id: Exact stable training run identity.
            limit: Maximum page size.
            offset: Number of previous spans to skip.

        Returns:
            Spans in start order.
        """
        if self._observability.database is None:
            raise NotImplementedError("Observability is not configured")
        scope = await self._authorized_scope(project_id)
        return await run_sync(
            self._observability.spans, scope.require("project_id"), run_id, limit=limit, offset=offset
        )

    async def algorithm_logs(
        self, project_id: str | None, run_id: str, *, limit: int = 200, offset: int = 0
    ) -> list[AuditLog]:
        """Read a project-scoped page of trace-correlated audit logs.

        Args:
            project_id: Explicit project or locally bound scope.
            run_id: Exact stable training run identity.
            limit: Maximum page size.
            offset: Number of previous logs to skip.

        Returns:
            Audit logs in event order.
        """
        if self._observability.database is None:
            raise NotImplementedError("Observability is not configured")
        scope = await self._authorized_scope(project_id)
        return await run_sync(self._observability.logs, scope.require("project_id"), run_id, limit=limit, offset=offset)

    @contextmanager
    def request_principal(self, principal: object):
        """Bind an externally authenticated principal to this async request.

        Args:
            principal: Trusted identity supplied by the host account system.

        Yields:
            The principal for service calls in this request.
        """
        if principal is None:
            raise PermissionError("An authenticated principal is required")
        lease = _PrincipalLease(principal)
        token = self._principal.set(lease)
        try:
            yield principal
        finally:
            lease.active = False
            self._principal.reset(token)

    async def _authorized_scope(self, requested: str | None) -> Scope:
        """Resolve local ownership or require external project authorization."""
        project_id = self._project_id(requested)
        if self.local_project_id is not None:
            scope = Scope(fields={"project_id": project_id})
            validate_skill_scope(scope)
            return scope
        lease = self._principal.get()
        if lease is None or not lease.active or self._project_authorizer is None:
            raise PermissionError("Project authorization is required")
        authorized = self._project_authorizer(lease.principal, project_id)
        if isawaitable(authorized):
            authorized = await authorized
        if not lease.active:
            raise PermissionError("Request identity is no longer active")
        scope = Scope.model_validate(authorized)
        try:
            validate_skill_scope(scope)
        except ValueError as exc:
            raise PermissionError("Authorized scope fields differ from the service scope") from exc
        if scope.require("project_id") != project_id:
            raise PermissionError("Authorized project differs from the requested project")
        return scope

    async def _execute(self, project_id: str, operation, *args):
        """Run one pipeline call within a request-scoped persistence handle."""
        scope = await self._authorized_scope(project_id)
        if self._persistence is None:
            return await operation(*args)
        with self._persistence.session(scope):
            return await operation(*args)

    def _project_id(self, requested: str | None) -> str:
        """Require explicit scope unless this service was assembled for local use."""
        if self.local_project_id is not None:
            if requested is not None and requested != self.local_project_id:
                raise ValueError("The requested project differs from the bound local scope")
            return self.local_project_id
        if requested is None:
            raise ValueError("project_id is required outside local service assembly")
        return requested

    def _scoped(self, request: RequestT) -> RequestT:
        """Bind request scope without mutating the caller's model."""
        return request.model_copy(update={"project_id": self._project_id(request.project_id)})

    async def graph_queries(self, project_id: str | None = None) -> GraphQueryPipeline:
        """Create a project-scoped graph query pipeline over shared persistence.

        Args:
            project_id: Explicit project, or the bound local scope when omitted.
        """
        scope = await self._authorized_scope(project_id)
        return await self.repository.graph_queries(
            scope.require("project_id"), lambda: self._authorized_scope(scope.require("project_id"))
        )

    async def register_skill(self, request: RegisterRequest) -> SkillDetail:
        """Register a complete source under a new central identity.

        Args:
            request: Project, source content, display metadata, and provenance.
        """
        request = self._scoped(RegisterRequest.model_validate(request))
        return await self._execute(request.project_id, self.repository.register_skill, request)

    async def list_skills(self, request: ListRequest) -> list[SkillRecord]:
        """List central records in one project.

        Args:
            request: Project and pagination settings.
        """
        request = self._scoped(ListRequest.model_validate(request))
        return await self._execute(request.project_id, self.repository.list_skills, request)

    async def show_skill(self, request: ShowRequest) -> SkillDetail:
        """Read central metadata and an owned version.

        Args:
            request: Project, Skill reference, and optional version.
        """
        request = self._scoped(ShowRequest.model_validate(request))
        return await self._execute(request.project_id, self.repository.show_skill, request)

    async def skill_cloud_status(self, request: SkillRequest) -> str:
        """Read recorded cloud synchronization status for a local Skill.

        Args:
            request: Project and Skill reference.
        """
        request = self._scoped(SkillRequest.model_validate(request))
        return await self._execute(request.project_id, self.repository.skill_cloud_status, request)

    async def export_skill(self, request: ExportRequest) -> str:
        """Export a complete stored snapshot without tracking the directory.

        Args:
            request: Skill/version selection and explicit destination replacement policy.

        Returns:
            Absolute path of the exported application directory.
        """
        request = self._scoped(ExportRequest.model_validate(request))
        return await self._execute(request.project_id, self.repository.export_skill, request)

    async def commit_skill(self, request: CommitRequest) -> SkillDetail:
        """Save a new version and select it only if the captured parent is still current.

        Args:
            request: Target Skill, source, optional expected parent, and commit message.
        """
        request = self._scoped(CommitRequest.model_validate(request))
        return await self._execute(request.project_id, self.repository.commit_skill, request)

    async def skill_history(self, request: HistoryRequest) -> list[SkillVersionRecord]:
        """Read a page of versions without changing the current selection.

        Args:
            request: Project, Skill reference, and pagination.
        """
        request = self._scoped(HistoryRequest.model_validate(request))
        return await self._execute(request.project_id, self.repository.skill_history, request)

    async def diff_skill(self, request: DiffRequest) -> DiffResult:
        """Compare complete snapshots from the same Skill.

        Args:
            request: Skill and version pair; the starting version defaults to current.
        """
        request = self._scoped(DiffRequest.model_validate(request))
        return await self._execute(request.project_id, self.repository.diff_skill, request)

    async def switch_skill(self, request: SwitchRequest) -> SkillRecord:
        """Change the current pointer without editing any application directory.

        Args:
            request: Skill and existing owned version to select.
        """
        request = self._scoped(SwitchRequest.model_validate(request))
        return await self._execute(request.project_id, self.repository.switch_skill, request)

    async def set_skill_meta(self, request: SetMetaRequest) -> SkillRecord:
        """Update name or alias without creating a version.

        Args:
            request: Skill reference and explicit metadata changes.
        """
        request = self._scoped(SetMetaRequest.model_validate(request))
        return await self._execute(request.project_id, self.repository.set_skill_meta, request)

    async def delete_skill(self, request: SkillRequest) -> None:
        """Delete central data only when no retained evidence references it.

        Args:
            request: Project and Skill identity, alias, or unique prefix.
        """
        request = self._scoped(SkillRequest.model_validate(request))
        return await self._execute(request.project_id, self.repository.delete_skill, request)

    async def auto_evolve_skill(self, request: SkillPipelineRequest | None = None) -> SkillPipelineResult:
        """Delegate evolution to the selected pipeline.

        Args:
            request: Skill/version selection; required when a pipeline is configured.

        Raises:
            NotImplementedError: No evolution pipeline is configured.
            ValueError: Request is missing, or pipeline selection is missing or unavailable.
        """
        if not self._evolution_pipelines:
            raise NotImplementedError("Automatic Skill evolution is not implemented")
        if request is None:
            raise ValueError("An evolution request is required")
        request = SkillPipelineRequest.model_validate(request)
        scoped = request.model_copy(update={"project_id": self._project_id(request.project_id)})
        pipeline = _select_pipeline(self._evolution_pipelines, request.pipeline)
        return await self._execute(scoped.project_id, pipeline.evolve, scoped)

    async def auto_train_skill(self, request: TrainingRequest | None = None) -> SkillPipelineResult:
        """Delegate tasks, splits, and an exact input version to the training pipeline.

        Args:
            request: Training inputs; required when a pipeline is configured.

        Raises:
            NotImplementedError: No training pipeline is configured.
            ValueError: Request is missing, or pipeline selection is missing or unavailable.
        """
        if not self._training_pipelines:
            raise NotImplementedError("Automatic Skill training is not implemented")
        if request is None:
            raise ValueError("A training request is required")
        request = TrainingRequest.model_validate(request)
        scoped = request.model_copy(update={"project_id": self._project_id(request.project_id)})
        pipeline = _select_pipeline(self._training_pipelines, request.pipeline)
        run_id = training_run_id(scoped.project_id, scoped.work_dir)

        async def run_training():
            """Keep audit publication within the authorized request scope."""
            with self._observability.trace_run(
                project_id=scoped.project_id,
                run_id=run_id,
                algorithm=request.pipeline or next(iter(self._training_pipelines)),
                base_version_id=scoped.base_version_id,
            ) as audit:
                result = await pipeline.train(scoped)
                audit.published_version_id = result.version_id
                return result

        return await self._execute(scoped.project_id, run_training)
