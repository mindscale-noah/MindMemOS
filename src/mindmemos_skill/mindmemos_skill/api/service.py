"""Transport-independent facade delegating business operations to pipelines."""

from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack, contextmanager
from contextvars import ContextVar
from datetime import datetime
from fnmatch import fnmatchcase
from inspect import isawaitable
from typing import Any, TypeVar
from uuid import UUID, uuid4

from pydantic import JsonValue, TypeAdapter

from ..algos.manage.skill_manage.parsing import extract_frontmatter
from ..common.asyncio import run_sync
from ..common.checkpoint import fingerprint
from ..config.config import ServiceName, TrainingEnvironmentConfig
from ..envs import Environment
from ..envs.data import DataSample
from ..errors import SkillNotFoundError, SkillVersionNotFoundError
from ..infra.database import Database, Scope
from ..infra.task_manager import ProgressEvent, ProgressReporter, TaskManager, TaskSnapshot, TaskSpec
from ..persistence import SkillRecord, SkillVersionRecord, SkillVersionSummary, TrajectoryRecord, TrajectorySummary
from ..persistence.observability import AlgorithmAuditPersistence, AuditLog, AuditRun, AuditSpan
from ..persistence.store import SkillPersistence, validate_skill_scope
from ..pipelines import EvolutionPipeline, ManagePipeline, Pipeline, TrainingPipeline
from ..pipelines.builtin_pipelines.repository.repository import RepositoryPipeline
from ..pipelines.builtin_pipelines.training.training import training_run_id
from ..pipelines.config import PipelineComponentConfig
from ..skill import Skill, SkillBundle, SkillSnapshot
from ..skill.typing import DiffResult
from .base import MindMemOSSkill
from .typing import (
    CommitRequest,
    DiffRequest,
    ExportRequest,
    HistoryRequest,
    ListRequest,
    ListTrajectoriesRequest,
    OrganizeRequest,
    OrganizeResult,
    ProjectRequest,
    RecommendRequest,
    RecommendResult,
    RegisterRequest,
    ReportTrajectoryRequest,
    SetMetaRequest,
    ShowRequest,
    ShowTrajectoryRequest,
    SkillDetail,
    SkillPage,
    SkillPageRequest,
    SkillPipelineRequest,
    SkillPipelineResult,
    SkillRequest,
    SwitchRequest,
    TrainingRequest,
    VersionResolution,
    VersionResolveRequest,
)

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


def parse_skill_header(bundle: SkillBundle) -> tuple[str, str, dict[str, JsonValue]]:
    """Validate SKILL.md frontmatter and extract its display metadata.

    Args:
        bundle: Complete source bundle containing SKILL.md.

    Returns:
        Name, description, and remaining frontmatter fields.

    Raises:
        ValueError: Required frontmatter fields are missing or invalid.
    """
    content = bundle.files["SKILL.md"]
    if not content.startswith("---\n"):
        raise ValueError("SKILL.md must start with YAML frontmatter")
    frontmatter, _ = extract_frontmatter(content, version_id="SKILL.md")
    name = frontmatter.get("name")
    description = frontmatter.get("description")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("SKILL.md frontmatter requires a nonempty name")
    if not isinstance(description, str) or not description.strip():
        raise ValueError("SKILL.md frontmatter requires a nonempty description")
    metadata = TypeAdapter(dict[str, JsonValue]).validate_python(
        {key: value for key, value in frontmatter.items() if key not in {"name", "description"}}
    )
    return name, description, metadata


def _frontmatter_source(source: SkillSnapshot | str) -> tuple[SkillSnapshot, str, str, dict[str, JsonValue]]:
    """Capture one source and validate its SKILL.md metadata."""
    snapshot = Skill.from_path(source).snapshot() if isinstance(source, str) else source
    name, description, metadata = parse_skill_header(SkillBundle.deserialize(snapshot.bundle))
    return snapshot, name, description, metadata


class SkillService(MindMemOSSkill):
    """Expose injected pipelines and own the assembled backend lifecycle.

    Args:
        backends: Already constructed dependencies, ordered by construction.
        observability: Algorithm audit sink owned by this service.
        repository: Selected repository operations, or None when unconfigured.
        manage_pipeline: Skill Set management operations, or None when unconfigured.
        training_pipelines: Named training instances available to requests.
        training_environments: Named, deployment-owned environments selectable per run.
        evolution_pipelines: Named evolution instances available to requests.
        local_project_id: Scope bound only by explicit local assembly.
        persistence: Shared Skill storage, if configured.
        project_authorizer: Trusted project authorization callback for authenticated requests.
        disabled_services: Business capabilities omitted from this service.
    """

    def __init__(
        self,
        *,
        backends: dict[str, Any],
        observability: AlgorithmAuditPersistence,
        repository: RepositoryPipeline | None = None,
        manage_pipeline: ManagePipeline | None = None,
        training_pipelines: dict[str, TrainingPipeline] | None = None,
        training_environments: dict[str, TrainingEnvironmentConfig] | None = None,
        evolution_pipelines: dict[str, EvolutionPipeline] | None = None,
        local_project_id: str | None = None,
        persistence: SkillPersistence | None = None,
        project_authorizer: Callable[[object, str], Scope | Awaitable[Scope]] | None = None,
        disabled_services: set[ServiceName] | None = None,
    ) -> None:
        self._backends = backends.copy()
        self._repository = repository
        self._manage_pipeline = manage_pipeline
        self._training_pipelines = dict(training_pipelines or {})
        self._training_environments = dict(training_environments or {})
        self._evolution_pipelines = dict(evolution_pipelines or {})
        self._observability = observability
        self.local_project_id = local_project_id
        self._persistence = persistence
        self._project_authorizer = project_authorizer
        self.disabled_services = frozenset(disabled_services or ())
        self._principal: ContextVar[_PrincipalLease | None] = ContextVar("skill_service_principal", default=None)
        self._task_manager: TaskManager | None = None

    @property
    def backends(self) -> dict[str, Any]:
        """Return assembled backend instances for diagnostics and lifecycle inspection."""
        return self._backends.copy()

    @property
    def task_manager(self) -> TaskManager | None:
        """Return the configured asynchronous executor, if any."""
        return self._task_manager

    def _training_profile(self, pipeline_name: str | None, environment_name: str | None):
        """Resolve the environment profile selected for a training run."""
        _select_pipeline(self._training_pipelines, pipeline_name)
        if environment_name is None:
            if len(self._training_environments) != 1:
                raise ValueError("Select a configured training environment")
            environment_name = next(iter(self._training_environments))
        profile = self._training_environments.get(environment_name)
        if profile is None:
            raise ValueError(f"Unknown training environment {environment_name!r}")
        return environment_name, profile

    def _training_request(self, value: TrainingRequest | dict) -> TrainingRequest:
        """Restore environment-specific sample types from serialized task input."""
        if isinstance(value, TrainingRequest):
            name, _ = self._training_profile(value.pipeline, value.environment)
            return value if value.environment == name else value.model_copy(update={"environment": name})
        name, profile = self._training_profile(value.get("pipeline"), value.get("environment"))
        environment_type = Environment.resolve(profile.environment.class_name)
        raw_samples = value.get("tasks")
        if not isinstance(raw_samples, list):
            return TrainingRequest.model_validate(value)
        samples = []
        for sample in raw_samples:
            if not isinstance(sample, dict) or "task" not in sample or "reward_data" not in sample:
                raise ValueError("Each training sample requires task and reward_data")
            samples.append(
                DataSample.model_validate(
                    {
                        **sample,
                        "task": environment_type.task_type.model_validate(sample["task"]),
                        "reward_data": environment_type.reward_data_type.model_validate(sample["reward_data"]),
                    }
                )
            )
        return TrainingRequest.model_validate({**value, "environment": name, "tasks": samples})

    def _training_config_fingerprint(self, request: TrainingRequest) -> str:
        """Identify the pipeline and environment configuration selected at submission."""
        pipeline = _select_pipeline(self._training_pipelines, request.pipeline)
        _, profile = self._training_profile(request.pipeline, request.environment)
        return fingerprint(
            {
                "pipeline_class": f"{type(pipeline).__module__}.{type(pipeline).__qualname__}",
                "pipeline_config": pipeline.config.model_dump(mode="json"),
                "environment": profile.model_dump(mode="json"),
            }
        )

    async def close(self) -> None:
        """Close pipelines before their borrowed database in reverse construction order."""
        if self._task_manager is not None:
            await self._task_manager.close()
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

    async def _execute(self, project_id: str, operation, *args, **kwargs):
        """Run one pipeline call within a request-scoped persistence handle."""
        scope = await self._authorized_scope(project_id)
        if self._persistence is None:
            return await operation(*args, **kwargs)
        with self._persistence.session(scope):
            return await operation(*args, **kwargs)

    def attach_task_manager(self, manager: TaskManager) -> None:
        """Attach the factory-owned manager before the service becomes visible.

        Args:
            manager: Configured local or Kafka task manager.
        """
        if self._task_manager is not None:
            raise RuntimeError("Task manager is already attached")
        self._task_manager = manager

    async def submit_training(self, request: TrainingRequest | dict) -> TaskSnapshot:
        """Authorize and enqueue a training request without waiting for the algorithm.

        Args:
            request: Complete training input and stable work directory.

        Returns:
            Project-owned task state.
        """
        if self._task_manager is None:
            raise NotImplementedError("Asynchronous Skill tasks are not configured")
        request = self._scoped(self._training_request(request))
        await self._authorized_scope(request.project_id)
        spec = TaskSpec(
            task_id=uuid4().hex,
            project_id=request.project_id,
            kind="skill.train",
            payload={
                "request": request.model_dump(mode="json"),
                "config_fingerprint": self._training_config_fingerprint(request),
            },
        )
        return await self._task_manager.submit(spec)

    async def submit_evolution(
        self, project_id: str | None, version_id: str, *, pipeline: str | None = None, auto_accept: bool = True
    ) -> TaskSnapshot:
        """Validate one owned baseline and queue its configured evolution pipeline.

        Args:
            project_id: Authorized project or the bound local project.
            version_id: Exact baseline version identity.
            pipeline: Configured evolution pipeline name; omission requires one pipeline.
            auto_accept: Whether to select the evolved version as active.

        Returns:
            Project-owned task state.

        Raises:
            NotImplementedError: Task execution or evolution is not configured.
            SkillVersionNotFoundError: The baseline is absent from the project.
        """
        if self._task_manager is None:
            raise NotImplementedError("Asynchronous Skill tasks are not configured")
        if self._persistence is None:
            raise NotImplementedError("Repository persistence is not configured")
        if not self._evolution_pipelines:
            raise NotImplementedError("Automatic Skill evolution is not configured")
        _select_pipeline(self._evolution_pipelines, pipeline)
        pipeline_name = pipeline or next(iter(self._evolution_pipelines))
        project_id = self._project_id(project_id)

        async def prepare() -> SkillPipelineRequest:
            """Resolve the exact baseline within the authorized storage session."""
            try:
                version = await run_sync(self._persistence.skills.version_by_id, project_id, version_id)
            except SkillNotFoundError as exc:
                raise SkillVersionNotFoundError("Skill version not found in this project") from exc
            return SkillPipelineRequest(
                pipeline=pipeline_name,
                project_id=project_id,
                skill_ref=version.skill_id,
                base_version_id=version.version_id,
                auto_accept=auto_accept,
            )

        request = await self._execute(project_id, prepare)
        spec = TaskSpec(
            task_id=uuid4().hex,
            project_id=project_id,
            kind="skill.evolve",
            payload=request.model_dump(mode="json"),
        )
        return await self._task_manager.submit(spec)

    async def task_status(self, project_id: str | None, task_id: str) -> TaskSnapshot | None:
        """Read one authorized task state.

        Args:
            project_id: Requested project or the bound local project.
            task_id: Stable submitted task identity.

        Returns:
            Task state, or None when the project does not own it.
        """
        if self._task_manager is None:
            raise NotImplementedError("Asynchronous Skill tasks are not configured")
        scope = await self._authorized_scope(project_id)
        return await self._task_manager.get(scope.require("project_id"), task_id)

    async def task_progress(self, project_id: str | None, task_id: str, *, after: int = 0) -> list[ProgressEvent]:
        """Read authorized algorithm-selected progress events.

        Args:
            project_id: Requested project or the bound local project.
            task_id: Stable submitted task identity.
            after: Last observed one-based event sequence.

        Returns:
            Subsequent progress events in report order.
        """
        if self._task_manager is None:
            raise NotImplementedError("Asynchronous Skill tasks are not configured")
        scope = await self._authorized_scope(project_id)
        return await self._task_manager.events(scope.require("project_id"), task_id, after=after)

    async def execute_task(self, spec: TaskSpec, progress: ProgressReporter) -> dict:
        """Execute a previously authorized task in a trusted worker scope.

        Args:
            spec: Persisted immutable Skill request.
            progress: Reporter injected into the selected algorithm.

        Returns:
            JSON-compatible published algorithm result.
        """
        if spec.kind == "skill.train":
            payload = spec.payload
            request = self._training_request(payload["request"] if "request" in payload else payload)
            if "config_fingerprint" in payload and payload["config_fingerprint"] != self._training_config_fingerprint(
                request
            ):
                raise ValueError("Training configuration changed after task submission")
        elif spec.kind == "skill.evolve":
            request = SkillPipelineRequest.model_validate(spec.payload)
        else:
            raise ValueError("Unknown Skill task kind")
        if request.project_id != spec.project_id:
            raise ValueError("Task payload project differs from its owner")
        if self._persistence is None:
            raise RuntimeError("Skill task worker requires persistence")
        with self._persistence.session(Scope(fields={"project_id": spec.project_id})):
            if spec.kind == "skill.train":
                result = await self._run_training(request, reporter=progress)
            else:
                pipeline = _select_pipeline(self._evolution_pipelines, request.pipeline)
                result = await pipeline.evolve(request, reporter=progress)
        return result.model_dump(mode="json")

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

    async def list_skill_sets(self, project_id: str | None = None) -> list[dict]:
        """List Skill Sets in the authorized project.

        Args:
            project_id: Explicit project, or the bound local scope when omitted.

        Returns:
            Project-owned Skill Set records and publication pointers.
        """
        if self._manage_pipeline is None:
            raise NotImplementedError("Management pipeline is not configured")
        project_id = self._project_id(project_id)
        return await self._execute(project_id, self._manage_pipeline.list_skill_sets, project_id)

    async def resolve_graph(self, reference: str, project_id: str | None = None) -> dict:
        """Resolve a graph family or immutable version in the authorized project.

        Args:
            reference: Graph alias or prefixed identity.
            project_id: Explicit project, or the bound local scope.

        Returns:
            Selected public graph identity and version.
        """
        if self._manage_pipeline is None:
            raise NotImplementedError("Management pipeline is not configured")
        project_id = self._project_id(project_id)
        return await self._execute(project_id, self._manage_pipeline.resolve_graph, project_id, reference)

    async def delete_graph(self, reference: str, project_id: str | None = None) -> None:
        """Soft-delete a graph in the authorized project.

        Args:
            reference: Active graph identity or alias.
            project_id: Explicit project, or the bound local scope.
        """
        if self._manage_pipeline is None:
            raise NotImplementedError("Management pipeline is not configured")
        project_id = self._project_id(project_id)
        await self._execute(project_id, self._manage_pipeline.delete_graph, project_id, reference)

    async def resolve_skill_reference(self, reference: str, project_id: str | None = None) -> SkillDetail:
        """Resolve a Skill family or immutable version in the authorized project.

        Args:
            reference: Skill alias or prefixed identity.
            project_id: Explicit project, or the bound local scope.

        Returns:
            Owning family and selected version.
        """
        project_id = self._project_id(project_id)
        return await self._execute(project_id, self.repository.resolve_skill_reference, project_id, reference)

    async def list_skill_set_snapshots(self, skill_set_id: str, project_id: str | None = None) -> list[dict]:
        """List immutable snapshots and revisions of one Skill Set.

        Args:
            skill_set_id: Exact project-local Skill Set identity.
            project_id: Explicit project, or the bound local scope when omitted.

        Returns:
            Project-owned snapshots with their publication revisions.
        """
        if self._manage_pipeline is None:
            raise NotImplementedError("Management pipeline is not configured")
        project_id = self._project_id(project_id)
        return await self._execute(project_id, self._manage_pipeline.list_skill_set_snapshots, project_id, skill_set_id)

    async def get_skill_set_snapshot(self, skill_set_id: str, snapshot_id: UUID, project_id: str | None = None) -> dict:
        """Read one Skill Set snapshot in the authorized project.

        Args:
            skill_set_id: Exact project-local Skill Set identity.
            snapshot_id: Immutable snapshot identity.
            project_id: Explicit project, or the bound local scope when omitted.

        Returns:
            Snapshot members and relations for its representation mode.
        """
        if self._manage_pipeline is None:
            raise NotImplementedError("Management pipeline is not configured")
        project_id = self._project_id(project_id)
        return await self._execute(
            project_id, self._manage_pipeline.get_skill_set_snapshot, project_id, skill_set_id, snapshot_id
        )

    async def get_skill_version(self, version_id: str, project_id: str | None = None) -> SkillVersionRecord:
        """Read an exact Skill version in the authorized project.

        Args:
            version_id: Exact immutable Skill version identity.
            project_id: Explicit project, or the bound local scope when omitted.

        Returns:
            The project-owned Skill version record.
        """
        project_id = self._project_id(project_id)
        return await self._execute(project_id, self.repository.get_skill_version, project_id, version_id)

    async def list_skill_versions(self, skill_id: str, project_id: str | None = None) -> list[SkillVersionRecord]:
        """Read the complete version history of an exact Skill family.

        Args:
            skill_id: Exact project-local Skill family identity.
            project_id: Explicit project, or the bound local scope when omitted.

        Returns:
            Project-owned versions in creation order.
        """
        project_id = self._project_id(project_id)
        return await self._execute(project_id, self.repository.list_skill_versions, project_id, skill_id)

    async def organize_skills(
        self, request: OrganizeRequest, *, reporter: ProgressReporter | None = None
    ) -> OrganizeResult:
        """Authorize an explicit version collection before publishing its graph.

        Args:
            request: Graph identity, actor, exact versions, and optional scenes.
            reporter: Optional progress sink for graph publication.

        Returns:
            Published immutable graph snapshot identity.
        """
        if self._manage_pipeline is None:
            raise NotImplementedError("Management pipeline is not configured")
        scoped = self._scoped(OrganizeRequest.model_validate(request))
        return await self._execute(scoped.project_id, self._manage_pipeline.organize, scoped, reporter=reporter)

    async def recommend_skills(self, request: RecommendRequest) -> RecommendResult:
        """Authorize a pinned project graph before recommending versions.

        Args:
            request: Graph snapshot, public task, and selection limit.

        Returns:
            Ordered exact version IDs from the pinned snapshot.
        """
        if self._manage_pipeline is None:
            raise NotImplementedError("Management pipeline is not configured")
        scoped = self._scoped(RecommendRequest.model_validate(request))
        return await self._execute(scoped.project_id, self._manage_pipeline.recommend, scoped)

    async def register_skill(self, request: RegisterRequest) -> SkillDetail:
        """Register a complete source under a new central identity.

        Args:
            request: Project, source content, display metadata, and provenance.
        """
        request = self._scoped(RegisterRequest.model_validate(request))
        if request.require_frontmatter:
            source, name, description, metadata = _frontmatter_source(request.source)
            if request.name != name:
                raise ValueError("name must match SKILL.md frontmatter")
            request = request.model_copy(update={"source": source, "description": description, "file_meta": metadata})
        return await self._execute(request.project_id, self.repository.register_skill, request)

    async def list_skills(self, request: ListRequest) -> list[SkillRecord]:
        """List central records in one project.

        Args:
            request: Project and pagination settings.
        """
        request = self._scoped(ListRequest.model_validate(request))
        return await self._execute(request.project_id, self.repository.list_skills, request)

    async def list_skill_page(self, request: SkillPageRequest) -> SkillPage:
        """Filter a stable family page using repository and trajectory evidence.

        Args:
            request: Project scope, wildcard and actor filters, and cursor position.

        Returns:
            Matching families plus an indicator for another page.
        """
        request = self._scoped(SkillPageRequest.model_validate(request))
        after_created_at = request.after_created_at
        after_skill_id = request.after_skill_id
        matches: list[SkillRecord] = []
        while len(matches) <= request.limit:
            rows = await self.list_skills(
                ListRequest(
                    project_id=request.project_id,
                    limit=1000,
                    snapshot_at=request.snapshot_at,
                    after_created_at=after_created_at,
                    after_skill_id=after_skill_id,
                )
            )
            candidates = [
                family
                for family in rows
                if (request.name_pattern is None or fnmatchcase(family.name, request.name_pattern))
                and (
                    request.alias_pattern is None
                    or (family.alias is not None and fnmatchcase(family.alias, request.alias_pattern))
                )
            ]
            if (request.user_id is not None or request.agent_id is not None) and candidates:
                reported_ids: set[str] = set()
                for start in range(0, len(candidates), 500):
                    reported_ids.update(
                        await self._execute(
                            request.project_id,
                            self.repository.reported_skill_ids,
                            [family.skill_id for family in candidates[start : start + 500]],
                            user_id=request.user_id,
                            agent_id=request.agent_id,
                            snapshot_at=request.snapshot_at,
                        )
                    )
                candidates = [family for family in candidates if family.skill_id in reported_ids]
            for family in candidates:
                matches.append(family)
                if len(matches) > request.limit:
                    break
            if len(matches) > request.limit or len(rows) < 1000:
                break
            after_created_at = rows[-1].created_at
            after_skill_id = rows[-1].skill_id
        return SkillPage(items=matches[: request.limit], has_more=len(matches) > request.limit)

    async def report_trajectory(self, request: ReportTrajectoryRequest) -> TrajectoryRecord:
        """Persist one authenticated Agent report against its loaded version.

        Args:
            request: Authorized project and validated Harness execution.

        Returns:
            Stored trajectory with its generated identity.
        """
        request = self._scoped(ReportTrajectoryRequest.model_validate(request))
        return await self._execute(request.project_id, self.repository.report_trajectory, request)

    async def show_trajectory(self, request: ShowTrajectoryRequest) -> TrajectoryRecord:
        """Read one physical attempt in the caller's project.

        Args:
            request: Project and exact trajectory identity.
        """
        request = self._scoped(ShowTrajectoryRequest.model_validate(request))
        return await self._execute(request.project_id, self.repository.show_trajectory, request)

    async def list_trajectories(self, request: ListTrajectoriesRequest) -> list[TrajectorySummary]:
        """Page reported attempts for one family.

        Args:
            request: Project, exact family, filters, and cursor position.
        """
        request = self._scoped(ListTrajectoriesRequest.model_validate(request))
        return await self._execute(request.project_id, self.repository.list_trajectories, request)

    async def skill_has_reported_trajectory(
        self,
        project_id: str,
        skill_id: str,
        *,
        user_id: str | None,
        agent_id: str | None,
        snapshot_at: datetime,
    ) -> bool:
        """Check same-execution actor filters for one authorized family.

        Args:
            project_id: Authorized project identity.
            skill_id: Exact family identity.
            user_id: Optional business user identity.
            agent_id: Optional Agent identity.
            snapshot_at: First-page receipt cutoff.
        """
        project_id = self._project_id(project_id)
        return await self._execute(
            project_id,
            self.repository.skill_has_reported_trajectory,
            project_id,
            skill_id,
            user_id=user_id,
            agent_id=agent_id,
            snapshot_at=snapshot_at,
        )

    async def show_skill(self, request: ShowRequest) -> SkillDetail:
        """Read central metadata and an owned version.

        Args:
            request: Project, Skill reference, and optional version.
        """
        request = self._scoped(ShowRequest.model_validate(request))
        return await self._execute(request.project_id, self.repository.show_skill, request)

    async def resolve_skill_version(self, request: VersionResolveRequest) -> VersionResolution:
        """Resolve a project-scoped source hash to one immutable version.

        Args:
            request: Canonical source hash and optional exact identities.
        """
        request = self._scoped(VersionResolveRequest.model_validate(request))

        async def resolve():
            """Read the matching version and its current family pointer."""
            store = self._persistence.skills
            version = await run_sync(
                store.resolve_version_hash,
                request.project_id,
                request.content_hash,
                skill_id=request.skill_id,
                version_id=request.skill_version_id,
            )
            family = await run_sync(store.resolve_family, request.project_id, version.skill_id)
            return VersionResolution(
                skill_id=version.skill_id,
                skill_version_id=version.version_id,
                content_hash=version.content_hash,
                version_label=version.version_label,
                is_current=family.current_version_id == version.version_id,
            )

        if self._persistence is None:
            raise NotImplementedError("Repository persistence is not configured")
        return await self._execute(request.project_id, resolve)

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
        """Save a child of an owned version and optionally select it as current.

        Args:
            request: Target Skill, source, optional parent, and commit message.
        """
        request = self._scoped(CommitRequest.model_validate(request))
        if request.require_frontmatter:
            source, _, description, metadata = _frontmatter_source(request.source)
            request = request.model_copy(update={"source": source, "description": description, "file_meta": metadata})
        return await self._execute(request.project_id, self.repository.commit_skill, request)

    async def skill_history(self, request: HistoryRequest) -> list[SkillVersionSummary]:
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

    async def run_evolution(
        self, request: SkillPipelineRequest | None = None, *, reporter: ProgressReporter | None = None
    ) -> SkillPipelineResult:
        """Run the selected evolution pipeline and return its result.

        Args:
            request: Skill/version selection; required when a pipeline is configured.
            reporter: Optional progress sink for evolution milestones.

        Returns:
            Completed evolution result.

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
        if reporter is None:
            return await self._execute(scoped.project_id, pipeline.evolve, scoped)
        return await self._execute(scoped.project_id, pipeline.evolve, scoped, reporter=reporter)

    async def run_training(
        self, request: TrainingRequest | None = None, *, reporter: ProgressReporter | None = None
    ) -> SkillPipelineResult:
        """Run the selected training pipeline and return its result.

        Args:
            request: Training inputs; required when a pipeline is configured.
            reporter: Optional progress sink for training milestones.

        Returns:
            Completed training result.

        Raises:
            NotImplementedError: No training pipeline is configured.
            ValueError: Request is missing, or pipeline selection is missing or unavailable.
        """
        if not self._training_pipelines:
            raise NotImplementedError("Automatic Skill training is not implemented")
        if request is None:
            raise ValueError("A training request is required")
        request = self._training_request(request)
        scoped = request.model_copy(update={"project_id": self._project_id(request.project_id)})
        return await self._execute(scoped.project_id, self._run_training, scoped, reporter=reporter)

    async def _run_training(
        self, request: TrainingRequest, *, reporter: ProgressReporter | None = None
    ) -> SkillPipelineResult:
        """Execute one authorized training run with the same algorithm contract."""
        pipeline = _select_pipeline(self._training_pipelines, request.pipeline)
        _, profile = self._training_profile(request.pipeline, request.environment)
        run_id = training_run_id(request.project_id, request.work_dir)

        with self._observability.trace_run(
            project_id=request.project_id,
            run_id=run_id,
            algorithm=request.pipeline or next(iter(self._training_pipelines)),
            base_version_id=request.base_version_id,
        ) as audit:
            spec = profile.environment
            env = Environment.build(spec.class_name, spec.config)
            pipeline_name = request.pipeline or next(iter(self._training_pipelines))
            adapter_spec = profile.adapters.get(pipeline_name)
            adapter = (
                PipelineComponentConfig.model_validate(adapter_spec.model_dump(mode="json"))
                if adapter_spec is not None
                else None
            )
            result = await pipeline.train(request, env=env, adapter=adapter, reporter=reporter)
            audit.published_version_id = result.version_id
            return result
