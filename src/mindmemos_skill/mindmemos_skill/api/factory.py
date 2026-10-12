"""Assemble the Skill service from explicit deployment configuration."""

from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from filelock import AsyncFileLock

from ..config.config import FastAPIConfig, ImplementationConfig, ServiceConfig, SkillConfig
from ..config.loader import read_skill_config, resolve_service_config, validate_service_config
from ..infra import Infrastructure
from ..infra.database import Database, Scope
from ..infra.task_manager import LocalTaskManager
from ..infra.task_manager.kafka import KafkaTaskManager
from ..persistence.db_services.tasks import TaskDataService
from ..persistence.observability import AlgorithmAuditPersistence
from ..persistence.store import SkillPersistence
from ..pipelines import Pipeline
from .service import SkillService


async def create_service(
    config: SkillConfig | ServiceConfig | str | Path,
    *,
    project_authorizer: Callable[[object, str], Scope | Awaitable[Scope]] | None = None,
) -> SkillService:
    """Assemble and initialize resources from explicit deployment configuration.

    Args:
        config: Complete YAML or SkillConfig; ServiceConfig constructs unbound resources
            for a trusted host that supplies request identity and project authorization.
        project_authorizer: Host authorization callback for authenticated requests.

    Returns:
        Initialized service owned by the caller. Only none mode binds a default project.

    Raises:
        ValueError: Configuration or authentication settings are inconsistent.
        OSError: Configuration or storage files cannot be accessed.
    """
    policy = None
    if isinstance(config, ServiceConfig):
        resources = config
    else:
        deployment = (
            SkillConfig.model_validate(config.model_dump(exclude_unset=True))
            if isinstance(config, SkillConfig)
            else read_skill_config(config)
        )
        policy = deployment.fastapi
        resources = deployment.service
    resolved = resolve_service_config(resources)
    scope = policy.auth.default_project_id if policy is not None and policy.auth.mode == "none" else None
    if scope is not None and project_authorizer is not None:
        raise ValueError("Unauthenticated project binding cannot be combined with a project authorizer")
    if policy is not None and policy.auth.mode != "none" and project_authorizer is None:
        from .auth import authorize_project

        project_authorizer = authorize_project
    resolved.home.mkdir(parents=True, exist_ok=True)

    service = await _assemble_service(
        resolved, default_project_id=scope, project_authorizer=project_authorizer, api_config=policy
    )
    try:
        async with AsyncExitStack() as startup:
            await startup.enter_async_context(
                AsyncFileLock(str(resolved.home / ".storage.lock"), run_in_executor=False)
            )
            await service.initialize_storage()
            manager = service.task_manager
            if isinstance(manager, KafkaTaskManager):
                await manager.start()
    except BaseException:
        await service.close()
        raise
    return service


async def _assemble_service(
    config: ServiceConfig,
    *,
    default_project_id: str | None,
    api_config: FastAPIConfig | None,
    project_authorizer: Callable[[object, str], Scope | Awaitable[Scope]] | None = None,
) -> SkillService:
    """Validate definitions, build one store, and inject it into independent pipelines."""
    validate_service_config(config)
    disabled = config.disabled_services
    backends: dict[str, Any] = {}
    observability = None
    observability_database = None
    try:
        persistence = None
        if config.persistence is not None:
            database_spec = config.persistence.database
            database = Infrastructure.build(database_spec.class_name, database_spec.config)
            backends["database"] = database
            persistence = SkillPersistence(config.persistence.config, database=database, require_session=True)
            backends["persistence"] = persistence
        if (
            "observability" not in disabled
            and config.observability is not None
            and config.observability.backend is not None
        ):
            backend_spec = config.observability.backend
            observability_database = Infrastructure.build(backend_spec.class_name, backend_spec.config)
            backends["observability_database"] = observability_database
        otlp = (
            config.observability.otlp if config.observability is not None and "observability" not in disabled else None
        )
        observability = AlgorithmAuditPersistence(
            observability_database,
            capture_level=config.observability.capture_level if config.observability is not None else "info",
            service_name=config.observability.service_name if config.observability is not None else "mindmemos-skill",
            otlp_endpoint=str(otlp.endpoint) if otlp is not None else None,
            otlp_timeout=otlp.timeout if otlp is not None else 5.0,
        )
        if observability_database is not None:
            backends["observability"] = observability

        def build_pipeline(path: str, spec: ImplementationConfig) -> Pipeline:
            """Construct a pipeline whose typed config owns all non-storage dependencies."""
            pipeline = Pipeline.build(spec.class_name, spec.config, persistence=persistence)
            backends[path] = pipeline
            return pipeline

        repository = (
            build_pipeline("repository_pipeline", config.repository_pipeline)
            if config.repository_pipeline is not None and "repository" not in disabled
            else None
        )
        manage = (
            build_pipeline("manage_pipeline", config.manage_pipeline)
            if config.manage_pipeline is not None and "manage" not in disabled
            else None
        )
        training = {
            name: build_pipeline(f"training_pipelines.{name}", spec)
            for name, spec in config.training_pipelines.items()
            if "training" not in disabled
        }
        evolution = {
            name: build_pipeline(f"evolution_pipelines.{name}", spec)
            for name, spec in config.evolution_pipelines.items()
            if "evolution" not in disabled
        }
        service = SkillService(
            backends=backends,
            repository=repository,
            manage_pipeline=manage,
            training_pipelines=training,
            evolution_pipelines=evolution,
            observability=observability,
            default_project_id=default_project_id,
            api_config=api_config,
            persistence=persistence,
            project_authorizer=project_authorizer,
            disabled_services=config.disabled_services,
        )
        if config.task_manager is not None and "tasks" not in disabled:
            if config.task_manager.mode == "local":
                manager = LocalTaskManager(
                    service.execute_task,
                    tracer=observability.telemetry.tracer,
                    max_concurrency=config.task_manager.max_concurrency,
                    max_buffered=config.task_manager.max_buffered,
                )
            else:
                manager = KafkaTaskManager(
                    store=TaskDataService(database),
                    bootstrap_servers=config.task_manager.bootstrap_servers,
                    topic=config.task_manager.topic,
                    group_id=config.task_manager.group_id,
                    handler=service.execute_task if config.task_manager.worker else None,
                    tracer=observability.telemetry.tracer,
                    max_concurrency=config.task_manager.max_concurrency,
                    max_buffered=config.task_manager.max_buffered,
                    scan_interval_seconds=config.task_manager.scan_interval_seconds,
                )
            service.attach_task_manager(manager)
        return service
    except BaseException:
        if observability is not None and observability_database is None:
            (await observability.close())
        async with AsyncExitStack() as cleanup:
            for backend in backends.values():
                if isinstance(backend, Pipeline):
                    cleanup.push_async_callback(backend.close)
                elif isinstance(backend, Database):
                    cleanup.push_async_callback(backend.close)
                elif isinstance(backend, AlgorithmAuditPersistence):
                    cleanup.push_async_callback(backend.close)
        raise
