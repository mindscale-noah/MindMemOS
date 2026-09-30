"""Assemble the Skill service from explicit configuration or local defaults."""

from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from filelock import AsyncFileLock

from ..config.config import ImplementationConfig, ServiceConfig
from ..config.loader import resolve_service_config, validate_service_config
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
    config: ServiceConfig | str | Path | None = None,
    *,
    home: str | Path | None = None,
    database_path: str | Path | None = None,
    project_id: str | None = None,
    project_authorizer: Callable[[object, str], Scope | Awaitable[Scope]] | None = None,
) -> SkillService:
    """Assemble one service, using local defaults only when config is omitted.

    Args:
        config: Explicit configuration or YAML path; otherwise load home/service.yaml or local defaults.
        home: Local root; omitted local roots use the shared HOME resolver.
        database_path: Optional local SQLite override, not persisted.
        project_id: Optional local scope override, never a cloud authentication claim.
        project_authorizer: Host-supplied project access check for explicit cloud configurations.

    Returns:
        A service with all configured storage initialized. Local defaults bind a
        project scope; explicit configurations require project IDs on requests.
        The caller must close the service.

    Raises:
        ValueError: Explicit configuration is mixed with local overrides, or configuration is invalid.
        OSError: A configuration or local storage path cannot be accessed.
    """
    resolved, scope, local_root = resolve_service_config(
        config, home=home, database_path=database_path, project_id=project_id
    )
    if scope is not None and project_authorizer is not None:
        raise ValueError("Local project binding cannot be combined with a cloud project authorizer")

    service = await _assemble_service(resolved, local_project_id=scope, project_authorizer=project_authorizer)
    try:
        async with AsyncExitStack() as startup:
            if local_root is not None:
                await startup.enter_async_context(AsyncFileLock(str(local_root / ".workspace.lock")))
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
    local_project_id: str | None,
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
            training_environments=config.training_environments,
            evolution_pipelines=evolution,
            observability=observability,
            local_project_id=local_project_id,
            persistence=persistence,
            project_authorizer=project_authorizer,
            disabled_services=config.disabled_services,
        )
        if config.task_manager is not None and "tasks" not in disabled:
            if config.task_manager.mode == "local":
                manager = LocalTaskManager(service.execute_task)
            else:
                manager = KafkaTaskManager(
                    store=TaskDataService(database),
                    bootstrap_servers=config.task_manager.bootstrap_servers,
                    topic=config.task_manager.topic,
                    group_id=config.task_manager.group_id,
                    handler=service.execute_task if config.task_manager.worker else None,
                    max_concurrency=config.task_manager.max_concurrency,
                    scan_interval_seconds=config.task_manager.scan_interval_seconds,
                )
            service.attach_task_manager(manager)
        return service
    except BaseException:
        if observability is not None and observability_database is None:
            observability.close()
        async with AsyncExitStack() as cleanup:
            for backend in backends.values():
                if isinstance(backend, Pipeline):
                    cleanup.push_async_callback(backend.close)
                elif isinstance(backend, Database):
                    cleanup.callback(backend.close)
                elif isinstance(backend, AlgorithmAuditPersistence):
                    cleanup.callback(backend.close)
        raise
