"""Assemble the Skill service from explicit configuration or local defaults."""

import os
import tempfile
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack
from inspect import signature
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml
from dotenv import load_dotenv
from filelock import AsyncFileLock
from mindmemos_sdk.config.paths import get_mindmemos_home
from sqlalchemy.engine import URL

from ..infra import Infrastructure
from ..infra.database import Database, Scope
from ..persistence.observability import AlgorithmAuditPersistence
from ..persistence.store import SkillPersistence
from ..pipelines import EvolutionPipeline, Pipeline, TrainingPipeline
from ..pipelines.repository import RepositoryPipeline
from .config import ImplementationConfig, LocalServiceConfig, PersistenceConfig, ServiceConfig
from .service import SkillService

SERVICE_CONFIG_FILENAME = "service.yaml"
"""Optional service definition inside the selected local home."""


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
    local_root = None
    scope = None
    if config is None:
        local_root, settings = resolve_local_settings(home)
        scope = LocalServiceConfig(project_id=settings.project_id if project_id is None else project_id).project_id
        source = local_root / SERVICE_CONFIG_FILENAME
        load_dotenv(source.with_suffix(".env"), override=False, interpolate=False)
        resolved = (
            load_service_config(source.read_text(encoding="utf-8")) if source.exists() else default_service_config()
        )
        if resolved.local_project_id is not None and project_id is None:
            scope = resolved.local_project_id
        if database_path is not None:
            if resolved.persistence is None or resolved.persistence.database.class_name != "sqlite":
                raise ValueError("database_path requires SQLite persistence")
            path = Path(database_path).expanduser().resolve()
            resolved.persistence.database.config["url"] = URL.create(
                "sqlite+pysqlite", database=str(path)
            ).render_as_string(hide_password=False)
            path.parent.mkdir(parents=True, exist_ok=True)
    else:
        if any(value is not None for value in (home, database_path, project_id)):
            raise ValueError("Explicit config cannot be combined with home, database_path or project_id")
        if isinstance(config, ServiceConfig):
            resolved = ServiceConfig.model_validate(config.model_dump())
        else:
            source = Path(config).expanduser()
            load_dotenv(source.with_suffix(".env"), override=False, interpolate=False)
            resolved = load_service_config(source.read_text(encoding="utf-8"))
        scope = resolved.local_project_id
    if scope is not None and project_authorizer is not None:
        raise ValueError("Local project binding cannot be combined with a cloud project authorizer")
    if resolved.persistence is not None:
        database = resolved.persistence.database
        if database.class_name == "sqlite" and "url" not in database.config:
            root, settings = resolve_local_settings(local_root)
            path = (root / Path(settings.database_path).expanduser()).resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            database.config["url"] = URL.create("sqlite+pysqlite", database=str(path)).render_as_string(
                hide_password=False
            )

    service = await _assemble_service(resolved, local_project_id=scope, project_authorizer=project_authorizer)
    try:
        async with AsyncExitStack() as startup:
            if local_root is not None:
                await startup.enter_async_context(AsyncFileLock(str(local_root / ".workspace.lock")))
            await service.initialize_storage()
    except BaseException:
        await service.close()
        raise
    return service


def resolve_local_settings(home: str | Path | None = None) -> tuple[Path, LocalServiceConfig]:
    """Load or atomically publish workspace metadata, preserving concurrent initializers.

    Args:
        home: Explicit root, otherwise the shared SDK home resolver is used.
    """
    root = (Path(home).expanduser() if home is not None else get_mindmemos_home()).resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "workspace.json"
    if not manifest.exists():
        config = LocalServiceConfig(project_id=str(uuid4()))
        fd, temporary = tempfile.mkstemp(dir=root, prefix=".workspace-")
        try:
            with os.fdopen(fd, "w") as stream:
                stream.write(config.model_dump_json(indent=2) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, manifest)
            except FileExistsError:
                pass
        finally:
            os.unlink(temporary)
    return root, LocalServiceConfig.model_validate_json(manifest.read_text())


def _validate_pipeline(spec: ImplementationConfig, family: type[Pipeline]) -> None:
    """Validate a pipeline's family, typed options and fixed persistence injection."""
    implementation = Pipeline.resolve(spec.class_name)
    if not issubclass(implementation, family):
        raise ValueError(f"Pipeline {spec.class_name!r} must implement {family.__name__}")
    implementation.config_type.model_validate(spec.config)
    try:
        signature(implementation).bind(None, persistence=None)
    except TypeError as exc:
        raise TypeError(f"Pipeline {spec.class_name!r} must accept config and persistence: {exc}") from exc


def default_service_config() -> ServiceConfig:
    """Return local defaults whose omitted SQLite URL resolves through workspace metadata."""
    return ServiceConfig(
        persistence=PersistenceConfig(
            config={"include_remote": False},
            database=ImplementationConfig(class_name="sqlite"),
        ),
        repository_pipeline=ImplementationConfig(class_name="repository"),
    )


def load_service_config(content: str) -> ServiceConfig:
    """Parse and preflight YAML without constructing resources.

    Args:
        content: Complete service YAML document.

    Returns:
        Validated service definition.
    """
    config = ServiceConfig.model_validate(yaml.safe_load(content))
    validate_service_config(config)
    return config


def validate_service_config(config: ServiceConfig) -> None:
    """Check registered types and nested options without connecting or constructing backends.

    Args:
        config: Service definition to preflight.
    """
    if config.persistence is not None:
        database_spec = config.persistence.database
        database_type = Infrastructure.resolve(database_spec.class_name)
        if not issubclass(database_type, Database):
            raise ValueError("persistence.database must select a Database implementation")
        database_type.config_type.model_validate(database_spec.config)
        signature(database_type).bind(None)
    if config.observability is not None and config.observability.backend is not None:
        backend_spec = config.observability.backend
        backend_type = Infrastructure.resolve(backend_spec.class_name)
        if not issubclass(backend_type, Database):
            raise ValueError("observability.backend must select a Database implementation")
        backend_type.config_type.model_validate(backend_spec.config)
    if config.repository_pipeline is not None:
        if config.persistence is None:
            raise ValueError("repository_pipeline requires global persistence")
        _validate_pipeline(config.repository_pipeline, RepositoryPipeline)
    for spec in config.training_pipelines.values():
        _validate_pipeline(spec, TrainingPipeline)
    for spec in config.evolution_pipelines.values():
        _validate_pipeline(spec, EvolutionPipeline)


async def _assemble_service(
    config: ServiceConfig,
    *,
    local_project_id: str | None,
    project_authorizer: Callable[[object, str], Scope | Awaitable[Scope]] | None = None,
) -> SkillService:
    """Validate definitions, build one store, and inject it into independent pipelines."""
    validate_service_config(config)
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
        if config.observability is not None and config.observability.backend is not None:
            backend_spec = config.observability.backend
            observability_database = Infrastructure.build(backend_spec.class_name, backend_spec.config)
            backends["observability_database"] = observability_database
        otlp = config.observability.otlp if config.observability is not None else None
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
            if config.repository_pipeline is not None
            else None
        )
        training = {
            name: build_pipeline(f"training_pipelines.{name}", spec) for name, spec in config.training_pipelines.items()
        }
        evolution = {
            name: build_pipeline(f"evolution_pipelines.{name}", spec)
            for name, spec in config.evolution_pipelines.items()
        }
        return SkillService(
            backends=backends,
            repository=repository,
            training_pipelines=training,
            evolution_pipelines=evolution,
            observability=observability,
            local_project_id=local_project_id,
            persistence=persistence,
            project_authorizer=project_authorizer,
        )
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
