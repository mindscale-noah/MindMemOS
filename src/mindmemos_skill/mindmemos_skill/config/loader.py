"""Resolve and validate explicit Skill configuration before constructing resources."""

from inspect import signature
from pathlib import Path

import yaml
from dotenv import load_dotenv
from sqlalchemy.engine import URL

from ..agents.config import AgentCatalog
from ..infra import Infrastructure
from ..infra.database import Database
from ..infra.database.config import SQLiteConfig
from ..pipelines import AnalyzePipeline, EvolutionPipeline, ManagePipeline, Pipeline, TrainingPipeline
from ..pipelines.builtin_pipelines.repository.repository import RepositoryPipeline
from .config import ImplementationConfig, ServiceConfig, SkillConfig


def _validate_pipeline(spec: ImplementationConfig, family: type[Pipeline], agents: AgentCatalog) -> None:
    """Validate a pipeline's family, typed options and fixed persistence injection."""
    implementation = Pipeline.resolve(spec.class_name)
    if not issubclass(implementation, family):
        raise ValueError(f"Pipeline {spec.class_name!r} must implement {family.__name__}")
    spec.config = implementation.config_type.model_validate(spec.config, context=agents)
    try:
        signature(implementation).bind(None, persistence=None)
    except TypeError as exc:
        raise TypeError(f"Pipeline {spec.class_name!r} must accept config and persistence: {exc}") from exc


def load_skill_config(content: str) -> SkillConfig:
    """Parse and preflight deployment YAML without constructing resources.

    Args:
        content: YAML containing service resources and HTTP authentication policy.

    Returns:
        Complete deployment configuration containing implementation-owned models.
    """
    config = SkillConfig.model_validate(yaml.safe_load(content))
    validate_service_config(config.service)
    return config


def read_skill_config(path: str | Path) -> SkillConfig:
    """Read a deployment YAML and resolve its known file paths.

    Args:
        path: Configuration file whose directory anchors home and the local key file.

    Returns:
        Validated configuration with absolute home and local key-file paths.
    """
    source = Path(path).expanduser().resolve()
    load_dotenv(source.with_suffix(".env"), override=False, interpolate=False)
    config = load_skill_config(source.read_text(encoding="utf-8"))
    config.service.home = _resolve_path(config.service.home, source.parent)
    if config.fastapi.auth.api_key_file is not None:
        config.fastapi.auth.api_key_file = str(_resolve_path(Path(config.fastapi.auth.api_key_file), source.parent))
    return config


def _resolve_path(path: Path, base: Path) -> Path:
    """Expand a path and anchor relative paths to the given directory."""
    return (base / path.expanduser()).resolve()


def validate_service_config(config: ServiceConfig) -> None:
    """Check registered types and nested options without connecting or constructing backends.

    Args:
        config: Service definition to preflight.
    """
    agents = AgentCatalog(agents=config.agents)
    if config.persistence is not None:
        database_spec = config.persistence.database
        database_type = Infrastructure.resolve(database_spec.class_name)
        if not issubclass(database_type, Database):
            raise ValueError("persistence.database must select a Database implementation")
        database_spec.config = database_type.config_type.model_validate(database_spec.config)
        signature(database_type).bind(None)
    if config.task_manager is not None and "tasks" not in config.disabled_services and config.persistence is None:
        raise ValueError("Skill tasks require the existing Skill persistence database")
    if (
        "observability" not in config.disabled_services
        and config.observability is not None
        and config.observability.backend is not None
    ):
        backend_spec = config.observability.backend
        backend_type = Infrastructure.resolve(backend_spec.class_name)
        if not issubclass(backend_type, Database):
            raise ValueError("observability.backend must select a Database implementation")
        backend_spec.config = backend_type.config_type.model_validate(backend_spec.config)
    if "repository" not in config.disabled_services and config.repository_pipeline is not None:
        if config.persistence is None:
            raise ValueError("repository_pipeline requires global persistence")
        _validate_pipeline(config.repository_pipeline, RepositoryPipeline, agents)
    if "manage" not in config.disabled_services and config.manage_pipeline is not None:
        if config.persistence is None:
            raise ValueError("manage_pipeline requires global persistence")
        _validate_pipeline(config.manage_pipeline, ManagePipeline, agents)
    if "analyze" not in config.disabled_services and config.analyze_pipeline is not None:
        if config.persistence is None:
            raise ValueError("analyze_pipeline requires global persistence")
        _validate_pipeline(config.analyze_pipeline, AnalyzePipeline, agents)
    if "training" not in config.disabled_services:
        for spec in config.training_pipelines.values():
            _validate_pipeline(spec, TrainingPipeline, agents)
    if "evolution" not in config.disabled_services:
        for spec in config.evolution_pipelines.values():
            _validate_pipeline(spec, EvolutionPipeline, agents)


def resolve_service_config(config: ServiceConfig) -> ServiceConfig:
    """Copy resource settings and fill omitted SQLite URLs from the explicit home.

    Args:
        config: Resource configuration with YAML-relative home already anchored.

    Returns:
        Independent configuration containing validated implementation-owned models.
    """
    resolved = ServiceConfig.model_validate(config.model_dump(exclude_unset=True))
    resolved.home = resolved.home.expanduser().resolve()
    if resolved.persistence is not None:
        _fill_sqlite_url(resolved.persistence.database, resolved.home / "data" / "central.db")
    if resolved.observability is not None and resolved.observability.backend is not None:
        _fill_sqlite_url(resolved.observability.backend, resolved.home / "data" / "observability.db")
    validate_service_config(resolved)
    return resolved


def _fill_sqlite_url(database: ImplementationConfig, path: Path) -> None:
    """Fill an omitted SQLite URL while preserving explicit connection settings."""
    if database.class_name != "sqlite":
        return
    if isinstance(database.config, SQLiteConfig):
        if "url" in database.config.model_fields_set:
            return
        options = database.config.model_dump(exclude_unset=True)
    elif isinstance(database.config, dict):
        if "url" in database.config:
            return
        options = database.config
    else:
        raise TypeError("SQLite selection requires SQLiteConfig")
    url = URL.create("sqlite+aiosqlite", database=str(path)).render_as_string(hide_password=False)
    database.config = SQLiteConfig.model_validate({**options, "url": url})
