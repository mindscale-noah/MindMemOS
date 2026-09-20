"""Assemble the Skill service from explicit configuration or local defaults."""

import os
import tempfile
from graphlib import TopologicalSorter
from inspect import signature
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml
from filelock import FileLock
from mindmemos_sdk.config.paths import get_mindmemos_home
from sqlalchemy.engine import URL

from ..agents import Agent
from ..algos import Algorithm
from ..envs import Environment
from ..infra import Infrastructure
from ..infra.database import Database
from ..logging import Logger
from ..persistence import Persistence
from ..persistence.store import SkillPersistence
from ..pipelines import EvolutionPipeline, Pipeline, TrainingPipeline
from ..pipelines.config import RepositoryPipelineConfig
from ..pipelines.repository import RepositoryPipeline
from ..registry import Registrable
from .config import BackendConfig, LocalServiceConfig, ServiceConfig
from .service import SkillService


def create_service(
    config: ServiceConfig | str | Path | None = None,
    *,
    home: str | Path | None = None,
    database_path: str | Path | None = None,
    project_id: str | None = None,
) -> SkillService:
    """Assemble one service, using local defaults only when config is omitted.

    Args:
        config: Explicit configuration or YAML path; storage initialization stays caller-owned.
        home: Local root; omitted local roots use the shared HOME resolver.
        database_path: Optional local SQLite override, not persisted.
        project_id: Optional local scope override, never a cloud authentication claim.

    Returns:
        The assembled service. Local defaults initialize storage and bind a scope;
        explicit configurations require project IDs on requests. The caller must close it.

    Raises:
        ValueError: Explicit configuration is mixed with local overrides, or configuration is invalid.
        OSError: A configuration or local storage path cannot be accessed.
    """
    local_root = None
    scope = None
    if config is None:
        local_root, settings = resolve_local_settings(home)
        scope = LocalServiceConfig(project_id=settings.project_id if project_id is None else project_id).project_id
        path = (
            Path(database_path).expanduser().resolve()
            if database_path is not None
            else (local_root / Path(settings.database_path).expanduser()).resolve()
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        resolved = ServiceConfig(
            repository_pipeline="repository_ops",
            backends={
                "repository_ops": BackendConfig(
                    module="pipelines",
                    class_name="repository",
                    dependencies={"persistence": "repository"},
                ),
                "database": BackendConfig(
                    module="infra",
                    class_name="sqlite",
                    config={
                        "url": URL.create("sqlite+pysqlite", database=str(path)).render_as_string(hide_password=False)
                    },
                ),
                "repository": BackendConfig(
                    module="persistence",
                    class_name="skill",
                    config={"include_remote": False},
                    dependencies={"database": "database"},
                ),
            },
        )
    else:
        if any(value is not None for value in (home, database_path, project_id)):
            raise ValueError("Explicit config cannot be combined with home, database_path or project_id")
        if isinstance(config, ServiceConfig):
            resolved = ServiceConfig.model_validate(config.model_dump())
        else:
            with Path(config).open(encoding="utf-8") as stream:
                resolved = ServiceConfig.model_validate(yaml.safe_load(stream))
        for backend in resolved.backends.values():
            if backend.module == "infra" and backend.class_name == "sqlite" and "url" not in backend.config:
                root, settings = resolve_local_settings()
                path = (root / Path(settings.database_path).expanduser()).resolve()
                path.parent.mkdir(parents=True, exist_ok=True)
                backend.config["url"] = URL.create("sqlite+pysqlite", database=str(path)).render_as_string(
                    hide_password=False
                )

    service = _assemble_service(resolved, local_project_id=scope)
    if local_root is not None:
        try:
            with FileLock(str(local_root / ".workspace.lock")):
                service.initialize_storage()
        except BaseException:
            service.close()
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


_MODULES: dict[str, type[Registrable[Any]]] = {
    "infra": Infrastructure,
    "algos": Algorithm,
    "agents": Agent,
    "envs": Environment,
    "logging": Logger,
    "persistence": Persistence,
    "pipelines": Pipeline,
}
"""Backend families sharing the existing registration mechanism."""


def _assemble_service(config: ServiceConfig, *, local_project_id: str | None) -> SkillService:
    """Validate the complete graph before constructing and selecting pipelines."""
    implementations = {}
    for name, spec in config.backends.items():
        implementation = _MODULES[spec.module].resolve(spec.class_name)
        implementation.config_type.model_validate(spec.config)
        try:
            signature(implementation).bind(None, **dict.fromkeys(spec.dependencies))
        except TypeError as exc:
            raise TypeError(f"Invalid dependencies for backend {name!r}: {exc}") from exc
        implementations[name] = implementation

    def select(name, family):
        if name is not None:
            if name not in implementations or not issubclass(implementations[name], family):
                raise ValueError(f"Backend {name!r} must select a {family.__name__}")
            return name
        candidates = [key for key, value in implementations.items() if issubclass(value, family)]
        if len(candidates) > 1:
            raise ValueError(f"Select a {family.__name__} explicitly when multiple backends are configured")
        return candidates[0] if candidates else None

    repository_name = select(config.repository_pipeline, RepositoryPipeline)
    training_name = select(config.training_pipeline, TrainingPipeline)
    evolution_name = select(config.evolution_pipeline, EvolutionPipeline)
    persistence_name = None
    if repository_name is None:
        persistence_name = select(config.persistence_backend, SkillPersistence)
    elif config.persistence_backend is not None:
        raise ValueError("Use repository_pipeline or persistence_backend, not both")
    for name, implementation in implementations.items():
        if issubclass(implementation, RepositoryPipeline):
            dependency = config.backends[name].dependencies.get("persistence")
            if dependency is None or not issubclass(implementations[dependency], SkillPersistence):
                raise ValueError("Repository pipeline requires a SkillPersistence dependency")

    backends = {}
    graph = {name: set(spec.dependencies.values()) for name, spec in config.backends.items()}
    try:
        for name in TopologicalSorter(graph).static_order():
            spec = config.backends[name]
            dependencies = {key: backends[target] for key, target in spec.dependencies.items()}
            backends[name] = _MODULES[spec.module].build(spec.class_name, spec.config, **dependencies)
        repository = backends[repository_name] if repository_name is not None else None
        if persistence_name is not None:
            repository = RepositoryPipeline(RepositoryPipelineConfig(), persistence=backends[persistence_name])
        return SkillService(
            backends=backends,
            repository=repository,
            training=backends[training_name] if training_name is not None else None,
            evolution=backends[evolution_name] if evolution_name is not None else None,
            local_project_id=local_project_id,
        )
    except BaseException:
        for backend in reversed(list(backends.values())):
            if isinstance(backend, Database):
                backend.close()
        raise
