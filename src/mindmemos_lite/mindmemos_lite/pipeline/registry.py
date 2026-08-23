"""Registration and configuration-aware creation of pipeline classes."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from ..config import get_config
from .base import PipelineBase

PipelineType = str

_VALID_PIPELINE_TYPES = {
    "add",
    "search",
    "get",
    "delete",
    "update",
    "feedback",
    "dreaming",
    "skill_evolve",
}
_PIPELINE_REGISTRY: dict[PipelineType, dict[str, type[PipelineBase]]] = {}
_BUILTINS_LOADED = False


def register(*, type: PipelineType, name: str):
    """Register a pipeline class without constructing an instance."""

    if type not in _VALID_PIPELINE_TYPES:
        valid = ", ".join(sorted(_VALID_PIPELINE_TYPES))
        raise ValueError(f"Unknown pipeline type {type!r}. Valid pipeline types: {valid}")
    if not name:
        raise ValueError("pipeline name must not be empty")

    def decorator(cls: type[PipelineBase]) -> type[PipelineBase]:
        if not issubclass(cls, PipelineBase):
            raise TypeError(f"{cls.__name__} must inherit PipelineBase")

        pipelines = _PIPELINE_REGISTRY.setdefault(type, {})
        if name in pipelines:
            raise ValueError(f"{type} pipeline {name!r} is already registered")
        pipelines[name] = cls
        return cls

    return decorator


def create_pipeline(
    *,
    type: PipelineType,
    name: str,
    config: Any | None = None,
    **kwargs: Any,
) -> PipelineBase:
    """Create one registered pipeline using the effective configuration.

    When ``config`` is omitted, :func:`mindmemos_lite.config.get_config` supplies
    the current ContextVar-bound config, so project-scoped overrides naturally
    reach ``PipelineBase.from_config``.
    """

    load_builtin_pipelines()
    pipeline_cls = _PIPELINE_REGISTRY.get(type, {}).get(name)
    if pipeline_cls is None:
        available = ", ".join(sorted(_PIPELINE_REGISTRY.get(type, {}))) or "<none>"
        raise ValueError(f"Unknown {type} pipeline {name!r}. Available {type} pipelines: {available}")

    return pipeline_cls.from_config(
        get_config() if config is None else config,
        **kwargs,
    )


def pipeline_requires_task(*, type: PipelineType, name: str) -> bool:
    """Return whether a registered pipeline class declares ``requires_task``.

    Reads the class attribute without constructing an instance, so callers can
    enforce a mandatory ``task`` field based purely on the configured pipeline.
    """

    load_builtin_pipelines()
    pipeline_cls = _PIPELINE_REGISTRY.get(type, {}).get(name)
    if pipeline_cls is None:
        available = ", ".join(sorted(_PIPELINE_REGISTRY.get(type, {}))) or "<none>"
        raise ValueError(f"Unknown {type} pipeline {name!r}. Available {type} pipelines: {available}")
    return bool(getattr(pipeline_cls, "requires_task", False))


def load_builtin_pipelines() -> None:
    """Import registered algorithm modules when the Lite package provides them.

    Importing is delayed so config and database composition remain under the
    runtime owner rather than module import side effects.
    """

    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return

    _BUILTIN_MODULES: tuple[str, ...] = (
        ".dreaming.consolidation",
        ".feedback.default",
        ".mixed_memory.add",
        ".mixed_memory.search",
        ".task_experience.add",
        ".task_experience.search",
        ".vanilla_memory.add",
        ".vanilla_memory.search",
    )
    for module_name in _BUILTIN_MODULES:
        import_module(module_name, package=__package__)

    _BUILTINS_LOADED = True


__all__ = [
    "PipelineType",
    "create_pipeline",
    "load_builtin_pipelines",
    "pipeline_requires_task",
    "register",
]
