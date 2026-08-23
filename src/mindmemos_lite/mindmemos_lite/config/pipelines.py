"""Configuration for routing memory modes to algorithm pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..errors import InvalidConfigError
from .base import MindMemOSConfig
from .validation import join_path, require_string


@dataclass
class MemoryModePipelineConfig(MindMemOSConfig):
    """Bind one public memory mode to its add and search implementations."""

    add_pipeline: str = "vanilla_add"
    """Registered add-pipeline name used when mixed add dispatches this mode."""

    search_pipeline: str = "vanilla_search"
    """Registered search-pipeline name used when a request selects this mode."""

    @classmethod
    def validate_self(cls, value, path: str) -> None:
        require_string(join_path(path, "add_pipeline"), value.add_pipeline)
        require_string(join_path(path, "search_pipeline"), value.search_pipeline)


def _default_mode_pipelines() -> dict[str, MemoryModePipelineConfig]:
    return {"vanilla": MemoryModePipelineConfig()}


@dataclass
class MixedAddPipelineConfig(MindMemOSConfig):
    """Select the memory modes that process every inferred add request."""

    modes: list[str] = field(default_factory=lambda: ["vanilla"])
    """Ordered mode names dispatched concurrently by ``MixedAddPipeline``."""

    @classmethod
    def validate_self(cls, value, path: str) -> None:
        if not value.modes:
            raise InvalidConfigError(join_path(path, "modes"), support="at least one configured memory mode")
        seen: set[str] = set()
        for index, mode in enumerate(value.modes):
            require_string(f"{join_path(path, 'modes')}[{index}]", mode)
            if mode in seen:
                raise InvalidConfigError(
                    f"{join_path(path, 'modes')}[{index}]",
                    support="unique memory mode names",
                )
            seen.add(mode)


@dataclass
class PipelineRoutingConfig(MindMemOSConfig):
    """Top-level mapping from stable public modes to concrete pipelines."""

    default_search_mode: str = "vanilla"
    """Mode used when a search request omits ``memory_mode``."""

    default_add_pipeline: str = "trajectory_add"
    """Registered add-pipeline name used for every inferred add request.

    Pipelines that declare ``requires_task`` (for example ``trajectory_add``)
    require a non-empty ``task`` on each add request; configure ``vanilla_add``
    or ``mixed_add`` here to accept task-less adds.
    """

    default_search_pipeline: str = "task_experience_search"
    """Registered search-pipeline name used by every ``/v1/memory/search`` request.

    ``task_experience_search`` treats the query as a task text and returns that
    task plus its one-hop experiences (the default). Configure ``vanilla_search``
    for a direct vanilla hybrid search, or ``mode_search`` to keep request-time
    ``memory_mode`` routing.
    """

    modes: dict[str, MemoryModePipelineConfig] = field(default_factory=_default_mode_pipelines)
    """All modes exposed to mixed add and mode-aware search."""

    mixed_add: MixedAddPipelineConfig = field(default_factory=MixedAddPipelineConfig)
    """Concurrent add fan-out configuration."""

    @classmethod
    def validate_self(cls, value, path: str) -> None:
        require_string(join_path(path, "default_search_mode"), value.default_search_mode)
        require_string(join_path(path, "default_add_pipeline"), value.default_add_pipeline)
        require_string(join_path(path, "default_search_pipeline"), value.default_search_pipeline)
        if not value.modes:
            raise InvalidConfigError(join_path(path, "modes"), support="at least one memory mode binding")

        for mode, binding in value.modes.items():
            require_string(join_path(join_path(path, "modes"), str(mode)), mode)
            if binding.add_pipeline == "mixed_add":
                raise InvalidConfigError(
                    join_path(join_path(join_path(path, "modes"), mode), "add_pipeline"),
                    support="a concrete child add pipeline, not mixed_add itself",
                )

        if value.default_search_mode not in value.modes:
            raise InvalidConfigError(
                join_path(path, "default_search_mode"),
                support="one of the configured pipeline mode names",
            )
        unknown_modes = [mode for mode in value.mixed_add.modes if mode not in value.modes]
        if unknown_modes:
            raise InvalidConfigError(
                join_path(join_path(path, "mixed_add"), "modes"),
                support=f"configured mode names; unknown: {', '.join(unknown_modes)}",
            )


__all__ = [
    "MemoryModePipelineConfig",
    "MixedAddPipelineConfig",
    "PipelineRoutingConfig",
]
