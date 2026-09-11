"""Stage abstraction and the fixed execution order.

:data:`PIPELINE_ORDER` is the single source of truth; the numbered CLI
maps stage numbers to indices into it. Stage lifecycle (mark running,
validate, mark done) is owned by the Runner, not the stages.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from mindmemos_skill.algos.trace2skill.offline_evolution.impl.pipeline.types import PipelinePaths

PIPELINE_ORDER: tuple[str, ...] = (
    "staging",
    "extract",
    "pattern",
    "evolve",
    "finalize",
)

# staging is unnumbered: user-facing stage numbers are 1..4.
PREPARE_STAGE: str = "staging"

STAGE_IDS: tuple[str, ...] = PIPELINE_ORDER
STAGE_NUMBER_BY_ID: dict[str, int] = {sid: i for i, sid in enumerate(PIPELINE_ORDER)}
STAGE_ID_BY_NUMBER: dict[int, str] = {i: sid for i, sid in enumerate(PIPELINE_ORDER)}

TOTAL_USER_STAGES: int = len(PIPELINE_ORDER) - 1
MIN_FORK_STAGE: int = 2  # ``pattern`` and later can be re-run from a fork.


@dataclass
class StageContext:
    """Per-stage execution context threaded in by the runner."""

    state: dict[str, Any]
    paths: PipelinePaths
    run_dir: Path
    logger: Any | None
    prev_result: "StageResult | None" = None
    settings: Any | None = None  # PipelineSettings (typed forward ref)


@dataclass
class StageResult:
    """Base class for stage return values.

    The runner calls ``validate(ctx)`` before persisting; the result is
    serialized via :meth:`to_dict` and rehydrated on resume via ``from_dict``.
    """

    def validate(self, ctx: StageContext) -> None:  # noqa: D401
        """Self-check after running. Override per stage."""
        return None

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable state.json shape for this result."""
        return {}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StageResult":
        """Reconstruct this result from its persisted state.json dict."""
        return cls()


@dataclass
class PipelineStage(ABC):
    """Base class for one pipeline stage.

    Subclasses set ``title`` and ``result_type`` and implement :meth:`run`;
    ``id`` is set by :func:`register_stage`. Stages must not mutate
    ``state["stages"]`` status -- the Runner owns the lifecycle.
    """

    id: ClassVar[str]
    title: ClassVar[str]
    result_type: ClassVar[type[StageResult]]

    @abstractmethod
    def run(self, ctx: StageContext) -> StageResult:
        """Execute this stage. Must NOT mutate state["stages"] status."""


_STAGE_REGISTRY: dict[str, type[PipelineStage]] = {}


def register_stage(stage_id: str):
    """Decorator: register a :class:`PipelineStage` subclass under ``stage_id`` and set ``cls.id``."""

    def decorate(cls: type[PipelineStage]) -> type[PipelineStage]:
        cls.id = stage_id  # type: ignore[misc]
        _STAGE_REGISTRY[stage_id] = cls
        return cls

    return decorate


def get_stage_class(stage_id: str) -> type[PipelineStage]:
    """Return the :class:`PipelineStage` subclass registered for ``stage_id``."""
    try:
        return _STAGE_REGISTRY[stage_id]
    except KeyError as exc:
        raise KeyError(f"unknown stage {stage_id!r}; known: {sorted(_STAGE_REGISTRY)}") from exc


def registered_stages() -> list[str]:
    """Names of all stages currently registered, in :data:`PIPELINE_ORDER` order."""
    return [sid for sid in PIPELINE_ORDER if sid in _STAGE_REGISTRY]


def stage_index(stage_id: str) -> int:
    """Numeric index of ``stage_id`` in :data:`PIPELINE_ORDER` (0 = prepare)."""
    try:
        return PIPELINE_ORDER.index(stage_id)
    except ValueError as exc:
        raise ValueError(f"unknown stage_id: {stage_id!r}") from exc


__all__ = [
    "PIPELINE_ORDER",
    "PREPARE_STAGE",
    "STAGE_IDS",
    "STAGE_NUMBER_BY_ID",
    "STAGE_ID_BY_NUMBER",
    "TOTAL_USER_STAGES",
    "MIN_FORK_STAGE",
    "StageContext",
    "StageResult",
    "PipelineStage",
    "register_stage",
    "get_stage_class",
    "registered_stages",
    "stage_index",
]
