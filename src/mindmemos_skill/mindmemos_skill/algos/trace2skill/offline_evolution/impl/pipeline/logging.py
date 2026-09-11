"""Private logging and stage-display helpers shared by pipeline sub-modules."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from mindmemos_skill.algos.trace2skill.offline_evolution.impl.pipeline.stage import (
    PREPARE_STAGE,
    STAGE_ID_BY_NUMBER,
    STAGE_NUMBER_BY_ID,
    TOTAL_USER_STAGES,
    get_stage_class,
)

LogFn = Callable[[str], None]


def format_duration(seconds: float) -> str:
    """Return a compact human-readable duration for pipeline logs."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {sec:.1f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {sec:.1f}s"


def _log(logger: LogFn | None, message: str) -> None:
    if logger is not None:
        logger(message)


def _display_path(path: str | Path, *, root: Path | None = None) -> str:
    """Format paths compactly for logs."""
    p = Path(path)
    if root is not None:
        try:
            return str(p.resolve().relative_to(root.resolve())) or "."
        except (FileNotFoundError, ValueError):
            try:
                return str(p.relative_to(root)) or "."
            except ValueError:
                pass
    return str(p)


def _stage_label(stage_id: str | None) -> str:
    if stage_id is None:
        return "none"
    if stage_id == PREPARE_STAGE:
        return "staging"
    number = STAGE_NUMBER_BY_ID.get(stage_id)
    name = get_stage_class(stage_id).title
    if number is None or name is None:
        return stage_id
    return f"stage {number} ({name})"


def _stop_after_stage_id(stop_after_stage: int | None) -> str | None:
    if stop_after_stage is None or stop_after_stage == TOTAL_USER_STAGES:
        return None
    if stop_after_stage not in STAGE_ID_BY_NUMBER:
        raise ValueError(f"stop_after_stage must be between 1 and {TOTAL_USER_STAGES}")
    return STAGE_ID_BY_NUMBER[stop_after_stage]


__all__ = [
    "LogFn",
    "format_duration",
    "_log",
    "_display_path",
    "_stage_label",
    "_stop_after_stage_id",
]
