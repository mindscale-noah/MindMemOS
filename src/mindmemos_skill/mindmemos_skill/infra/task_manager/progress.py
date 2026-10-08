"""Report algorithm progress to a task sink and structured logging."""

from collections.abc import Awaitable, Callable
from typing import Protocol

from ...logging import get_logger
from .typing import ProgressEvent

logger = get_logger(__name__)


class ProgressReporter(Protocol):
    """Algorithm-facing progress contract independent of execution mode."""

    async def report(self, *, stage: str, message: str, completed: int | None = None, total: int | None = None) -> None:
        """Commit one algorithm-selected progress event."""


class NullProgressReporter:
    """Ignore progress when a direct caller does not request tracking."""

    async def report(self, *, stage: str, message: str, completed: int | None = None, total: int | None = None) -> None:
        """Accept a valid progress call without a task sink."""
        ProgressEvent(stage=stage, message=message, completed=completed, total=total)


class EventProgressReporter:
    """Commit progress before emitting its correlated structured log."""

    def __init__(self, sink: Callable[[ProgressEvent], Awaitable[str]], *, task_id: str):
        """Bind a single task-specific sink and task identity.

        Args:
            sink: Commit the event and return its durable event identity.
            task_id: Stable task identity used for log correlation.
        """
        self._sink = sink
        self._task_id = task_id

    async def report(self, *, stage: str, message: str, completed: int | None = None, total: int | None = None) -> None:
        """Persist one progress event and emit the matching INFO log."""
        event = ProgressEvent(stage=stage, message=message, completed=completed, total=total)
        event_id = await self._sink(event)
        logger.info(
            "algorithm.progress",
            extra={"event_body": {"task_id": self._task_id, "event_id": event_id, **event.model_dump(mode="json")}},
        )
