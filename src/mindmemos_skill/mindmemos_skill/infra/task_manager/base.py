"""Common task-manager contract for local and durable execution."""

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable

from .progress import ProgressReporter
from .typing import ProgressEvent, TaskSnapshot, TaskSpec

TaskHandler = Callable[[TaskSpec, ProgressReporter], Awaitable[dict]]


class TaskManager(ABC):
    """Schedule and inspect work without exposing its transport to callers."""

    @abstractmethod
    async def submit(self, spec: TaskSpec) -> TaskSnapshot:
        """Admit work without waiting for its result; queue capacity may require waiting."""

    @abstractmethod
    async def get(self, project_id: str, task_id: str) -> TaskSnapshot | None:
        """Read a project-owned task or return None."""

    @abstractmethod
    async def events(self, project_id: str, task_id: str, *, after: int = 0) -> list[ProgressEvent]:
        """Read algorithm-selected progress after the given sequence."""

    @abstractmethod
    async def close(self) -> None:
        """Release execution resources owned by this manager."""
