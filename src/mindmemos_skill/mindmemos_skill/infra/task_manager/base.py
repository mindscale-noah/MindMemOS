"""Common task-manager contract for local and durable execution."""

from abc import ABC, abstractmethod

from ...logging.typing import TaskLog
from .typing import TaskSnapshot, TaskSpec


class TaskManager(ABC):
    """Schedule and inspect work without exposing its transport to callers."""

    @abstractmethod
    async def submit(self, spec: TaskSpec) -> TaskSnapshot:
        """Admit work without waiting for its result; queue capacity may require waiting."""

    @abstractmethod
    async def get(self, project_id: str, task_id: str) -> TaskSnapshot | None:
        """Read a project-owned task or return None."""

    @abstractmethod
    async def logs(self, project_id: str, task_id: str, *, after: int = 0) -> list[TaskLog]:
        """Read task logs after the given committed sequence."""

    @abstractmethod
    async def close(self) -> None:
        """Release execution resources owned by this manager."""
