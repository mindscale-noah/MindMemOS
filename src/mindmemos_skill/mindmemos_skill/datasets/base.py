"""Task dataset split contract."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..typing import Task


class TaskDataset(ABC):
    """Expose trainer-ready task splits."""

    @abstractmethod
    def split(self, name: str) -> list[Task]:
        """Return tasks for ``train``, ``validation`` or ``test``."""

    def train_tasks(self) -> list[Task]:
        return self.split("train")

    def validation_tasks(self) -> list[Task]:
        return self.split("validation")

    def test_tasks(self) -> list[Task]:
        return self.split("test")


__all__ = ["TaskDataset"]
