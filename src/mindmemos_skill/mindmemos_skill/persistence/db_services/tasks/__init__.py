"""Task persistence on the Skill service database."""

from .service import TaskDataService
from .typing import TaskRecord

__all__ = ["TaskDataService", "TaskRecord"]
