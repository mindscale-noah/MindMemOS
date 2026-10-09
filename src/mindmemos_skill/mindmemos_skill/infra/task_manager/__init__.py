"""Task execution and progress interfaces for Skill services."""

from .base import TaskHandler, TaskManager
from .local import LocalTaskManager
from .progress import EventProgressReporter, NullProgressReporter, ProgressReporter
from .typing import ProgressEvent, TaskSnapshot, TaskSpec, TaskStatus

__all__ = [
    "EventProgressReporter",
    "LocalTaskManager",
    "NullProgressReporter",
    "ProgressEvent",
    "ProgressReporter",
    "TaskHandler",
    "TaskManager",
    "TaskSnapshot",
    "TaskSpec",
    "TaskStatus",
]
