"""Memory operation errors."""

from .base import MindMemOSError


class MemoryNotFoundError(MindMemOSError):
    """Raised when a project-scoped memory does not exist."""

    def __init__(self, memory_id: str) -> None:
        self.memory_id = memory_id
        super().__init__(f"memory not found: {memory_id}")


class MemoryUpdateError(MindMemOSError):
    """Raised when an in-place memory update cannot be completed safely."""
