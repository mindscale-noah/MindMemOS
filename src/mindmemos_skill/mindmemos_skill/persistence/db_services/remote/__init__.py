"""Local cloud bindings; this package never changes the cloud schema."""

from .typing import (
    RemoteOperationRecord,
    RemoteSkillBindingRecord,
    RemoteTrajectoryBindingRecord,
    RemoteVersionBindingRecord,
)

__all__ = [
    "RemoteOperationRecord",
    "RemoteSkillBindingRecord",
    "RemoteTrajectoryBindingRecord",
    "RemoteVersionBindingRecord",
]
