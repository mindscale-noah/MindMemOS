"""Public service interface and construction entry point."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import MindMemOSSkill
    from .factory import create_service


def __getattr__(name: str):
    """Load public service entry points when requested."""
    if name == "MindMemOSSkill":
        from .base import MindMemOSSkill

        return MindMemOSSkill
    if name == "create_service":
        from .factory import create_service

        return create_service
    raise AttributeError(name)


__all__ = ["MindMemOSSkill", "create_service"]
