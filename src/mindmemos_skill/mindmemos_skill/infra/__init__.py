"""Business-independent infrastructure families and registration."""

from .base import Infrastructure
from .registry import ensure_registered

ensure_registered()

__all__ = ["Infrastructure"]
