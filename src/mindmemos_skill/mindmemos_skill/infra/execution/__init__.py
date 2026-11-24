"""Shared cross-platform execution for environment-owned workspaces."""

from .config import ExecutionConfig
from .executor import execute

__all__ = ["ExecutionConfig", "execute"]
