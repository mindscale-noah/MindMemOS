"""Logging backend implementations."""

from ..registry.base import ConfigT, Registrable


class Logger(Registrable[ConfigT]):
    """Registration family for logging backend implementations."""
