"""Agent invocation implementations."""

from ..registry.base import ConfigT, Registrable


class Agent(Registrable[ConfigT]):
    """Registration family for agent invocation implementations."""
