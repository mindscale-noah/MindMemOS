"""Agent invocation implementations."""

from typing import Self

from ..registry.base import ConfigT, Registrable
from .config import AgentConnectionConfig


class Agent(Registrable[ConfigT]):
    """Registration family for agent invocation implementations."""

    @classmethod
    def from_config(cls, config: AgentConnectionConfig) -> Self:
        """Construct the registered Agent selected by validated configuration.

        Args:
            config: Backend identity and implementation-specific validated settings.

        Returns:
            A new Agent instance; instances are never globally cached.

        Raises:
            ValueError: The backend identity is absent or is not registered.
            ValidationError: Settings do not match the selected backend.
        """
        if config.class_name is None:
            raise ValueError("Agent construction requires class_name in its configuration")
        return cls.build(config.class_name, config)
