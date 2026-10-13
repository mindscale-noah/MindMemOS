"""Registered asynchronous embedding implementations."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Self

from ...registry.base import ConfigT, Registrable
from .config import EmbeddingConfig


class EmbeddingModel(Registrable[ConfigT], ABC):
    """Create configured embedding backends and embed ordered public text."""

    @classmethod
    def from_config(cls, config: EmbeddingConfig) -> Self:
        """Construct the registered embedding backend selected by configuration.

        Args:
            config: Validated backend identity and implementation-specific settings.

        Returns:
            A new embedding model without making provider requests.

        Raises:
            ValueError: The selected backend is not registered.
            ValidationError: Settings do not match the selected backend.
        """
        return cls.build(config.class_name, config)

    @abstractmethod
    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed public text in input order.

        Args:
            texts: Public strings; an empty sequence performs no provider I/O.

        Returns:
            One vector per input string in the same order.
        """
        raise NotImplementedError
