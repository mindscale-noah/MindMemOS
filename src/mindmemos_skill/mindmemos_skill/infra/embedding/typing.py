"""Structural interface for injectable asynchronous embedding models."""

from collections.abc import Sequence
from typing import Protocol

from .config import OpenAIEmbeddingConfig


class EmbeddingModel(Protocol):
    """Embed ordered public text without exposing transport details."""

    config: OpenAIEmbeddingConfig
    """Recorded provider and vector settings for this model."""

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one vector per input text in the same order.

        Args:
            texts: Complete public strings to embed; empty input performs no I/O.
        """
        ...
