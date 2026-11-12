"""Concurrent batches of validated AsyncOpenAI-compatible embeddings."""

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from typing import TYPE_CHECKING

from ..database.vector import Vector
from ..model_access.openai import open_embedding_batches
from .base import EmbeddingModel
from .config import OpenAIEmbeddingConfig

if TYPE_CHECKING:
    from openai import AsyncOpenAI
    from openai.types import CreateEmbeddingResponse


@EmbeddingModel.register("openai_embedding")
class OpenAIEmbeddingClient(EmbeddingModel[OpenAIEmbeddingConfig]):
    """Construct transports lazily and preserve the ownership of injected clients."""

    config_type = OpenAIEmbeddingConfig
    """Configuration model validated by the shared embedding registry."""

    def __init__(self, config: OpenAIEmbeddingConfig, *, client: "AsyncOpenAI | None" = None):
        """Retain settings without opening a connection.

        Args:
            config: Explicit model, endpoint, vector shape, and batching settings.
            client: Borrowed SDK client for testing or caller-managed transport.
        """
        super().__init__(config.model_copy(deep=True))
        self._client = client

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed ordered batches and validate count, indices, and vector contents.

        Args:
            texts: Public text in output order; no request is sent for empty input.

        Returns:
            Finite nonzero float32 vectors with the configured dimension.

        Raises:
            ValueError: The provider returns invalid indices, count, or vectors.
            RuntimeError: The provider request fails after configured retries.
        """
        if not texts:
            return []
        if any(not isinstance(text, str) for text in texts):
            raise TypeError("Embedding inputs must be strings")
        async with open_embedding_batches(self.config, client=self._client) as send:
            return await self._embed(send, texts)

    async def _embed(
        self,
        send: Callable[[list[str]], Awaitable["CreateEmbeddingResponse"]],
        texts: Sequence[str],
    ) -> list[list[float]]:
        """Run all batches concurrently and flatten validated vectors in input order."""
        validator = Vector(self.config.dimensions)

        async def embed_batch(batch: list[str]) -> list[list[float]]:
            """Request and validate the vectors for one input batch."""
            response = await send(batch)
            entries = sorted(response.data, key=lambda entry: entry.index)
            if [entry.index for entry in entries] != list(range(len(batch))):
                raise ValueError("Embedding response indices must match the input batch")
            return [validator.validate(entry.embedding) for entry in entries]

        tasks = [
            asyncio.create_task(embed_batch(list(texts[start : start + self.config.batch_size])))
            for start in range(0, len(texts), self.config.batch_size)
        ]
        try:
            batches = await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        return [vector for batch in batches for vector in batch]
