"""Bounded batches of validated OpenAI-compatible embeddings."""

from collections.abc import Sequence
from typing import TYPE_CHECKING

from ..database.vector import Vector
from .config import OpenAIEmbeddingConfig

if TYPE_CHECKING:
    from openai import OpenAI


class OpenAIEmbeddingClient:
    """Construct transports lazily and preserve the ownership of injected clients."""

    def __init__(self, config: OpenAIEmbeddingConfig, *, client: "OpenAI | None" = None):
        """Retain settings without opening a connection.

        Args:
            config: Explicit model, endpoint, vector shape, and batching settings.
            client: Borrowed SDK client for testing or caller-managed transport.
        """
        self.config = config.model_copy(deep=True)
        self._client = client

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
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
        if self._client is not None:
            return self._embed(self._client, texts)
        from openai import OpenAI

        with OpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key.get_secret_value() if self.config.api_key else "unused",
            timeout=self.config.timeout_s,
            max_retries=self.config.max_retries,
        ) as client:
            return self._embed(client, texts)

    def _embed(self, client: "OpenAI", texts: Sequence[str]) -> list[list[float]]:
        """Validate each response before appending its ordered vectors."""
        vectors = []
        validator = Vector(self.config.dimensions)
        for start in range(0, len(texts), self.config.batch_size):
            batch = list(texts[start : start + self.config.batch_size])
            options = {"dimensions": self.config.dimensions} if self.config.send_dimensions else {}
            try:
                response = client.embeddings.create(
                    model=self.config.model, input=batch, encoding_format="float", **options
                )
            except Exception as exc:
                raise RuntimeError(f"Embedding request failed: {type(exc).__name__}") from None
            entries = sorted(response.data, key=lambda entry: entry.index)
            if [entry.index for entry in entries] != list(range(len(batch))):
                raise ValueError("Embedding response indices must match the input batch")
            vectors.extend(validator.validate(entry.embedding) for entry in entries)
        return vectors
