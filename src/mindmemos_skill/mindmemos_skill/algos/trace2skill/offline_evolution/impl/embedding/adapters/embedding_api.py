"""Remote embedding adapter for an OpenAI-compatible /embeddings endpoint.

Produces the same L2-normalized float32 matrices as the local adapter, so
either can be selected per run. Requests are batched and retried.
"""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np

# Waits between retry attempts, in seconds; the list length is the budget.
_RETRY_BACKOFF_SEC = (1.0, 2.0, 4.0)


def _normalize(matrix: np.ndarray) -> np.ndarray:
    """Scale each row to unit length, leaving all-zero rows alone.

    Applied even when the endpoint normalizes, so a non-normalizing endpoint
    still yields comparable cosine distances.
    """
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class EmbeddingAPITextVectorizer:
    """Encode text through a remote embeddings endpoint."""

    method = "embedding_api"

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        batch_size: int = 64,
        timeout: float = 120.0,
        dimensions: int | None = None,
    ) -> None:
        if not str(model).strip():
            raise ValueError(
                "embedding_api vectorizer requires a model name; set "
                "vectorizer.embedding_api_model in configs/default.yaml"
            )
        if batch_size < 1:
            raise ValueError(f"embedding batch size must be >= 1, got {batch_size!r}")
        self.model = str(model).strip()
        self.base_url = base_url
        self.batch_size = int(batch_size)
        self.timeout = float(timeout)
        self.dimensions = int(dimensions) if dimensions else None
        self._api_key = api_key
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "embedding_api text vectorizer requires the openai SDK installed in the active Python environment"
                ) from exc
            kwargs: dict = {"timeout": self.timeout}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input, retrying transport failures.

        Response items are re-sorted by their ``index`` field; endpoint
        order is not trusted.
        """
        kwargs: dict = {"model": self.model, "input": texts}
        if self.dimensions:
            kwargs["dimensions"] = self.dimensions
        last: Exception | None = None
        for attempt in range(len(_RETRY_BACKOFF_SEC) + 1):
            try:
                response = self._ensure_client().embeddings.create(**kwargs)
            except Exception as exc:
                last = exc
                if attempt < len(_RETRY_BACKOFF_SEC):
                    time.sleep(_RETRY_BACKOFF_SEC[attempt])
                    continue
                raise RuntimeError(
                    f"embedding endpoint failed after {attempt + 1} attempts "
                    f"on a batch of {len(texts)}: {type(exc).__name__}: {exc}"
                ) from last
            items = sorted(response.data, key=lambda item: item.index)
            if len(items) != len(texts):
                raise RuntimeError(f"embedding endpoint returned {len(items)} vectors for {len(texts)} inputs")
            return [list(item.embedding) for item in items]
        raise AssertionError("unreachable")

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        cleaned = [str(text) for text in texts]
        if not cleaned:
            return np.zeros((0, 0), dtype=np.float32)
        vectors: list[list[float]] = []
        for start in range(0, len(cleaned), self.batch_size):
            vectors.extend(self._embed_batch(cleaned[start : start + self.batch_size]))
        widths = {len(vector) for vector in vectors}
        if len(widths) > 1:
            raise RuntimeError(f"embedding endpoint returned mixed dimensions: {sorted(widths)}")
        return _normalize(np.asarray(vectors, dtype=np.float32))

    def fit_transform(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts)

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_field(self, texts: Sequence[str]) -> np.ndarray | None:
        return self._encode(texts)

    def feature_source(self) -> dict:
        """Describe the encoder for the provenance record; never includes the key."""
        source = {
            "method": self.method,
            "embedding_model": self.model,
            "embedding_batch_size": self.batch_size,
            "embedding_base_url": self.base_url,
        }
        if self.dimensions:
            source["embedding_dimensions"] = self.dimensions
        return source

    def content_token_set(self, text: str) -> set[str]:
        raise NotImplementedError(
            "EmbeddingAPITextVectorizer has no token-level coverage; call content_token_set() on the raw text instead"
        )


__all__ = ["EmbeddingAPITextVectorizer"]
