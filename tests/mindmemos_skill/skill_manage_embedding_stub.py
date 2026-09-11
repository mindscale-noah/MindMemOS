"""Deterministic embedding stubs shared by ``skill_manage`` unit and integration tests.

``ContentHashEmbedder`` mirrors the GOS golden generator's content-hash stub:
feature hashing over word tokens and character trigrams with signed buckets,
L2-normalized. Similar text yields similar vectors, so KNN over these stubs is
meaningful enough to assert seed order without a real model.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from mindmemos_skill.errors import EmbeddingDimensionError


@dataclass
class StubEmbeddingResponse:
    embeddings: list[list[float]]
    model: str = "stub"


@dataclass
class ContentHashEmbedder:
    """Stand-in for ``EmbedClient`` exposing the same ``embed`` signature."""

    dimension: int = 3072
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def embed(
        self,
        task: str,
        text: str | list[str],
        *,
        model: str | None = None,
        expected_dim: int | None = None,
        **kwargs: Any,
    ) -> StubEmbeddingResponse:
        texts = [text] if isinstance(text, str) else list(text)
        self.calls.append({"task": task, "count": len(texts), "model": model, "expected_dim": expected_dim})
        vectors = [self.embed_one(item) for item in texts]
        if expected_dim is not None and expected_dim != self.dimension:
            raise EmbeddingDimensionError(
                expected=expected_dim, actual=self.dimension, model=model or "stub", task=task
            )
        return StubEmbeddingResponse(embeddings=vectors, model=model or "stub")

    def embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        lowered = text.lower()
        grams = re.findall(r"[a-z0-9_]+", lowered)
        compact = re.sub(r"\s+", " ", lowered)
        grams.extend(compact[index : index + 3] for index in range(max(len(compact) - 2, 0)))
        for gram in grams:
            digest = int(hashlib.md5(gram.encode("utf-8")).hexdigest()[:8], 16)
            vector[digest % self.dimension] += 1.0 if (digest >> 8) % 2 == 0 else -1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm > 0:
            vector = [value / norm for value in vector]
        return vector


def stub_algorithm_context(dimension: int = 3072) -> SimpleNamespace:
    """Minimal ``AlgorithmBuildContext`` stand-in carrying one stub embedding model."""

    embedder = ContentHashEmbedder(dimension=dimension)
    return SimpleNamespace(
        models={"embedding": embedder},
        chat_models={},
        embedding_models={"embedding": embedder},
        agents={},
        execution=None,
        config_hash="stub",
    )


__all__ = ["ContentHashEmbedder", "StubEmbeddingResponse", "stub_algorithm_context"]
