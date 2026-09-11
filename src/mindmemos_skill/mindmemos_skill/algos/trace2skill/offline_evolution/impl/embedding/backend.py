"""Facade selecting TF-IDF, local SentenceTransformer, or remote embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np
from scipy.sparse import hstack

from mindmemos_skill.algos.trace2skill.offline_evolution.impl.config import (
    setting_int,
    setting_optional_str,
    setting_raw,
)
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.embedding.adapters import (
    EmbeddingAPITextVectorizer,
    TfidfTextVectorizer,
)


class TextVectorizer(Protocol):
    """Backend-neutral interface implemented by text vectorizer adapters."""

    method: str

    def fit_transform(self, texts: Sequence[str]): ...
    def transform(self, texts: Sequence[str]): ...
    def encode_field(self, texts: Sequence[str]): ...
    def feature_source(self) -> dict: ...
    def content_token_set(self, text: str) -> set[str]: ...


TEXT_VECTORIZER_METHODS = ("tfidf", "embedding_api")

# Methods that encode by meaning rather than token overlap; "is this dense?"
# checks ask this rather than naming one of them.
DENSE_METHODS = ("embedding_api",)

_TEXT_VECTORIZER_KEYS = {
    "method": "vectorizer.method",
    "embedding_model": "vectorizer.embedding_model",
    "embedding_batch_size": "vectorizer.embedding_batch_size",
    "embedding_device": "vectorizer.embedding_device",
}

# Endpoint credentials live in .env; the model name is run configuration,
# not a secret, so it stays a config key.
_EMBEDDING_API_KEYS = {
    "base_url": "EMBEDDING_BASE_URL",
    "api_key": "EMBEDDING_API_KEY",
    "model": "vectorizer.embedding_api_model",
}


@dataclass(frozen=True)
class TextVectorizerSettings:
    """One fully-resolved vectorizer choice."""

    method: str
    embedding_model: str | None
    embedding_batch_size: int
    embedding_device: str
    embedding_base_url: str | None = None
    embedding_api_key: str | None = None


def hstack_combined(matrices: list) -> object | None:
    if not matrices:
        return None
    if all(hasattr(matrix, "toarray") for matrix in matrices):
        return hstack(matrices)
    return np.hstack(matrices)


def resolve_text_vectorizer_config(
    *,
    method: str | None = None,
    embedding_model: str | None = None,
    embedding_batch_size: int | None = None,
    embedding_device: str | None = None,
) -> TextVectorizerSettings:
    """Resolve each field from the explicit argument, else from settings.

    No fallbacks here: an unset key raises. ``embedding_api`` credentials
    come from ``.env`` only and have no explicit-argument form.
    """
    resolved_method = (method or setting_raw(_TEXT_VECTORIZER_KEYS["method"])).strip().lower()
    if resolved_method not in TEXT_VECTORIZER_METHODS:
        raise ValueError(f"text vectorizer method must be one of {TEXT_VECTORIZER_METHODS}, got {resolved_method!r}")

    resolved_batch = (
        int(embedding_batch_size)
        if embedding_batch_size is not None
        else setting_int(_TEXT_VECTORIZER_KEYS["embedding_batch_size"])
    )
    if resolved_batch < 1:
        raise ValueError(f"vectorizer.embedding_batch_size must be >= 1, got {resolved_batch!r}")

    resolved_device = embedding_device or setting_raw(_TEXT_VECTORIZER_KEYS["embedding_device"])

    if resolved_method == "embedding_api":
        model = setting_optional_str(_EMBEDDING_API_KEYS["model"])
        if not model:
            raise ValueError(f"{_EMBEDDING_API_KEYS['model']} is required when vectorizer.method=embedding_api")
        return TextVectorizerSettings(
            method=resolved_method,
            embedding_model=model,
            embedding_batch_size=resolved_batch,
            embedding_device=resolved_device,
            embedding_base_url=setting_optional_str(_EMBEDDING_API_KEYS["base_url"]),
            embedding_api_key=setting_optional_str(_EMBEDDING_API_KEYS["api_key"]),
        )

    resolved_model = embedding_model or setting_optional_str(_TEXT_VECTORIZER_KEYS["embedding_model"])
    if resolved_method == "embedding" and not resolved_model:
        raise ValueError(f"vectorizer.embedding_model is required when vectorizer.method={resolved_method!r}")
    return TextVectorizerSettings(
        method=resolved_method,
        embedding_model=resolved_model,
        embedding_batch_size=resolved_batch,
        embedding_device=resolved_device,
    )


def build_text_vectorizer(
    *,
    method: str | None = None,
    embedding_model: str | None = None,
    embedding_batch_size: int | None = None,
    embedding_device: str | None = None,
) -> TextVectorizer:
    """Build the adapter named by ``method``, resolving what was not passed.

    ``embedding_api`` always re-resolves: its endpoint and key have no
    argument form.
    """
    normalized = (method or "").strip().lower()
    needs_resolution = (
        not normalized
        or normalized == "embedding_api"
        or any(value is None for value in (embedding_model, embedding_batch_size, embedding_device))
    )
    settings = None
    if needs_resolution:
        settings = resolve_text_vectorizer_config(
            method=method,
            embedding_model=embedding_model,
            embedding_batch_size=embedding_batch_size,
            embedding_device=embedding_device,
        )
        normalized = settings.method
        embedding_model = settings.embedding_model
        embedding_batch_size = settings.embedding_batch_size
        embedding_device = settings.embedding_device

    if normalized not in TEXT_VECTORIZER_METHODS:
        raise ValueError(f"text vectorizer method must be one of {TEXT_VECTORIZER_METHODS}, got {method!r}")
    if normalized == "tfidf":
        return TfidfTextVectorizer()
    resolved_batch = int(embedding_batch_size)
    if resolved_batch < 1:
        raise ValueError(f"embedding batch size must be >= 1, got {resolved_batch!r}")
    if normalized == "embedding_api":
        return EmbeddingAPITextVectorizer(
            embedding_model,
            base_url=settings.embedding_base_url,
            api_key=settings.embedding_api_key,
            batch_size=resolved_batch,
        )
    if normalized == "embedding":
        raise ValueError(
            "vectorizer.method=embedding (local SentenceTransformer) was removed; use tfidf or embedding_api"
        )
    raise ValueError(f"text vectorizer method must be one of {TEXT_VECTORIZER_METHODS}, got {method!r}")


__all__ = [
    "DENSE_METHODS",
    "TEXT_VECTORIZER_METHODS",
    "TextVectorizer",
    "TextVectorizerSettings",
    "build_text_vectorizer",
    "hstack_combined",
    "resolve_text_vectorizer_config",
]
