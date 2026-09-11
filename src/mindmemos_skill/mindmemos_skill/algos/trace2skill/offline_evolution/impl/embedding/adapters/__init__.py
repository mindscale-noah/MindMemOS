"""Concrete text-vectorizer adapters."""

from mindmemos_skill.algos.trace2skill.offline_evolution.impl.embedding.adapters.embedding_api import (
    EmbeddingAPITextVectorizer,
)
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.embedding.adapters.tfidf import (
    TextVectorizerSpec,
    TfidfTextVectorizer,
    build_tfidf_vectorizer,
)

__all__ = [
    "EmbeddingAPITextVectorizer",
    "TextVectorizerSpec",
    "TfidfTextVectorizer",
    "build_tfidf_vectorizer",
]
