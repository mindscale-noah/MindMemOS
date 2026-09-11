"""Backend-neutral text vectorization facade."""

from mindmemos_skill.algos.trace2skill.offline_evolution.impl.embedding.adapters import (
    EmbeddingAPITextVectorizer,
    TextVectorizerSpec,
    TfidfTextVectorizer,
    build_tfidf_vectorizer,
)
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.embedding.backend import (
    DENSE_METHODS,
    TEXT_VECTORIZER_METHODS,
    TextVectorizer,
    TextVectorizerSettings,
    build_text_vectorizer,
    hstack_combined,
    resolve_text_vectorizer_config,
)
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.embedding.tokenization import (
    content_token_set,
    mixed_text_tokenize,
)

__all__ = [
    "DENSE_METHODS",
    "EmbeddingAPITextVectorizer",
    "TEXT_VECTORIZER_METHODS",
    "TextVectorizer",
    "TextVectorizerSettings",
    "TextVectorizerSpec",
    "TfidfTextVectorizer",
    "build_text_vectorizer",
    "build_tfidf_vectorizer",
    "content_token_set",
    "hstack_combined",
    "mixed_text_tokenize",
    "resolve_text_vectorizer_config",
]
