"""TF-IDF adapter using the shared mixed-language tokenizer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from sklearn.feature_extraction.text import TfidfVectorizer

from mindmemos_skill.algos.trace2skill.offline_evolution.impl.embedding.tokenization import (
    content_token_set,
    jieba_available,
    mixed_text_tokenize,
)


@dataclass(frozen=True)
class TextVectorizerSpec:
    method: str = "tfidf"
    tokenizer: str = "mixed-ascii-jieba-word"
    jieba_available: bool = jieba_available()
    cjk_char_ngrams: tuple[int, ...] = ()
    preserves_ascii_compounds: bool = True

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "tokenizer": self.tokenizer,
            "jieba_available": self.jieba_available,
            "cjk_char_ngrams": list(self.cjk_char_ngrams),
            "preserves_ascii_compounds": self.preserves_ascii_compounds,
        }


def build_tfidf_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        tokenizer=mixed_text_tokenize,
        token_pattern=None,
        lowercase=False,
    )


class TfidfTextVectorizer:
    method = "tfidf"

    def __init__(self) -> None:
        self._vectorizer: TfidfVectorizer | None = None

    def feature_source(self) -> dict:
        return TextVectorizerSpec().to_dict()

    def content_token_set(self, text: str) -> set[str]:
        return content_token_set(text)

    def fit_transform(self, texts: Sequence[str]):
        self._vectorizer = build_tfidf_vectorizer()
        return self._vectorizer.fit_transform(list(texts))

    def transform(self, texts: Sequence[str]):
        if self._vectorizer is None:
            raise RuntimeError("TfidfTextVectorizer.transform called before fit")
        return self._vectorizer.transform(list(texts))

    def encode_field(self, texts: Sequence[str]):
        if not texts or not any(str(text or "").strip() for text in texts):
            return None
        fresh = TfidfTextVectorizer()
        try:
            return fresh.fit_transform(texts)
        except ValueError:
            return None


__all__ = ["TextVectorizerSpec", "TfidfTextVectorizer", "build_tfidf_vectorizer"]
