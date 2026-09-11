"""Mixed ASCII, code, and CJK tokenization shared by text vectorizers."""

from __future__ import annotations

import logging
import re
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message="pkg_resources is deprecated as an API.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"invalid escape sequence.*",
        category=SyntaxWarning,
        module=r"jieba.*",
    )
    try:
        import jieba as _jieba
    except ImportError as exc:  # pragma: no cover - jieba is a hard dependency.
        raise RuntimeError(
            "jieba is required for mixed Chinese/English tokenization; "
            "it is declared in pyproject.toml -- run `uv sync`."
        ) from exc

_jieba.setLogLevel(logging.WARNING)


ASCII_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+#:/-]*")
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")
TOKEN_SPAN_RE = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9_.+#:/-]*|"
    r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+"
)


def jieba_available() -> bool:
    return _jieba is not None


def _append_unique(out: list[str], token: str) -> None:
    token = token.strip().lower()
    if token and token not in out:
        out.append(token)


def _ascii_tokens(raw: str) -> list[str]:
    out: list[str] = []
    token = raw.lower()
    _append_unique(out, token)
    for part in re.split(r"[^A-Za-z0-9]+", token):
        _append_unique(out, part)
    return out


def _cjk_tokens(raw: str) -> list[str]:
    out: list[str] = []
    for token in _jieba.cut(raw, cut_all=False):
        if token.strip():
            _append_unique(out, token)
    return out


def mixed_text_tokenize(text: str) -> list[str]:
    if not isinstance(text, str) or not text:
        return []
    out: list[str] = []
    for match in TOKEN_SPAN_RE.finditer(text):
        raw = match.group(0)
        if ASCII_TOKEN_RE.fullmatch(raw):
            tokens = _ascii_tokens(raw)
        elif CJK_RE.fullmatch(raw):
            tokens = _cjk_tokens(raw)
        else:
            tokens = []
        for token in tokens:
            _append_unique(out, token)
    return out


def content_token_set(
    text: str,
    *,
    stopwords: set[str] | None = None,
) -> set[str]:
    """Tokens of ``text`` minus single-char noise and any given stopwords.

    No built-in stopword list; callers pass a language-specific one explicitly.
    """
    stops = stopwords or frozenset()
    return {token for token in mixed_text_tokenize(text) if len(token) > 1 and token not in stops}


__all__ = ["content_token_set", "jieba_available", "mixed_text_tokenize"]
