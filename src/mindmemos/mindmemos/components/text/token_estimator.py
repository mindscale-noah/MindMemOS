"""Small deterministic token estimator used by memory chunking."""

from __future__ import annotations

TOKEN_ESTIMATOR_VERSION = "heuristic-v1"


def estimate_tokens(text: str) -> int:
    """Estimate CJK and Latin token cost without a model-specific dependency."""

    if not text or not text.strip():
        return 0
    cjk = sum(1 for char in text if _is_cjk(char))
    non_cjk = "".join(" " if _is_cjk(char) else char for char in text)
    latin_words = len(non_cjk.split())
    return max(1, int(cjk / 1.5) + latin_words)


def _is_cjk(char: str) -> bool:
    return "一" <= char <= "鿿" or "㐀" <= char <= "䶿"
