"""Token estimator private to the search retention path.

Retention budgets are a caller-facing contract (``token_budget`` in the
search request), so unbroken runs (URLs, base64, hashes) must scale with
length instead of costing one flat token. The shared estimator in
``components.text`` keeps its word-count semantics for the add pipeline's
chunking and compaction budgets; do not route those callers here.
"""

from __future__ import annotations

import math

RETENTION_TOKEN_ESTIMATOR_VERSION = "heuristic-v2"

# Non-CJK words cost at least one token; long unbroken runs scale linearly
# at this many characters per token so a single word cannot bypass token
# budgets by omitting spaces. Normal prose words are shorter than this cap
# and keep costing one token each.
_LATIN_CHARS_PER_TOKEN = 8


def estimate_tokens(text: str) -> int:
    """Estimate CJK and Latin token cost without a model-specific dependency."""

    if not text or not text.strip():
        return 0
    cjk = sum(1 for char in text if _is_cjk(char))
    non_cjk = "".join(" " if _is_cjk(char) else char for char in text)
    latin_words = sum(max(1, math.ceil(len(word) / _LATIN_CHARS_PER_TOKEN)) for word in non_cjk.split())
    return max(1, int(cjk / 1.5) + latin_words)


def _is_cjk(char: str) -> bool:
    return "一" <= char <= "鿿" or "㐀" <= char <= "䶿"
