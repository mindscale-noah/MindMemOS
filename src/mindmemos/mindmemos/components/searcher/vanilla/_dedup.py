"""Near-duplicate folding for search candidates.

The add pipeline can persist the same fact many times with slightly different
wording (e.g. one activity restated across conversation turns), and rerank
scores those near-identical copies almost equally. Without folding, a top-k of
distinct-looking slots is really filled with a handful of repeated facts,
starving the answer of independent evidence. This collapses near-duplicates on
the recall pool *before* rerank/truncation so the k slots go to k distinct
memories.

Similarity uses token-set (Jaccard) overlap rather than character-level
sequence matching. Restatements of the same fact reorder words and swap
synonyms, which character-level ratios (``difflib``) barely register, while
Jaccard on content words cleanly separates same-fact restatements (high
overlap) from merely related but distinct memories (low overlap).
"""

from __future__ import annotations

import re
from collections.abc import Hashable, Sequence

from ....logging import get_logger
from ....typing import MemorySearchItem

logger = get_logger(__name__)

_CJK_TOKEN = re.compile(r"[㐀-䶿一-鿿]")
_WORD_OR_CJK_TOKEN = re.compile(r"[0-9a-z]{2,}|[㐀-䶿一-鿿]")
_MIN_TOKENS_FOR_NEAR_DEDUP = 5
_MIN_CJK_CHARACTERS_FOR_NEAR_DEDUP = 10
_MAX_TOKENS_FOR_NEAR_DEDUP = 512


def _tokens(text: str) -> frozenset[str]:
    """Extract a bounded word/CJK token fingerprint for overlap."""
    lowered = text.lower()
    tokens: set[str] = set()
    for match in _WORD_OR_CJK_TOKEN.finditer(lowered):
        tokens.add(match.group())
        if len(tokens) >= _MAX_TOKENS_FOR_NEAR_DEDUP:
            break
    return frozenset(tokens)


def _is_short_for_near_dedup(text: str, tokens: frozenset[str]) -> bool:
    """Keep short facts intact even when generic terms overlap."""
    cjk_characters = _CJK_TOKEN.findall(text)
    if cjk_characters:
        return len(cjk_characters) < _MIN_CJK_CHARACTERS_FOR_NEAR_DEDUP
    return len(tokens) < _MIN_TOKENS_FOR_NEAR_DEDUP


def _normalized_text(text: str) -> str:
    """Normalize only case and whitespace for exact duplicate detection."""
    return " ".join(text.casefold().split())


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """Token-set Jaccard similarity; 0.0 when both sides are empty."""
    if not a and not b:
        return 0.0
    intersection_size = len(a & b)
    union_size = len(a) + len(b) - intersection_size
    if not union_size:
        return 0.0
    return intersection_size / union_size


def dedup_by_text_similarity(
    candidates: list[MemorySearchItem],
    *,
    threshold: float = 0.6,
    group_keys: Sequence[Hashable] | None = None,
) -> list[MemorySearchItem]:
    """Greedily fold near-duplicate memories within each group, preserving order.

    A candidate is kept when its token set is less than ``threshold`` Jaccard
    similar to every already-kept candidate in the same group; otherwise it is
    dropped as a near-duplicate of the earlier one. When ``group_keys`` is
    omitted, all candidates belong to one group, preserving the legacy behavior.
    Recall order is preserved so a later rerank still decides the final ranking.

    ``threshold`` of 1.0 folds long memories with identical bounded token
    fingerprints; lower values fold looser paraphrases. Exact duplicate texts
    always fold. Near-duplicate folding is skipped for short memories to avoid
    dropping facts that differ in a single salient token.
    """
    if group_keys is None:
        resolved_group_keys: Sequence[Hashable] = [None] * len(candidates)
    elif len(group_keys) != len(candidates):
        raise ValueError("group_keys must contain one entry per candidate")
    else:
        resolved_group_keys = group_keys

    if len(candidates) <= 1:
        return list(candidates)

    kept: list[MemorySearchItem] = []
    kept_tokens: list[frozenset[str]] = []
    kept_is_short: list[bool] = []
    kept_group_keys: list[Hashable] = []
    kept_texts: set[tuple[Hashable, str]] = set()
    for candidate, group_key in zip(candidates, resolved_group_keys, strict=True):
        normalized_text = _normalized_text(candidate.memory)
        grouped_text = (group_key, normalized_text)
        if grouped_text in kept_texts:
            continue

        tokens = _tokens(candidate.memory)
        is_short = _is_short_for_near_dedup(candidate.memory, tokens)
        if not is_short and any(
            previous_group_key == group_key
            and not previous_is_short
            and _jaccard(tokens, previous_tokens) >= threshold
            for previous_tokens, previous_is_short, previous_group_key in zip(
                kept_tokens,
                kept_is_short,
                kept_group_keys,
                strict=True,
            )
        ):
            continue
        kept.append(candidate)
        kept_tokens.append(tokens)
        kept_is_short.append(is_short)
        kept_group_keys.append(group_key)
        kept_texts.add(grouped_text)

    dropped = len(candidates) - len(kept)
    if dropped:
        logger.debug("search_dedup_folded", folded=dropped, kept=len(kept), total=len(candidates))
    return kept
