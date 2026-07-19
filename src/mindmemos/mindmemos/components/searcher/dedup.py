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

from ...logging import get_logger
from ...typing import MemorySearchItem

logger = get_logger(__name__)

_WORD_TOKEN = re.compile(r"[0-9a-z]{2,}")
_CJK_TOKEN = re.compile(r"[㐀-䶿一-鿿]")
_MIN_TOKENS_FOR_NEAR_DEDUP = 5
_MIN_CJK_CHARACTERS_FOR_NEAR_DEDUP = 10


def _tokens(text: str) -> frozenset[str]:
    """Extract word tokens plus individual CJK characters for overlap."""
    lowered = text.lower()
    return frozenset(_WORD_TOKEN.findall(lowered)) | frozenset(_CJK_TOKEN.findall(lowered))


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
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def dedup_by_text_similarity(
    candidates: list[MemorySearchItem],
    *,
    threshold: float = 0.6,
) -> list[MemorySearchItem]:
    """Greedily fold near-duplicate memories, preserving input order.

    A candidate is kept when its token set is less than ``threshold`` Jaccard
    similar to every already-kept candidate; otherwise it is dropped as a
    near-duplicate of the earlier one. Recall order is preserved so a later
    rerank still decides the final ranking.

    ``threshold`` of 1.0 folds only token-identical long memories; lower values
    fold looser paraphrases. Exact duplicate texts always fold. Near-duplicate
    folding is skipped for short memories to avoid dropping facts that differ in
    a single salient token.
    """
    if len(candidates) <= 1:
        return list(candidates)

    kept: list[MemorySearchItem] = []
    kept_tokens: list[frozenset[str]] = []
    kept_texts: set[str] = set()
    for candidate in candidates:
        normalized_text = _normalized_text(candidate.memory)
        if normalized_text in kept_texts:
            continue

        tokens = _tokens(candidate.memory)
        is_short = _is_short_for_near_dedup(candidate.memory, tokens)
        if not is_short and any(
            not _is_short_for_near_dedup(previous_candidate.memory, previous_tokens)
            and _jaccard(tokens, previous_tokens) >= threshold
            for previous_candidate, previous_tokens in zip(kept, kept_tokens, strict=True)
        ):
            continue
        kept.append(candidate)
        kept_tokens.append(tokens)
        kept_texts.add(normalized_text)

    dropped = len(candidates) - len(kept)
    if dropped:
        logger.debug("search_dedup_folded", folded=dropped, kept=len(kept), total=len(candidates))
    return kept
