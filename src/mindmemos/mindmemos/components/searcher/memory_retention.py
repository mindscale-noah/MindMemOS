"""Deterministic token-budget selection for ranked memory candidates."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from math import exp, isfinite, log
from typing import Callable

from ...config.algo.search import MemoryRetentionConfig
from ..text import TextPreprocessor, estimate_tokens, get_text_preprocessor
from .scored_candidate import ScoredSearchCandidate, merge_scored_candidates


@dataclass(frozen=True, slots=True)
class CandidateRetentionScore:
    """Feature values used to rank one candidate for a request budget."""

    candidate: ScoredSearchCandidate
    estimated_tokens: int
    query_overlap: float
    recency: float
    cost_ratio: float
    priority: float


@dataclass(frozen=True, slots=True)
class RetentionSelection:
    """Strict-budget selection and aggregate observability values."""

    candidates: list[ScoredSearchCandidate]
    estimated_tokens_before: int
    estimated_tokens_after: int
    budget_induced_empty: bool


class MemoryRetentionSelector:
    """Select useful memories without exceeding the caller's token budget."""

    def __init__(
        self,
        *,
        config: MemoryRetentionConfig,
        text_preprocessor: TextPreprocessor | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._config = config
        self._text_preprocessor = text_preprocessor or get_text_preprocessor()
        self._now = now or (lambda: datetime.now(UTC))

    def score(
        self,
        *,
        query: str,
        candidates: list[ScoredSearchCandidate],
        token_budget: int,
    ) -> list[CandidateRetentionScore]:
        """Compute the documented mixed score once per bounded candidate."""

        if token_budget < 1:
            raise ValueError("token_budget must be positive")
        query_terms = self._terms_for_query(query)
        now = self._now()
        if now.tzinfo is None:
            now = now.replace(tzinfo=UTC)
        bounded = merge_scored_candidates(candidates[: self._config.max_candidates])
        scored: list[CandidateRetentionScore] = []
        for candidate in bounded:
            cost = estimate_tokens(candidate.memory)
            overlap = self._query_overlap(query_terms, candidate.memory)
            recency = self._recency(candidate, now)
            cost_ratio = min(cost / token_budget, 1.0)
            relevance = candidate.relevance_score if isfinite(candidate.relevance_score) else 0.0
            relevance = min(max(relevance, 0.0), 1.0)
            priority = (
                self._config.relevance_weight * relevance
                + self._config.query_overlap_weight * overlap
                + self._config.recency_weight * recency
                - self._config.cost_weight * cost_ratio
            )
            scored.append(
                CandidateRetentionScore(
                    candidate=candidate,
                    estimated_tokens=cost,
                    query_overlap=overlap,
                    recency=recency,
                    cost_ratio=cost_ratio,
                    priority=priority,
                )
            )
        return scored

    def select(
        self,
        *,
        query: str,
        candidates: list[ScoredSearchCandidate],
        token_budget: int,
    ) -> RetentionSelection:
        """Select by configured selector version under a strict token budget."""

        if self._config.selector_version == "mixed-v2":
            return self._select_mixed_v2(query=query, candidates=candidates, token_budget=token_budget)
        return self._select_mixed_v1(query=query, candidates=candidates, token_budget=token_budget)

    def _select_mixed_v1(
        self,
        *,
        query: str,
        candidates: list[ScoredSearchCandidate],
        token_budget: int,
    ) -> RetentionSelection:
        """Greedily select by priority, skipping entries that do not fit."""

        scored = self.score(query=query, candidates=candidates, token_budget=token_budget)
        estimated_before = sum(entry.estimated_tokens for entry in scored)
        selected: list[CandidateRetentionScore] = []
        used = 0
        for entry in sorted(
            scored,
            key=lambda value: (-value.priority, value.candidate.rank, value.candidate.id),
        ):
            if used + entry.estimated_tokens > token_budget:
                continue
            selected.append(entry)
            used += entry.estimated_tokens

        selected.sort(key=lambda value: (value.candidate.rank, value.candidate.id))
        return RetentionSelection(
            candidates=[entry.candidate for entry in selected],
            estimated_tokens_before=estimated_before,
            estimated_tokens_after=used,
            budget_induced_empty=bool(scored) and not selected,
        )

    def _select_mixed_v2(
        self,
        *,
        query: str,
        candidates: list[ScoredSearchCandidate],
        token_budget: int,
    ) -> RetentionSelection:
        """Phase-1 relevance guarantee, then MMR packing by priority vs redundancy."""

        scored = self.score(query=query, candidates=candidates, token_budget=token_budget)
        estimated_before = sum(entry.estimated_tokens for entry in scored)
        if not scored:
            return RetentionSelection(
                candidates=[],
                estimated_tokens_before=0,
                estimated_tokens_after=0,
                budget_induced_empty=False,
            )

        selected: list[CandidateRetentionScore] = []
        selected_ids: set[str] = set()
        used = 0
        term_cache = {entry.candidate.id: self._terms_for_text(entry.candidate.memory) for entry in scored}

        # Phase 1: force-keep up to top_m_guarantee by relevance_score (fit budget).
        top_m = max(0, int(self._config.top_m_guarantee))
        by_relevance = sorted(
            scored,
            key=lambda value: (
                -_relevance(value.candidate),
                value.candidate.rank,
                value.candidate.id,
            ),
        )
        for entry in by_relevance:
            if len(selected) >= top_m:
                break
            if used + entry.estimated_tokens > token_budget:
                continue
            selected.append(entry)
            selected_ids.add(entry.candidate.id)
            used += entry.estimated_tokens

        # Phase 2: MMR over remaining candidates.
        lam = min(max(float(self._config.mmr_lambda), 0.0), 1.0)
        remaining = [entry for entry in scored if entry.candidate.id not in selected_ids]
        while remaining:
            best: CandidateRetentionScore | None = None
            best_key: tuple[float, int, str] | None = None
            selected_term_sets = [term_cache[s.candidate.id] for s in selected]
            for entry in remaining:
                if used + entry.estimated_tokens > token_budget:
                    continue
                overlap = _max_jaccard(term_cache[entry.candidate.id], selected_term_sets)
                mmr = lam * entry.priority - (1.0 - lam) * overlap
                key = (mmr, -entry.candidate.rank, entry.candidate.id)
                if best_key is None or key > best_key:
                    best = entry
                    best_key = key
            if best is None:
                break
            selected.append(best)
            selected_ids.add(best.candidate.id)
            used += best.estimated_tokens
            remaining = [entry for entry in remaining if entry.candidate.id not in selected_ids]

        selected.sort(key=lambda value: (value.candidate.rank, value.candidate.id))
        return RetentionSelection(
            candidates=[entry.candidate for entry in selected],
            estimated_tokens_before=estimated_before,
            estimated_tokens_after=used,
            budget_induced_empty=bool(scored) and not selected,
        )

    def _terms_for_query(self, query: str) -> set[str]:
        processed = self._text_preprocessor.preprocess_query(query, include_entities=False)
        return {str(token).casefold() for token in processed.tokens if str(token).strip()}

    def _terms_for_text(self, text: str) -> set[str]:
        processed = self._text_preprocessor.preprocess_text(text or "", include_entities=False)
        return {str(token).casefold() for token in processed.tokens if str(token).strip()}

    def _query_overlap(self, query_terms: set[str], text: str) -> float:
        if not query_terms:
            return 0.0
        memory_terms = self._terms_for_text(text)
        return len(query_terms & memory_terms) / len(query_terms)

    def _recency(self, candidate: ScoredSearchCandidate, now: datetime) -> float:
        timestamp = _first_timestamp(
            candidate.event_time,
            candidate.source_timestamp,
            candidate.last_update_at,
        )
        if timestamp is None:
            return self._config.missing_recency_score
        age_days = max((now - timestamp).total_seconds() / 86400.0, 0.0)
        return exp(-log(2.0) * age_days / self._config.recency_half_life_days)


def _relevance(candidate: ScoredSearchCandidate) -> float:
    score = candidate.relevance_score if isfinite(candidate.relevance_score) else 0.0
    return min(max(score, 0.0), 1.0)


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _max_jaccard(terms: set[str], others: list[set[str]]) -> float:
    if not others:
        return 0.0
    return max(_jaccard(terms, other) for other in others)


def _first_timestamp(*values: str | None) -> datetime | None:
    for value in values:
        if not value:
            continue
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            continue
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
    return None
