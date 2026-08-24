"""Cluster-and-stitch consolidation for scored search candidates."""

from __future__ import annotations

import re
from dataclasses import dataclass
from math import isfinite

from ...typing import MemorySearchItem
from ..text import TextPreprocessor
from .scored_candidate import ScoredSearchCandidate


@dataclass(frozen=True, slots=True)
class ConsolidationResult:
    """Consolidated candidates and simple observability."""

    candidates: list[ScoredSearchCandidate]
    input_count: int
    cluster_count: int
    output_count: int


class MemoryConsolidator:
    """Merge near-duplicate / complementary memories before token-budget retention."""

    def __init__(
        self,
        *,
        text_preprocessor: TextPreprocessor,
        max_memories: int = 40,
        cluster_threshold: float = 0.50,
        near_dup_threshold: float = 0.85,
        stitch_max_members: int = 3,
        max_chars: int = 600,
    ) -> None:
        self._text_preprocessor = text_preprocessor
        self._max_memories = max(1, int(max_memories))
        self._cluster_threshold = float(cluster_threshold)
        self._near_dup_threshold = float(near_dup_threshold)
        self._stitch_max_members = max(1, int(stitch_max_members))
        self._max_chars = max(1, int(max_chars))

    def consolidate(self, candidates: list[ScoredSearchCandidate]) -> ConsolidationResult:
        """Cluster by Jaccard, cap clusters, then stitch each cluster to one memory."""

        if not candidates:
            return ConsolidationResult(candidates=[], input_count=0, cluster_count=0, output_count=0)

        ordered = sorted(candidates, key=lambda c: (-_relevance(c), c.rank, c.id))
        term_sets = [self._terms(c.memory) for c in ordered]
        clusters: list[list[int]] = []
        for idx, terms in enumerate(term_sets):
            placed = False
            for cluster in clusters:
                # Compare against cluster representative (first / highest-rel member).
                if _jaccard(terms, term_sets[cluster[0]]) >= self._cluster_threshold:
                    cluster.append(idx)
                    placed = True
                    break
            if not placed:
                clusters.append([idx])

        clusters.sort(
            key=lambda idxs: (
                -max(_relevance(ordered[i]) for i in idxs),
                min(ordered[i].rank for i in idxs),
                ordered[idxs[0]].id,
            )
        )
        clusters = clusters[: self._max_memories]

        out: list[ScoredSearchCandidate] = []
        for rank, idxs in enumerate(clusters):
            members = [ordered[i] for i in idxs]
            out.append(self._reduce_cluster(members, rank=rank))

        return ConsolidationResult(
            candidates=out,
            input_count=len(candidates),
            cluster_count=len(clusters),
            output_count=len(out),
        )

    def _reduce_cluster(self, members: list[ScoredSearchCandidate], *, rank: int) -> ScoredSearchCandidate:
        members = sorted(members, key=lambda c: (-_relevance(c), c.rank, c.id))
        kept: list[ScoredSearchCandidate] = []
        for member in members:
            if any(self._is_near_dup(member, existing) for existing in kept):
                continue
            kept.append(member)
            if len(kept) >= self._stitch_max_members:
                break
        if not kept:
            kept = [members[0]]

        if len(kept) == 1:
            primary = kept[0]
            return ScoredSearchCandidate(
                item=primary.item,
                original_rank=primary.original_rank,
                rank=rank,
                retrieval_score=primary.retrieval_score,
                retrieval_score_type=primary.retrieval_score_type,
                normalized_retrieval_score=primary.normalized_retrieval_score,
                rerank_score=primary.rerank_score,
                normalized_rerank_score=primary.normalized_rerank_score,
                relevance_score=_relevance(primary),
                final_score_source=primary.final_score_source,
                evidence=list(primary.evidence),
            )

        texts = [m.memory.strip() for m in kept if m.memory and m.memory.strip()]
        stitched = " | ".join(texts)
        if len(stitched) > self._max_chars:
            stitched = stitched[: self._max_chars].rstrip()
        primary = kept[0]
        consol_id = "consol:" + "+".join(m.id for m in kept)
        item = MemorySearchItem(
            id=consol_id,
            memory=stitched,
            memory_type=primary.memory_type,
            last_update_at=primary.last_update_at or "",
            event_time=primary.event_time,
            source_timestamp=primary.source_timestamp,
            lineage=primary.lineage,
            metadata={
                **(primary.metadata or {}),
                "consolidated_from": [m.id for m in kept],
            },
        )
        return ScoredSearchCandidate(
            item=item,
            original_rank=primary.original_rank,
            rank=rank,
            retrieval_score=primary.retrieval_score,
            retrieval_score_type=primary.retrieval_score_type,
            normalized_retrieval_score=primary.normalized_retrieval_score,
            rerank_score=primary.rerank_score,
            normalized_rerank_score=primary.normalized_rerank_score,
            relevance_score=max(_relevance(m) for m in kept),
            final_score_source=primary.final_score_source,
            evidence=list(primary.evidence),
        )

    def _is_near_dup(self, left: ScoredSearchCandidate, right: ScoredSearchCandidate) -> bool:
        if _digit_fingerprint(left.memory) != _digit_fingerprint(right.memory):
            return False
        return _jaccard(self._terms(left.memory), self._terms(right.memory)) >= self._near_dup_threshold

    def _terms(self, text: str) -> set[str]:
        processed = self._text_preprocessor.preprocess_text(text or "", include_entities=False)
        return {str(token).casefold() for token in processed.tokens if str(token).strip()}


def _relevance(candidate: ScoredSearchCandidate) -> float:
    score = candidate.relevance_score if isfinite(candidate.relevance_score) else 0.0
    return min(max(score, 0.0), 1.0)


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _digit_fingerprint(text: str) -> str:
    return "".join(re.findall(r"\d+", text or ""))
