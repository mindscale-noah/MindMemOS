"""Internal scored search candidates and query-local score utilities.

These types deliberately stay below the HTTP response boundary.  Public search
responses are projected from them only after reranking, retention, and top-k.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import isfinite
from typing import Literal

from ...typing import MemorySearchItem

RetrievalScoreType = Literal["bm25", "rrf", "graph_propagation", "schema", "agentic"]
FinalScoreSource = Literal["retrieval", "rerank", "rank_fallback"]
EvidenceSource = Literal["direct", "graph", "schema", "agentic", "lineage"]


@dataclass(frozen=True, slots=True)
class GraphPathEvidence:
    """Sanitized graph path metadata retained for one candidate."""

    seed_memory_id: str
    relation: str
    hops: int = 1
    decay: float | None = None
    path_score: float | None = None
    used_fallback: bool = False
    entity_id: str | None = None
    entity_name: str | None = None
    entity_type: str | None = None


@dataclass(frozen=True, slots=True)
class RetrievalEvidence:
    """One bounded, typed reason why a candidate was retrieved."""

    source: EvidenceSource
    score: float | None = None
    score_type: RetrievalScoreType | None = None
    rank: int | None = None
    query: str | None = None
    round_index: int | None = None
    engine: str | None = None
    graph: GraphPathEvidence | None = None


@dataclass(slots=True)
class ScoredSearchCandidate:
    """Internal search result carrying score and provenance until projection."""

    item: MemorySearchItem
    original_rank: int
    rank: int
    retrieval_score: float | None = None
    retrieval_score_type: RetrievalScoreType | None = None
    normalized_retrieval_score: float | None = None
    rerank_score: float | None = None
    normalized_rerank_score: float | None = None
    relevance_score: float = 0.0
    final_score_source: FinalScoreSource = "rank_fallback"
    evidence: list[RetrievalEvidence] = field(default_factory=list)

    @property
    def id(self) -> str:
        """Compatibility accessor for engine-level callers during migration."""

        return self.item.id

    @property
    def memory(self) -> str:
        return self.item.memory

    @property
    def memory_type(self):
        return self.item.memory_type

    @property
    def last_update_at(self) -> str:
        return self.item.last_update_at

    @property
    def event_time(self) -> str | None:
        return self.item.event_time

    @property
    def source_timestamp(self) -> str | None:
        return self.item.source_timestamp

    @property
    def lineage(self):
        return self.item.lineage

    @property
    def metadata(self):
        return self.item.metadata


def normalize_candidate_scores(
    candidates: list[ScoredSearchCandidate],
) -> list[ScoredSearchCandidate]:
    """Assign deterministic query-local relevance without reordering candidates."""

    if not candidates:
        return []
    scores = [candidate.retrieval_score for candidate in candidates]
    valid = all(score is not None and isfinite(score) for score in scores)
    if valid:
        numeric = [float(score) for score in scores if score is not None]
        lo, hi = min(numeric), max(numeric)
        if len(candidates) == 1:
            values = [1.0]
            source: FinalScoreSource = "retrieval"
        elif hi > lo:
            values = [(score - lo) / (hi - lo) for score in numeric]
            source = "retrieval"
        else:
            values = _rank_values(len(candidates))
            source = "rank_fallback"
    else:
        values = _rank_values(len(candidates))
        source = "rank_fallback"

    result: list[ScoredSearchCandidate] = []
    for candidate, value in zip(candidates, values, strict=True):
        candidate.normalized_retrieval_score = value if source == "retrieval" else None
        candidate.relevance_score = value
        candidate.final_score_source = source
        result.append(candidate)
    return result


def normalize_external_scores(scores: list[float]) -> list[float]:
    """Normalize a complete finite external score list while preserving its order."""

    if not scores:
        return []
    if len(scores) == 1:
        return [1.0]
    if not all(isfinite(score) for score in scores):
        return _rank_values(len(scores))
    lo, hi = min(scores), max(scores)
    if hi <= lo:
        return _rank_values(len(scores))
    return [(score - lo) / (hi - lo) for score in scores]


def merge_scored_candidates(
    candidates: list[ScoredSearchCandidate],
    *,
    evidence_limit: int = 8,
) -> list[ScoredSearchCandidate]:
    """Merge duplicate identities without summing path or round evidence."""

    by_id: dict[str, ScoredSearchCandidate] = {}
    order: list[str] = []
    for candidate in candidates:
        memory_id = candidate.item.id
        existing = by_id.get(memory_id)
        if existing is None:
            by_id[memory_id] = replace(candidate, evidence=list(candidate.evidence))
            order.append(memory_id)
            continue

        preferred = _preferred_primary(existing, candidate)
        merged_evidence = _merge_evidence(existing.evidence, candidate.evidence, evidence_limit)
        by_id[memory_id] = replace(
            preferred,
            original_rank=min(existing.original_rank, candidate.original_rank),
            rank=min(existing.rank, candidate.rank),
            evidence=merged_evidence,
        )
    return [by_id[memory_id] for memory_id in order]


def _preferred_primary(
    left: ScoredSearchCandidate,
    right: ScoredSearchCandidate,
) -> ScoredSearchCandidate:
    left_direct = left.retrieval_score_type != "graph_propagation"
    right_direct = right.retrieval_score_type != "graph_propagation"
    if left_direct != right_direct:
        return left if left_direct else right
    left_score = left.retrieval_score if _finite(left.retrieval_score) else float("-inf")
    right_score = right.retrieval_score if _finite(right.retrieval_score) else float("-inf")
    if right_score > left_score:
        return right
    return left


def _merge_evidence(
    left: list[RetrievalEvidence],
    right: list[RetrievalEvidence],
    limit: int,
) -> list[RetrievalEvidence]:
    if limit <= 0:
        return []
    seen: set[tuple[object, ...]] = set()
    merged: list[RetrievalEvidence] = []
    for evidence in [*left, *right]:
        graph = evidence.graph
        key = (
            evidence.source,
            evidence.score_type,
            evidence.rank,
            evidence.round_index,
            evidence.engine,
            graph.seed_memory_id if graph else None,
            graph.relation if graph else None,
            graph.hops if graph else None,
        )
        if key in seen:
            continue
        seen.add(key)
        merged.append(evidence)
        if len(merged) >= limit:
            break
    return merged


def _rank_values(size: int) -> list[float]:
    if size <= 0:
        return []
    if size == 1:
        return [1.0]
    return [1.0 - index / (size - 1) for index in range(size)]


def _finite(value: float | None) -> bool:
    return value is not None and isfinite(value)
