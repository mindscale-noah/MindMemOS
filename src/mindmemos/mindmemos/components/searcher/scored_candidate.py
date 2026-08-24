"""Internal scored search candidates below the HTTP response boundary.

These types deliberately stay below the HTTP response boundary.  Public search
responses are projected from them only after reranking, retention, and top-k.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from ...typing import MemorySearchItem


@dataclass(slots=True)
class ScoredSearchCandidate:
    """Internal search result carrying rank and relevance until projection."""

    item: MemorySearchItem
    rank: int
    relevance_score: float = 0.0

    @property
    def id(self) -> str:
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


def merge_scored_candidates(candidates: list[ScoredSearchCandidate]) -> list[ScoredSearchCandidate]:
    """Merge duplicate identities, keeping the strongest relevance per id.

    Engines and agentic rounds may surface the same memory more than once; the
    merged candidate keeps the highest relevance score and the best (lowest)
    rank so downstream retention scoring stays deterministic.
    """

    by_id: dict[str, ScoredSearchCandidate] = {}
    order: list[str] = []
    for candidate in candidates:
        memory_id = candidate.item.id
        existing = by_id.get(memory_id)
        if existing is None:
            by_id[memory_id] = candidate
            order.append(memory_id)
            continue
        preferred = candidate if candidate.relevance_score > existing.relevance_score else existing
        by_id[memory_id] = replace(preferred, rank=min(existing.rank, candidate.rank))
    return [by_id[memory_id] for memory_id in order]
