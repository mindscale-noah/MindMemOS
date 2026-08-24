"""Token-budget memory retention configuration."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MemoryRetentionConfig:
    """Validated request bounds and deterministic mixed-score parameters."""

    min_token_budget: int = field(default=1)
    """Minimum token budget accepted for request-gated retention."""

    max_token_budget: int = field(default=128000)
    """Maximum token budget accepted for request-gated retention."""

    max_candidates: int = field(default=100)
    """Maximum candidate memories scored per retention pass."""

    relevance_weight: float = field(default=0.50)
    """Weight of the rerank relevance term in the mixed priority score."""

    query_overlap_weight: float = field(default=0.25)
    """Weight of the query-term overlap term in the mixed priority score."""

    recency_weight: float = field(default=0.15)
    """Weight of the recency term in the mixed priority score."""

    cost_weight: float = field(default=0.10)
    """Weight subtracted for token cost relative to the request budget."""

    recency_half_life_days: float = field(default=30.0)
    """Half-life in days for the exponential recency decay."""

    missing_recency_score: float = field(default=0.5)
    """Recency score used when a candidate has no parsable timestamp."""

    graph_provenance_limit: int = field(default=8)
    """Maximum evidence entries kept per scored candidate."""

    selector_version: str = field(default="mixed-v1")
    """Retention selector implementation: ``mixed-v1`` or ``mixed-v2``."""

    estimator_version: str = field(default="heuristic-v1")
    """Token estimator implementation (currently heuristic only)."""

    top_m_guarantee: int = field(default=5)
    """Candidates force-kept by relevance before MMR re-ranking (mixed-v2)."""

    mmr_lambda: float = field(default=0.70)
    """Trade-off between priority and redundancy in MMR packing (mixed-v2)."""

    consolidation_enabled: bool = field(default=False)
    """Whether consolidation runs before retention when enabled."""

    consolidation_max_memories: int = field(default=40)
    """Maximum clusters produced per consolidation pass."""

    consolidation_cluster_threshold: float = field(default=0.50)
    """Jaccard similarity at which candidates join the same cluster."""

    consolidation_near_dup_threshold: float = field(default=0.85)
    """Jaccard similarity at which intra-cluster members count as duplicates."""

    consolidation_stitch_max_members: int = field(default=3)
    """Maximum cluster members stitched into one consolidated memory."""

    consolidation_max_chars: int = field(default=600)
    """Character cap for a stitched consolidated memory."""
