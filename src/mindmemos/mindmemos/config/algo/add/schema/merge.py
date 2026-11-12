"""Schema add merge policy configuration."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SchemaAddMergeConfig:
    """Entity and property merge configuration for schema add."""

    enable_entity_merge_decision: bool = field(default=True)
    """Whether schema add asks the LLM to decide create/update for recalled candidates (v1 flow)."""

    entity_recall_top_k: int = field(default=15)
    """Number of entity candidates recalled before the entity merge decision."""

    max_merge_retries: int = field(default=8)
    """Maximum LLM retries for entity merge decisions (v1 flow)."""

    use_property_merge: bool = field(default=False)
    """Whether schema add runs the property merge/delete decision prompt (v1 flow)."""

    secondary_search_limit: int = field(default=30)
    """Entity-name fallback search limit when an LLM update target is not in primary recall (v1 flow)."""

    secondary_search_retries: int = field(default=3)
    """Retry count for entity-name fallback search (v1 flow)."""

    secondary_search_retry_backoff_base: float = field(default=0.2)
    """Base seconds for exponential backoff between secondary search retries (v1 flow)."""

    secondary_search_retry_backoff_max: float = field(default=5.0)
    """Maximum seconds between secondary search retries (v1 flow)."""

    description_rewrite_threshold: int = field(default=1000)
    """Rule-merged entity description length (chars) that triggers one LLM rewrite to
    compress it (v2 flow). Below the threshold the merge stays a plain concatenation."""

    description_max_chars: int = field(default=2000)
    """Hard cap on a stored entity description. When the merge result exceeds it the
    oldest segments are dropped first so the newest information survives (v2 flow)."""

    reference_description_max_chars: int = field(default=500)
    """Per-entity description slice injected into the reference-entity prompt and the
    entity embedding text (v2 flow)."""
