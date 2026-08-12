"""Feedback-driven self-evolution search configuration (``feedback_evo`` mode)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FeedbackEvoSearchConfig:
    """Configuration for the feedback_evo search engine.

    ``tag_weights`` / ``top_k`` / ``rerank`` / ``search_strategy`` are read
    live from the evolution state; the remaining fields mirror the flat-memory
    hybrid recall defaults used by the feedback_evo engine.
    """

    recall_size: int = field(default=20)
    """Over-retrieval count for the hybrid recall phase."""

    hybrid_prefetch_factor: int = field(default=3)
    """Prefetch multiplier applied to the recall size."""

    hybrid_prefetch_min: int = field(default=30)
    """Minimum prefetch limit for hybrid recall."""

    hybrid_prefetch_max: int = field(default=300)
    """Maximum prefetch limit for hybrid recall."""

    tag_weights: dict[str, float] = field(default_factory=dict)
    """Score multipliers keyed by memory ``entity_type`` (fallback ``mem_type``)."""

    extra: dict[str, Any] = field(default_factory=dict)
    """Unstructured evolvable search parameters (forward compatibility)."""
