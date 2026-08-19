"""Schema add reference-recall configuration."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SchemaAddMergeConfig:
    """Reference-entity recall configuration for rule-based schema add fusion."""

    entity_recall_top_k: int = field(default=15)
    """Number of reference entities recalled to drive rule-based new/update decisions."""
