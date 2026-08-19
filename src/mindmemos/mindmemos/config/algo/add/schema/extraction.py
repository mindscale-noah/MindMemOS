"""Schema add extraction configuration."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SchemaAddExtractionConfig:
    """LLM extraction and episode search-field configuration."""

    enable_schema_selection: bool = field(default=True)
    """Whether schema add asks the LLM to select relevant entity schemas before extraction."""

    search_fields_max: int = field(default=10)
    """Maximum number of search fields generated for the episode entity."""

    max_entities_per_conversation: int = field(default=200)
    """Hard cap on the number of entities processed from a single conversation."""

    max_properties_per_entity: int = field(default=15)
    """Hard cap on properties processed per entity. Each property becomes a memory point with its own
    embedding + write, so this bounds per-entity fan-out (and, via the entity cap, total fan-out).
    Properties beyond this limit are dropped. Must be >= 1; defaults to ~12 schema properties + headroom."""
