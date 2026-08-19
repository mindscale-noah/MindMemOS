"""Schema add root configuration."""

from __future__ import annotations

from dataclasses import dataclass, field

from .chunker import EpisodesChunkerConfig
from .drain import DrainConfig
from .episode_edge import SchemaAddEpisodeEdgeConfig
from .extraction import SchemaAddExtractionConfig
from .merge import SchemaAddMergeConfig


@dataclass
class SchemaAddConfig:
    """Configuration for the schema_add pipeline and components."""

    extraction: SchemaAddExtractionConfig = field(default_factory=SchemaAddExtractionConfig)
    merge: SchemaAddMergeConfig = field(default_factory=SchemaAddMergeConfig)
    episode_edge: SchemaAddEpisodeEdgeConfig = field(default_factory=SchemaAddEpisodeEdgeConfig)
    drain: DrainConfig = field(default_factory=DrainConfig)
    chunker: EpisodesChunkerConfig = field(default_factory=EpisodesChunkerConfig)

    entity_modeling_path: str = field(default="config/presets/schema_general.json")
    """Path to the entity schema modeling JSON file used by schema add extraction."""
