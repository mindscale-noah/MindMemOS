"""Add-operation algorithm configuration."""

from __future__ import annotations

from dataclasses import dataclass, field

from .feedback_evo import FeedbackEvoAddConfig
from .schema import (
    DrainConfig,
    EpisodesChunkerConfig,
    SchemaAddConfig,
    SchemaAddEpisodeEdgeConfig,
    SchemaAddExtractionConfig,
    SchemaAddHigherOrderConfig,
    SchemaAddMergeConfig,
)
from .vanilla import VanillaAddConfig


@dataclass
class AddAlgoConfig:
    """Configuration for add-operation algorithms."""

    schema: SchemaAddConfig = field(default_factory=SchemaAddConfig)
    vanilla: VanillaAddConfig = field(default_factory=VanillaAddConfig)
    feedback_evo: FeedbackEvoAddConfig = field(default_factory=FeedbackEvoAddConfig)


__all__ = [
    "AddAlgoConfig",
    "DrainConfig",
    "EpisodesChunkerConfig",
    "FeedbackEvoAddConfig",
    "SchemaAddConfig",
    "SchemaAddEpisodeEdgeConfig",
    "SchemaAddExtractionConfig",
    "SchemaAddHigherOrderConfig",
    "SchemaAddMergeConfig",
    "VanillaAddConfig",
]
