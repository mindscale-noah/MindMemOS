"""Configuration owned by the migrated vanilla add/search algorithm."""

from dataclasses import dataclass, field

from ..base import MindMemOSConfig, frozen_field
from ..components import TextProcessingConfig
from .add import VanillaAddConfig, VanillaAddRecallConfig, VanillaAddSafetyGateConfig
from .dreaming import DreamingConfig
from .search import (
    VANILLA_DEDUP_MAX_CANDIDATES,
    VANILLA_HYBRID_PREFETCH_FACTOR_MAX,
    VANILLA_HYBRID_PREFETCH_MAX,
    VANILLA_RECALL_SIZE_MAX,
    VanillaSearchConfig,
)
from .trajectory import TrajectoryAddConfig


@dataclass
class VanillaAlgorithmConfig(MindMemOSConfig):
    """Complete process configuration consumed by vanilla memory pipelines."""

    text_processing: TextProcessingConfig = frozen_field(default_factory=TextProcessingConfig)
    add: VanillaAddConfig = field(default_factory=VanillaAddConfig)
    search: VanillaSearchConfig = field(default_factory=VanillaSearchConfig)
    dreaming: DreamingConfig = field(default_factory=DreamingConfig)
    trajectory: TrajectoryAddConfig = field(default_factory=TrajectoryAddConfig)


__all__ = [
    "VANILLA_DEDUP_MAX_CANDIDATES",
    "VANILLA_HYBRID_PREFETCH_FACTOR_MAX",
    "VANILLA_HYBRID_PREFETCH_MAX",
    "VANILLA_RECALL_SIZE_MAX",
    "TextProcessingConfig",
    "DreamingConfig",
    "TrajectoryAddConfig",
    "VanillaAddConfig",
    "VanillaAddRecallConfig",
    "VanillaAddSafetyGateConfig",
    "VanillaAlgorithmConfig",
    "VanillaSearchConfig",
]
