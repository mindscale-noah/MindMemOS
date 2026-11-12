"""Environment-specific settings for exact TreeSkill reference execution."""

from ..config import TreeSkillEnvAdapterConfig


class TreeSkillSpreadsheetBenchReferenceEnvAdapterConfig(TreeSkillEnvAdapterConfig):
    """Configure routing while preserving the released tool-analysis protocol."""

    analysis_system: None = None
    """Generic one-call analysis overrides are incompatible with reference tool analysis."""
