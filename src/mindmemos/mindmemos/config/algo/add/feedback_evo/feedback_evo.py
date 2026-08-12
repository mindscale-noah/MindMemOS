"""Feedback-driven self-evolution add configuration (``feedback_evo`` mode)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FeedbackEvoAddConfig:
    """Configuration for the feedback_evo add pipeline.

    Evolvable surface: ``extraction_prompt`` / ``entity_tagging_prompt`` /
    ``entity_types`` are read live from the evolution state by the
    feedback_evo pipeline; ``enable_entities`` toggles entity extraction.
    """

    enable_entities: bool = field(default=False)
    """Whether the extractor requests entity tagging output."""

    extraction_prompt: str | None = field(default=None)
    """Optional live extraction prompt override (from evolution state)."""

    entity_tagging_prompt: str | None = field(default=None)
    """Optional entity-tag selection instruction appended to extraction."""

    entity_types: list[str] = field(default_factory=list)
    """Entity-type tag vocabulary; empty disables tagging."""

    extra: dict[str, Any] = field(default_factory=dict)
    """Unstructured evolvable add parameters (forward compatibility)."""
