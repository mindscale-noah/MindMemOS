"""vanilla_mm (vanilla + multimodal) algorithm configuration."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class VanillaMmFeatures:
    """MHG-inspired feature switches layered on the vanilla_mm base. All default off.

    The base (all switches off) stores raw images and adds a text-to-image similarity channel;
    text handling is exactly vanilla. Dependencies between switches are listed in
    ``VANILLA_MM_FEATURE_REQUIREMENTS`` and checked by config validation.
    """

    image_index: bool = False
    """Index each image with a VLM description plus the text visible in it."""

    multimodal_extraction: bool = False
    """Let a VLM extract facts from images together with the surrounding dialogue."""

    fact_image_anchoring: bool = False
    """Bind extracted facts to their source images and return those images as evidence."""

    query_image_understanding: bool = False
    """Interpret images attached to a search query before retrieval."""

    visual_entity_linking: bool = False
    """Link entities seen in images to entities mentioned in text."""

    agentic_retrieval: bool = False
    """Multimodal ReAct retrieval; runs only when the request uses search_strategy=agentic."""


VANILLA_MM_FEATURES: tuple[str, ...] = (
    "image_index",
    "multimodal_extraction",
    "fact_image_anchoring",
    "query_image_understanding",
    "visual_entity_linking",
    "agentic_retrieval",
)

VANILLA_MM_FEATURE_REQUIREMENTS: dict[str, tuple[tuple[str, ...], ...]] = {
    # Linking needs a textual view of what an image shows.
    "visual_entity_linking": (("image_index", "multimodal_extraction"),),
}
"""Feature -> requirement groups; every group needs at least one of its features enabled."""

VANILLA_MM_FEATURES_IMPLEMENTED: frozenset[str] = frozenset()
"""Features this build implements; enabling any other one is a config error."""


@dataclass
class VanillaMmConfig:
    """Image handling on top of the reused vanilla add/search pipelines."""

    image_search: bool = True
    """Add the text-to-image similarity channel to search (needs ``mm_embed_model_router``)."""

    image_top_k: int = 5
    """Maximum image results appended after the text results."""

    image_score_threshold: float | None = None
    """Minimum text-image similarity for an image result; None keeps the top ``image_top_k``."""

    image_context_chars: int = 500
    """Characters of the surrounding dialogue kept with each stored image."""

    max_images_per_request: int = 32
    """Add requests carrying more images are rejected."""

    max_occurrences_per_image: int = 20
    """Contexts remembered per deduplicated image (oldest kept first)."""

    features: VanillaMmFeatures = field(default_factory=VanillaMmFeatures)
    """MHG-inspired feature switches."""
