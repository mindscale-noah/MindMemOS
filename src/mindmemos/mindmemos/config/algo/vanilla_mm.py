"""vanilla_mm (vanilla + multimodal) algorithm configuration."""

from __future__ import annotations

from dataclasses import dataclass

IMAGE_DETAILS = frozenset({"auto", "low", "high"})


@dataclass
class VanillaMmConfig:
    """Image handling on top of the reused vanilla add/search pipelines.

    ``enhanced: false`` is the base: raw images are stored and a text-to-image similarity channel is
    appended to search. ``enhanced: true`` turns on the MHG-inspired features together: VLM fact
    extraction for turns with images (facts anchored to their images, one description per image),
    evidence images taken from the images anchored to retrieved facts, facts recalled through
    matching images (text-to-image and query-image-to-image), and a simplified multimodal agentic
    retrieval when the request uses ``search_strategy=agentic``. Text-only behavior is vanilla
    either way.
    """

    image_search: bool = True
    """Search images by similarity (needs ``mm_embed_model_router``)."""

    image_top_k: int = 5
    """Base mode: maximum image results appended after the text results."""

    image_score_threshold: float | None = None
    """Minimum text-image similarity for an image hit; None keeps the top ``image_top_k``."""

    image_context_chars: int = 500
    """Characters of the surrounding dialogue kept with each stored image."""

    max_images_per_request: int = 32
    """Add requests carrying more images are rejected."""

    max_occurrences_per_image: int = 20
    """Contexts remembered per deduplicated image (oldest kept first)."""

    enhanced: bool = False
    """Turn on all MHG-inspired features (see the class docstring)."""

    evidence_images: int = 6
    """Enhanced: maximum evidence images returned per search."""

    anchored_fact_limit: int = 5
    """Enhanced: maximum facts recalled through matching images, beyond the text results."""

    image_hit_margin: float = 0.03
    """Enhanced: an image hit counts as a match when its similarity exceeds the median of the hits by this much."""

    extraction_image_detail: str = "auto"
    """Enhanced: image detail sent to the VLM during extraction (auto, low, high)."""

    max_extraction_images: int = 6
    """Enhanced: maximum images attached to one extraction call."""

    agentic_max_steps: int = 4
    """Enhanced: tool calls allowed in multimodal agentic retrieval before it must submit."""
