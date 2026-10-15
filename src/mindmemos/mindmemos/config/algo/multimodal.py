"""Multimodal memory configuration of the vanilla algorithm."""

from __future__ import annotations

from dataclasses import dataclass

MULTIMODAL_MODES = frozenset({"off", "basic", "enhanced"})
IMAGE_DETAILS = frozenset({"auto", "low", "high"})


@dataclass
class MultimodalConfig:
    """Images in vanilla memory, selected by ``mode`` (per project overridable).

    - ``off``: plain vanilla; requests carrying images are rejected (400).
    - ``basic``: raw images are stored and embedded; a multimodal search appends the images found by
      text-to-image and image-to-image similarity after the text results.
    - ``enhanced``: turns with images are extracted by a VLM; dialogue facts go to text memory,
      anchored to their images, while visual facts and image descriptions stay on the image records
      (with a text vector, so image search scores text and image similarity together). A multimodal
      search adds facts recalled through matching images, the visual facts most relevant to the query
      (from the best-matching images), evidence images (anchored, similar and from the session of the
      best match) and, with ``search_strategy=agentic``, a multimodal agent.

    Text-only requests, and searches that are not multimodal (no ``query_images`` and
    ``multimodal=false``), behave exactly like vanilla in every mode.
    """

    mode: str = "off"
    """One of ``MULTIMODAL_MODES``."""

    image_search: bool = True
    """Search images by similarity (needs ``mm_embed_model_router``)."""

    image_top_k: int = 5
    """Basic: maximum image results appended after the text results."""

    image_score_threshold: float | None = None
    """Basic: minimum text-image similarity for an image hit; None keeps the top ``image_top_k``."""

    image_context_chars: int = 500
    """Characters of the surrounding dialogue kept with each stored image."""

    max_images_per_request: int = 32
    """Add requests carrying more images are rejected."""

    max_occurrences_per_image: int = 20
    """Contexts remembered per deduplicated image (oldest kept first)."""

    evidence_images: int = 6
    """Enhanced: maximum evidence images returned per search."""

    session_expansion_images: int = 4
    """Enhanced: images taken from the session of the best image match (it included), in order; 0 disables."""

    anchored_fact_limit: int = 5
    """Enhanced: maximum facts recalled through matching images, beyond the text results."""

    image_hit_margin: float = 0.03
    """Enhanced: an image hit counts as a match when its similarity exceeds the median of the hits by this much."""

    image_text_weight: float = 0.5
    """Enhanced: weight of the text similarity (query vs. description and visual facts) in image scores;
    the rest is image similarity. Each channel is normalized before mixing."""

    visual_fact_limit: int = 10
    """Enhanced: visual facts returned as separate results (reranked against the query); 0 disables."""

    visual_fact_images: int = 12
    """Enhanced: best-matching images whose visual facts are candidates (evidence images always are)."""

    max_visual_facts_per_image: int = 30
    """Enhanced: visual facts kept per image record (newest extraction first)."""

    extraction_image_detail: str = "auto"
    """Enhanced: image detail sent to the VLM during extraction (auto, low, high)."""

    max_extraction_images: int = 6
    """Enhanced: maximum images attached to one extraction call."""

    agentic_max_steps: int = 4
    """Enhanced: tool calls allowed in multimodal agentic retrieval before it must submit."""
