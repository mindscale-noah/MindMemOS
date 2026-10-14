"""VLM fact extraction for chunks that carry images (vanilla_mm enhanced mode).

``MultimodalMemoryExtractor`` wraps the vanilla extractor. A chunk without images goes to the
vanilla extractor untouched. A chunk with images makes the same single extraction call, built
from the same vanilla prompt, with the images attached and rules for writing visual facts
(adapted from MHG's per-round memorizer). The same call also returns one description per image.
Each extracted fact is anchored to the images of the turns it came from
(``metadata["image_blob_ids"]``).

The images of the current request reach the shared extractor through a context variable that
the add pipeline sets per request (``request_images``).
"""

from __future__ import annotations

import base64
import contextvars
import json
import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from ...logging import get_logger
from ...typing import ExtractionEnvelope, MemoryRequestContext, PreprocessedText
from ..extractor.vanilla.memory import (
    MemoryExtractionResult,
    _envelope_prompt_messages,
    _mark_extractor,
    _normalize_extraction_payload,
    parse_memory_extraction_json,
)

logger = get_logger(__name__)
IMAGE_BLOB_IDS_KEY = "image_blob_ids"

VISUAL_RULES = """## Images in this chunk (these rules take precedence over any conflicting rule below)
Some extractable messages carry images. They are attached after the JSON payload, each labelled
"IMAGE k (evidence_index i)". The pixels are direct evidence of message i: a fact read from the
image is supported by source_ref "s{i}", like the message text.
- For EVERY attached image, output at least one memory describing what it shows: the subject
  (brand, product, person, place, screen), the key objects, colors, counts, spatial layout
  (left/right/above/below), and all visible text copied verbatim (signs, labels, prices, numbers,
  UI or chat text). Make it self-contained and dated, e.g. "On 2024-01-05 the user shared a
  Coca-Cola and Citi sponsorship ad with a red background and the slogan '...'". Use mem_type fact.
- An assistant message describing an attached image may be used when it agrees with the pixels.
- Name things explicitly; never write "this image", "the photo" or "the screen". The labels
  "IMAGE k" exist only in this request: never mention them in memories or descriptions.
- Describe the state that is shown; do not compute answers (no totals that are not displayed).
- Several distinct panels, items or people shown: write separate facts for them.
- An image resembling earlier ones (same template, game, route, dashboard): include the detail
  that makes this one identifiable (its date, its distinctive values).
- Output, in addition to "memories", the top-level field "image_descriptions":
  [{"image": k, "description": "<what it shows plus all visible text, 1-3 sentences>"}] with one
  entry per attached image. This extra field is required for chunks with images.

"""


REMINDER = """Before answering: the attached images ({numbers}) are evidence too. Output at least one
memory per image describing what it shows (subject, colors, layout, counts, visible text verbatim),
plus the top-level "image_descriptions" field with one entry per image."""


@dataclass(frozen=True)
class ExtractionImage:
    blob_id: str
    mime_type: str
    data: bytes


@dataclass
class RequestImages:
    """Images of one add request, keyed by the message index vanilla sees."""

    by_message: dict[int, list[ExtractionImage]]
    detail: str = "auto"
    max_images: int = 6
    descriptions: dict[str, str] = field(default_factory=dict)
    """Filled by the extractor: blob_id -> VLM description."""

    def for_message(self, index: int) -> list[ExtractionImage]:
        return self.by_message.get(index, [])


_REQUEST_IMAGES: contextvars.ContextVar[RequestImages | None] = contextvars.ContextVar(
    "vanilla_mm_request_images", default=None
)


@contextmanager
def request_images(images: RequestImages) -> Iterator[RequestImages]:
    token = _REQUEST_IMAGES.set(images)
    try:
        yield images
    finally:
        _REQUEST_IMAGES.reset(token)


class MultimodalMemoryExtractor:
    """Vanilla extraction, with one VLM call instead of the text call for chunks with images."""

    def __init__(self, inner: Any, *, llm_client: Any) -> None:
        self._inner = inner
        self._llm_client = llm_client

    async def extract_from_envelope(
        self,
        envelope: ExtractionEnvelope,
        preprocessed_texts: list[PreprocessedText],
        context: MemoryRequestContext,
    ) -> MemoryExtractionResult:
        images = _REQUEST_IMAGES.get()
        attached = _chunk_images(envelope, images) if images is not None else []
        if not attached or self._llm_client is None:
            return await self._inner.extract_from_envelope(envelope, preprocessed_texts, context)
        assert images is not None
        prompt = _envelope_prompt_messages(
            envelope, preprocessed_texts, context, enable_entities=bool(getattr(self._inner, "_enable_entities", False))
        )
        messages = _with_images(prompt, attached, detail=images.detail)
        try:
            response = await self._llm_client.chat(
                task="memory.add.extract_multimodal",
                messages=messages,
                format_parser=parse_memory_extraction_json,
            )
        except Exception:  # noqa: BLE001 - fall back to text-only vanilla extraction
            logger.warning("multimodal_extraction_failed", request_id=context.request_id, exc_info=True)
            return await self._inner.extract_from_envelope(envelope, preprocessed_texts, context)
        parsed = response.parsed if isinstance(response.parsed, dict) else {}
        result = MemoryExtractionResult.model_validate(_normalize_extraction_payload(parsed))
        result = result.model_copy(
            update={"memories": [m.model_copy(update={"content": strip_labels(m.content)}) for m in result.memories]}
        )
        _collect_descriptions(parsed, attached, images)
        return _anchor(_mark_extractor(result, "vanilla_mm_vlm"), envelope, attached)


@dataclass(frozen=True)
class _Attached:
    number: int
    evidence_index: int
    image: ExtractionImage


def _chunk_images(envelope: ExtractionEnvelope, images: RequestImages) -> list[_Attached]:
    attached: list[_Attached] = []
    seen: set[str] = set()
    for evidence_index, message in enumerate(envelope.extractable_messages):
        for image in images.for_message(message.message_index):
            if image.blob_id in seen or len(attached) >= images.max_images:
                continue
            seen.add(image.blob_id)
            attached.append(_Attached(number=len(attached) + 1, evidence_index=evidence_index, image=image))
    return attached


def _with_images(messages: list[dict[str, Any]], attached: list[_Attached], *, detail: str) -> list[dict[str, Any]]:
    system, user = messages[0], messages[1]
    payload = json.loads(user["content"])
    labels: dict[int, list[str]] = {}
    for item in attached:
        labels.setdefault(item.evidence_index, []).append(f"IMAGE {item.number}")
    for entry in payload.get("extractable", []):
        if entry.get("evidence_index") in labels:
            entry["images"] = labels[entry["evidence_index"]]
    parts: list[dict[str, Any]] = [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}]
    for item in attached:
        data = base64.b64encode(item.image.data).decode("ascii")
        parts.append({"type": "text", "text": f"IMAGE {item.number} (evidence_index {item.evidence_index}):"})
        parts.append(
            {"type": "image_url", "image_url": {"url": f"data:{item.image.mime_type};base64,{data}", "detail": detail}}
        )
    numbers = ", ".join(f"IMAGE {item.number}" for item in attached)
    parts.append({"type": "text", "text": REMINDER.format(numbers=numbers)})
    return [{"role": "system", "content": VISUAL_RULES + system["content"]}, {"role": "user", "content": parts}]


def _collect_descriptions(parsed: dict[str, Any], attached: list[_Attached], images: RequestImages) -> None:
    by_number = {item.number: item.image.blob_id for item in attached}
    for entry in parsed.get("image_descriptions") or []:
        if not isinstance(entry, dict):
            continue
        try:
            blob_id = by_number.get(int(entry.get("image")))
        except (TypeError, ValueError):
            continue
        description = str(entry.get("description") or "").strip()
        if blob_id and description:
            images.descriptions[blob_id] = strip_labels(description)[:1000]


def _anchor(
    result: MemoryExtractionResult, envelope: ExtractionEnvelope, attached: list[_Attached]
) -> MemoryExtractionResult:
    """Anchor each fact to the images of its source messages and of the message they reply to."""

    by_evidence: dict[int, list[str]] = {}
    for item in attached:
        by_evidence.setdefault(item.evidence_index, []).append(item.image.blob_id)
        # The reply right after an image message usually describes that image.
        by_evidence.setdefault(item.evidence_index + 1, []).append(item.image.blob_id)
    only_images = list(dict.fromkeys(item.image.blob_id for item in attached))
    memories = []
    for memory in result.memories:
        indices = [i for i in (_evidence_index(ref) for ref in memory.source_refs) if i is not None]
        blob_ids = [blob_id for i in indices for blob_id in by_evidence.get(i, [])]
        if not indices and len(only_images) == 1:
            # No usable provenance: in a chunk with a single image, the fact is about that turn.
            blob_ids = only_images
        blob_ids = list(dict.fromkeys(blob_ids))
        if blob_ids:
            memory = memory.model_copy(update={"metadata": {**dict(memory.metadata), IMAGE_BLOB_IDS_KEY: blob_ids}})
        memories.append(memory)
    return result.model_copy(update={"memories": memories})


def _evidence_index(ref: str) -> int | None:
    digits = str(ref).rsplit("s", 1)[-1]
    return int(digits) if digits.isdigit() else None


_PAREN_LABEL = re.compile(r"\s*\(\s*IMAGE\s+\d+[^)]*\)", re.IGNORECASE)
_LABEL = re.compile(r"\bIMAGE\s+\d+\b(\s*\(evidence_index\s+\d+\))?", re.IGNORECASE)


def strip_labels(text: str) -> str:
    """Remove leaked request-local labels: "(IMAGE 2)" disappears, "IMAGE 2 shows" -> "The image shows"."""

    cleaned = _LABEL.sub("the image", _PAREN_LABEL.sub("", text))
    return cleaned[:1].upper() + cleaned[1:] if cleaned.startswith("the image") else cleaned
