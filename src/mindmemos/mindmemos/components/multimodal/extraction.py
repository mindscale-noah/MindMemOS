"""VLM fact extraction for chunks that carry images (vanilla, ``algo_config.multimodal.mode=enhanced``).

``MultimodalMemoryExtractor`` wraps the vanilla extractor. A chunk without images goes to the
vanilla extractor untouched. A chunk with images makes the same single extraction call, built
from the same vanilla prompt, with the images attached and rules for reading them (adapted from
MHG's per-round memorizer). The call returns two separate outputs:

- ``memories``: dialogue facts, written to text memory by the vanilla pipeline as usual. Each is
  anchored to the images of the turns it came from (``metadata["image_blob_ids"]``).
- ``image_memories``: per image, a description and visual facts. They are collected on the
  request (``RequestImages.visuals``) and stored on the image records, never in text memory.

The call attaches up to ``max_images`` images; a chunk with more also makes visual-only calls (up to
``max_images`` images each, alongside the main call) that describe the rest from the turns they were
shared in. Dialogue facts are anchored to every image of their turns.

The images of the current request reach the shared extractor through a context variable that
the add pipeline sets per request (``request_images``).
"""

from __future__ import annotations

import asyncio
import base64
import contextvars
import json
import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
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
"IMAGE k (evidence_index i)"; the payload lists the labels under that message's "images".
Write two separate outputs:
1. "memories": the facts of the dialogue, following the rules below. Use the images only to
   understand what the messages refer to (e.g. which item "this one" is); do not put descriptions
   of image content in "memories".
2. "image_memories" (top-level field, required for chunks with images): one entry per attached
   image, [{"image": k, "description": "<what it shows plus all visible text, 1-3 sentences>",
   "facts": ["<visual fact>", ...]}]. Visual facts record what the pixels show: the subject (brand,
   product, person, place, screen), the key objects, colors, counts, spatial layout
   (left/right/above/below), and all visible text copied verbatim (signs, labels, prices, numbers,
   UI or chat text). Each fact is self-contained and dated, e.g. "On 2024-01-05 the user shared a
   Coca-Cola and Citi sponsorship ad with a red background and the slogan '...'".
   - An assistant message describing an attached image may be used when it agrees with the pixels.
   - Several distinct panels, items or people shown: write separate facts for them.
   - An image resembling earlier ones (same template, game, route, dashboard): include the detail
     that makes this one identifiable (its date, its distinctive values).
   - Describe the state that is shown; do not compute answers (no totals that are not displayed).
Name things explicitly; never write "this image", "the photo" or "the screen". The labels
"IMAGE k" exist only in this request: never mention them in memories, descriptions or facts.

"""


OVERFLOW_PROMPT = """You record what the images shared in a conversation show. Each image is labelled "IMAGE k" and
preceded by the turn it was shared in, with its date. Output only JSON:
{"image_memories": [{"image": k, "description": "<what it shows plus all visible text, 1-3 sentences>",
"facts": ["<visual fact>", ...]}]}, one entry per image, with k the number of its label.
Visual facts record what the pixels show: the subject (brand, product, person, place, screen), the key
objects, colors, counts, spatial layout (left/right/above/below), and all visible text copied verbatim
(signs, labels, prices, numbers, UI or chat text). Each fact is self-contained and dated. Use the turn to
know what the image is (an assistant reply describing it may be used when it agrees with the pixels).
Write separate facts for distinct panels, items or people; for an image resembling others, include what
makes this one identifiable; describe the state shown, without computing answers. Name things explicitly;
never write "this image" or "the photo", and never mention the "IMAGE k" labels."""


REMINDER = """Before answering: the attached images ({numbers}) are evidence too. Output the top-level
"image_memories" field with one entry per image: its description and its visual facts (subject,
colors, layout, counts, visible text verbatim). Keep "memories" for the facts of the dialogue."""


@dataclass(frozen=True)
class ImageVisuals:
    """What the VLM read from one image: stored on its image record, not in text memory."""

    description: str
    facts: list[str]


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
    max_facts: int = 30
    visuals: dict[str, ImageVisuals] = field(default_factory=dict)
    """Filled by the extractor: blob_id -> description and visual facts."""

    def for_message(self, index: int) -> list[ExtractionImage]:
        return self.by_message.get(index, [])


_REQUEST_IMAGES: contextvars.ContextVar[RequestImages | None] = contextvars.ContextVar(
    "multimodal_request_images", default=None
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
        chunk = _chunk_images(envelope, images) if images is not None else []
        if not chunk or self._llm_client is None:
            return await self._inner.extract_from_envelope(envelope, preprocessed_texts, context)
        assert images is not None
        # One call extracts the dialogue and the first max_images images; the rest of the chunk's
        # images are described by visual-only calls running alongside it.
        attached, overflow = chunk[: images.max_images], chunk[images.max_images :]
        batches = [overflow[i : i + images.max_images] for i in range(0, len(overflow), images.max_images)]
        result, *_ = await asyncio.gather(
            self._extract(envelope, preprocessed_texts, context, images, attached, chunk),
            *(self._describe(envelope, context, images, batch) for batch in batches),
        )
        missing = [item.number for item in chunk if item.image.blob_id not in images.visuals]
        if missing:
            # Those images stay searchable by similarity only (no description, no visual facts).
            logger.warning(
                "multimodal_visuals_missing", request_id=context.request_id, images=missing, chunk_images=len(chunk)
            )
        return result

    async def _extract(
        self,
        envelope: ExtractionEnvelope,
        preprocessed_texts: list[PreprocessedText],
        context: MemoryRequestContext,
        images: RequestImages,
        attached: list[_Attached],
        chunk: list[_Attached],
    ) -> MemoryExtractionResult:
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
        _collect_visuals(parsed, attached, images)
        try:
            result = MemoryExtractionResult.model_validate(_normalize_extraction_payload(parsed))
        except Exception:  # noqa: BLE001 - e.g. an unknown mem_type: dialogue facts from vanilla instead
            logger.warning("multimodal_extraction_invalid", request_id=context.request_id, exc_info=True)
            return await self._inner.extract_from_envelope(envelope, preprocessed_texts, context)
        result = result.model_copy(
            update={"memories": [m.model_copy(update={"content": strip_labels(m.content)}) for m in result.memories]}
        )
        return _anchor(_mark_extractor(result, "multimodal_vlm"), chunk)

    async def _describe(
        self,
        envelope: ExtractionEnvelope,
        context: MemoryRequestContext,
        images: RequestImages,
        batch: list[_Attached],
    ) -> None:
        """Visual-only call for images beyond the main call's limit: their descriptions and visual facts."""

        numbered = [
            _Attached(number=i, evidence_index=item.evidence_index, image=item.image) for i, item in enumerate(batch, 1)
        ]
        parts: list[dict[str, Any]] = []
        for item in numbered:
            data = base64.b64encode(item.image.data).decode("ascii")
            parts.append(
                {"type": "text", "text": f"IMAGE {item.number}, {_turn_context(envelope, item.evidence_index)}"}
            )
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{item.image.mime_type};base64,{data}", "detail": images.detail},
                }
            )
        try:
            response = await self._llm_client.chat(
                task="memory.add.describe_images",
                messages=[{"role": "system", "content": OVERFLOW_PROMPT}, {"role": "user", "content": parts}],
                format_parser=parse_memory_extraction_json,
            )
        except Exception:  # noqa: BLE001 - the images stay searchable by similarity
            logger.warning("multimodal_image_description_failed", request_id=context.request_id, exc_info=True)
            return
        _collect_visuals(response.parsed if isinstance(response.parsed, dict) else {}, numbered, images)


@dataclass(frozen=True)
class _Attached:
    number: int
    evidence_index: int
    image: ExtractionImage


def _chunk_images(envelope: ExtractionEnvelope, images: RequestImages) -> list[_Attached]:
    """Every distinct image of the chunk's extractable messages, numbered in order."""

    attached: list[_Attached] = []
    seen: set[str] = set()
    for evidence_index, message in enumerate(envelope.extractable_messages):
        for image in images.for_message(message.message_index):
            if image.blob_id in seen:
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


def _collect_visuals(parsed: dict[str, Any], attached: list[_Attached], images: RequestImages) -> None:
    by_number = {item.number: item.image.blob_id for item in attached}
    entries = [entry for entry in parsed.get("image_memories") or [] if isinstance(entry, dict)]
    # Entries without a usable number still map by position when there is one per attached image.
    by_position = len(entries) == len(attached)
    for position, entry in enumerate(entries):
        number = _image_number(entry.get("image"))
        if number is None and by_position:
            number = attached[position].number
        blob_id = by_number.get(number) if number is not None else None
        description = strip_labels(str(entry.get("description") or "").strip())[:1000]
        raw_facts = entry.get("facts") or []
        raw_facts = [raw_facts] if isinstance(raw_facts, str) else raw_facts
        facts = [strip_labels(str(fact).strip())[:1000] for fact in raw_facts if str(fact).strip()]
        facts = list(dict.fromkeys(facts))[: images.max_facts]
        if blob_id and (description or facts):
            images.visuals[blob_id] = ImageVisuals(description=description, facts=facts)


def _turn_context(envelope: ExtractionEnvelope, evidence_index: int) -> str:
    """The turn an image was shared in (and the reply right after it), with its date."""

    messages = envelope.extractable_messages
    shared = messages[evidence_index]
    lines = [f"{shared.role}: {shared.text}"]
    if evidence_index + 1 < len(messages) and messages[evidence_index + 1].role != shared.role:
        reply = messages[evidence_index + 1]
        lines.append(f"{reply.role}: {reply.text}")
    when = (
        datetime.fromtimestamp(shared.timestamp / 1000, tz=UTC).date().isoformat()
        if shared.timestamp
        else "date unknown"
    )
    return f"shared on {when} in this turn:\n" + "\n".join(line[:1000] for line in lines)


def _image_number(value: Any) -> int | None:
    """The image number of an entry: 2, "2", "IMAGE 2" and "image_2" all mean IMAGE 2."""

    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    match = re.fullmatch(r"\s*(?:image[\s_#-]*)?(\d+)\s*", str(value or ""), re.IGNORECASE)
    return int(match.group(1)) if match else None


def _anchor(result: MemoryExtractionResult, attached: list[_Attached]) -> MemoryExtractionResult:
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
