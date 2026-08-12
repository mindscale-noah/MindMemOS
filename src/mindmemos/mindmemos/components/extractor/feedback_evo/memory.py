"""Independent feedback_evo memory extractor.

Mirrors the flat-memory extraction approach (envelope prompt -> LLM -> parsed
candidates) but is implemented inside the feedback_evo component namespace and
adds entity-tagging support driven by the evolved ``entity_types`` vocabulary.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from ....logging import get_logger
from ....llm import LLMClient
from ....prompts.EN.feedback_evo import (
    FEEDBACK_EVO_ENTITY_TAGGING_PROMPT,
    FEEDBACK_EVO_EXTRACTION_SYSTEM_PROMPT,
)
from ....typing import (
    ExtractionEnvelope,
    MemoryRequestContext,
    PreprocessedText,
)
from ..vanilla.memory import MemoryExtractionResult

logger = get_logger(__name__)


class FeedbackEvoMemoryExtractor:
    """Extract memories with an optional live prompt and entity-type tags."""

    def __init__(
        self,
        *,
        llm_client: LLMClient | None = None,
        extraction_prompt: str | None = None,
        entity_tagging_prompt: str | None = None,
        entity_types: list[str] | None = None,
        enable_entities: bool = False,
    ) -> None:
        self._llm_client = llm_client
        self._extraction_prompt = extraction_prompt
        self._entity_tagging_prompt = entity_tagging_prompt
        self._entity_types = list(entity_types or [])
        self._enable_entities = enable_entities or bool(self._entity_types)

    @property
    def entity_types(self) -> list[str]:
        """The tag vocabulary this extractor was configured with."""

        return self._entity_types

    async def extract_from_envelope(
        self,
        envelope: ExtractionEnvelope,
        preprocessed_texts: list[PreprocessedText],
        context: MemoryRequestContext,
    ) -> MemoryExtractionResult:
        """Extract memories from a chunked envelope, tagging when configured."""

        if self._llm_client is None:
            logger.warning(
                "feedback_evo_extractor_no_llm",
                request_id=context.request_id,
            )
            return MemoryExtractionResult(memories=[])
        try:
            response = await self._llm_client.chat(
                task="memory.add.feedback_evo.extract",
                messages=_prompt_messages(
                    envelope,
                    preprocessed_texts,
                    context,
                    extraction_prompt=self._extraction_prompt,
                    entity_tagging_prompt=self._entity_tagging_prompt,
                    entity_types=self._entity_types,
                ),
                format_parser=_parse_extraction_json,
            )
            return MemoryExtractionResult.model_validate(
                _normalize_feedback_evo_extraction(response.parsed)
            )
        except Exception as exc:
            logger.warning(
                "feedback_evo_extraction_failed",
                request_id=context.request_id,
                error=str(exc),
            )
            return MemoryExtractionResult(memories=[])


def _normalize_feedback_evo_extraction(payload: Any) -> dict[str, Any]:
    """Normalize LLM shape drift before Pydantic validation.

    gpt-5.4 sometimes omits ``ref_id`` / ``content`` (or emits ``text``
    instead of ``content``) on memory objects. Without normalization the whole
    extraction is dropped; with it, memories without content are skipped and
    missing ref ids are synthesized so valid memories survive. It also
    normalizes ``source_refs`` (plain ints and ``{"evidence_index": N}`` /
    ``{"ref_id": ...}`` dicts are mapped onto the ``s{index}`` string form
    used by the mode), so one drifted field cannot invalidate the batch.
    """

    if not isinstance(payload, dict):
        return {"memories": []}
    normalized = dict(payload)
    memories: list[dict[str, Any]] = []
    for raw_memory in normalized.get("memories") or []:
        if not isinstance(raw_memory, dict):
            continue
        memory = dict(raw_memory)
        content = memory.get("content")
        if content is None or not str(content).strip():
            content = memory.get("text")
        if content is None or not str(content).strip():
            continue
        memory["content"] = str(content)
        if not isinstance(memory.get("ref_id"), str) or not memory["ref_id"].strip():
            memory["ref_id"] = f"m_{uuid.uuid4().hex}"
        memory["source_refs"] = _normalize_source_refs(memory.get("source_refs"))
        memories.append(memory)
    normalized["memories"] = memories
    return normalized


def _normalize_source_refs(source_refs: Any) -> list[str]:
    """Map LLM-shaped source references onto the ``s{evidence_index}`` form.

    Accepts strings unchanged; integers and ``{"evidence_index": N}`` dicts
    become ``s{N}``; ``{"ref_id": ...}`` dicts keep their ref id; anything
    else is dropped so the memory batch still validates.
    """

    if not isinstance(source_refs, list):
        return []
    normalized: list[str] = []
    for ref in source_refs:
        if isinstance(ref, str):
            if ref:
                normalized.append(ref)
            continue
        if isinstance(ref, int):
            normalized.append(f"s{ref}")
            continue
        if isinstance(ref, dict):
            ref_id = ref.get("ref_id")
            if isinstance(ref_id, str) and ref_id:
                normalized.append(ref_id)
                continue
            evidence_index = ref.get("evidence_index")
            if isinstance(evidence_index, int):
                normalized.append(f"s{evidence_index}")
    return normalized


def _prompt_messages(
    envelope: ExtractionEnvelope,
    preprocessed_texts: list[PreprocessedText],
    context: MemoryRequestContext,
    *,
    extraction_prompt: str | None,
    entity_tagging_prompt: str | None,
    entity_types: list[str],
) -> list[dict[str, Any]]:
    """Build the extraction prompt for one chunked envelope."""

    extractable: list[dict[str, Any]] = []
    for index, (msg_ref, preprocessed) in enumerate(
        zip(envelope.extractable_messages, preprocessed_texts, strict=False)
    ):
        extractable.append(
            {
                "index": msg_ref.message_index,
                "evidence_index": index,
                "role": msg_ref.role,
                "raw_role": msg_ref.raw_role,
                "speaker": msg_ref.speaker,
                "text": preprocessed.normalized_text,
                "is_extractable": msg_ref.is_extractable,
            }
        )

    context_section: dict[str, Any] = {}
    if envelope.history.in_request_history:
        context_section["history"] = [
            _turn_payload(turn) for turn in envelope.history.in_request_history
        ]
    if envelope.history.external_history:
        context_section["external_history"] = [
            _turn_payload(turn) for turn in envelope.history.external_history
        ]
    if envelope.recalled_memories:
        context_section["related_memories"] = envelope.recalled_memories

    instruction = (
        "EXTRACT memories ONLY from the 'extractable' section below. "
        "The 'context' section is for reference resolution and duplicate "
        "detection only — do not create new memories from context."
    )
    if entity_types:
        instruction += (
            "\n[Entity tagging] For every memory, assign the single most "
            f"specific entity_type from the vocabulary {entity_types!r}. "
            "Optionally assign a property_name for finer classification. "
            "Output entity_type and property_name fields on each memory object."
        )
        instruction += f"\n{entity_tagging_prompt or FEEDBACK_EVO_ENTITY_TAGGING_PROMPT}"

    payload: dict[str, Any] = {
        "request_id": context.request_id,
        "project_id": context.project_id,
        "chunk_index": envelope.chunk_index,
        "boundary": envelope.boundary,
        "instruction": instruction,
        "extractable": extractable,
        "context": context_section,
    }

    system_prompt = extraction_prompt or FEEDBACK_EVO_EXTRACTION_SYSTEM_PROMPT
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def _turn_payload(turn: Any) -> dict[str, Any]:
    """Convert one history turn into a compact prompt dict."""

    return {
        "text": getattr(turn, "text", "") or "",
        "messages": getattr(turn, "messages", []) or [],
    }


def _parse_extraction_json(content: str) -> dict[str, Any]:
    """Parse extractor JSON, tolerating simple markdown JSON fences."""

    text = content.strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").strip()
        text = text.removesuffix("```").strip()
    return json.loads(text)
