"""Prompt-driven schema add extraction operators."""

from __future__ import annotations

import copy
import json
from typing import Any

from ....llm import LLMClient
from ....logging import get_logger
from ....prompts import AddPromptSet
from ...memory_modeling.schema import EntitySchemaProvider
from ._runtime_clients import resolve_llm_client
from ._schema_utils import (
    build_filtered_schema,
    format_reference_entities,
    format_schema_summary,
    has_unique_entity_names,
    parse_json_object,
    strip_for_generation,
)
from .base import SchemaEpisodeExtractor
from .schema_normalizer import SchemaExtractionNormalizer

logger = get_logger(__name__)


class SchemaAddExtractor(SchemaEpisodeExtractor):
    """Run prompt-driven schema add extraction steps."""

    def __init__(
        self,
        *,
        llm_client: LLMClient | None,
        prompt_set: AddPromptSet,
        entity_manager: EntitySchemaProvider,
        enable_schema_selection: bool,
    ) -> None:
        self.llm_client = llm_client
        self.prompt_set = prompt_set
        self.entity_manager = entity_manager
        self.enable_schema_selection = enable_schema_selection
        self.normalizer = SchemaExtractionNormalizer(entity_manager=entity_manager)

    async def select_schema(
        self,
        conversation_text: str,
        full_schema: list[dict[str, Any]],
        *,
        prompt_set: AddPromptSet | None = None,
    ) -> list[dict[str, Any]]:
        if not self.enable_schema_selection:
            return full_schema
        try:
            return await self._select_schema(conversation_text, full_schema, prompt_set=prompt_set)
        except Exception:
            logger.warning("schema selection failed; using full schema", exc_info=True)
            return full_schema

    async def _select_schema(
        self,
        conversation_text: str,
        full_schema: list[dict[str, Any]],
        *,
        prompt_set: AddPromptSet | None = None,
    ) -> list[dict[str, Any]]:
        prompts = prompt_set or self.prompt_set
        schema_summary = format_schema_summary(full_schema)
        prompt = prompts.schema_selection_for_generation.format(
            dialogue_text=conversation_text[:2000],
            entity_schema=schema_summary,
        )
        response = await resolve_llm_client(self.llm_client).chat(
            task="memory.add.schema_selection",
            messages=[{"role": "user", "content": prompt}],
            format_parser=parse_json_object,
        )
        selected = response.parsed.get("selected_entities", []) if isinstance(response.parsed, dict) else []
        filtered = build_filtered_schema(full_schema, selected)
        return filtered or full_schema

    async def generate_memory(
        self,
        *,
        entity_schema: list[dict[str, Any]],
        reference_entities: list[Any] | None = None,
        dialogue_timestamp: str,
        conversation_text: str,
        prompt_set: AddPromptSet | None = None,
        entity_manager: Any = None,
    ) -> dict[str, Any]:
        """Generate entities and edges in a single call, deciding new vs update.

        Recalled reference entities are injected into the prompt so the model can
        mark an extracted entity as ``update`` (with ``merge_target``) or ``new``,
        and emit edges that reference either new or existing entity names.
        """
        prompts = prompt_set or self.prompt_set
        reference_entities = reference_entities or []
        reference_names = {getattr(entity, "entity_name", "") for entity in reference_entities}

        prompt = (
            prompts.entity_generation.replace("{entity_schema}", str(entity_schema))
            .replace("{reference_entities}", format_reference_entities(reference_entities))
            .replace("{dialogue_timestamp}", dialogue_timestamp)
            .replace("{chat_chunk}", conversation_text)
        )

        last_memory: dict[str, Any] | None = None
        for _ in range(3):
            response = await resolve_llm_client(self.llm_client).chat(
                task="memory.add.entity_generation",
                messages=[{"role": "user", "content": prompt}],
                format_parser=parse_json_object,
            )
            raw_memory = response.parsed
            if not isinstance(raw_memory, dict):
                raw_memory = {"entities": [], "edges": []}
            last_memory = raw_memory

            validation_error = self.validate_memory(
                raw_memory,
                entity_manager=entity_manager,
                reference_entity_names=reference_names,
            )
            if not validation_error and has_unique_entity_names(raw_memory):
                return raw_memory
            prompt += (
                "\nPrevious answer: "
                + json.dumps(raw_memory, ensure_ascii=False)
                + f"\nERROR: {validation_error or 'There are entities with duplicate names. Please merge them.'}"
            )
        return last_memory or {"entities": [], "edges": []}

    async def objectify_conversation(
        self,
        conversation_text: str,
        conversation_timestamp: str,
        *,
        prompt_set: AddPromptSet | None = None,
    ) -> str:
        prompts = prompt_set or self.prompt_set
        prompt = prompts.episode_objectify.replace("{conversation_text}", conversation_text).replace(
            "{conversation_timestamp}", conversation_timestamp
        )
        try:
            response = await resolve_llm_client(self.llm_client).chat(
                task="memory.add.episode_objectify",
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.content.strip()
        except Exception:
            logger.warning("episode objectify failed; using original conversation", exc_info=True)
            return conversation_text
        if "**OBJECTIVE DESCRIPTION:**" in content:
            content = content.split("**OBJECTIVE DESCRIPTION:**")[-1].strip()
        if len(content) < 20:
            return conversation_text
        return content

    async def generate_episode_entity(
        self,
        conversation_text: str,
        conversation_timestamp: str,
        max_fields: int,
        *,
        prompt_set: AddPromptSet | None = None,
    ) -> dict[str, Any]:
        """Generate the episode title, factual summary, and search fields in one call."""
        prompts = prompt_set or self.prompt_set
        prompt = (
            prompts.episode_entity.replace("{conversation_text}", conversation_text)
            .replace("{conversation_timestamp}", conversation_timestamp)
            .replace("{max_fields}", str(max_fields))
        )
        try:
            response = await resolve_llm_client(self.llm_client).chat(
                task="memory.add.episode_entity",
                messages=[{"role": "user", "content": prompt}],
                format_parser=parse_json_object,
            )
            parsed = response.parsed
            if isinstance(parsed, dict):
                return {
                    "title": str(parsed.get("title") or "").strip(),
                    "content": str(parsed.get("content") or "").strip(),
                    "search_fields": [str(field) for field in (parsed.get("search_fields") or []) if field],
                }
        except Exception:
            logger.warning("episode entity generation failed; using fallback", exc_info=True)
        return self._fallback_episode_entity(conversation_text)

    def _fallback_episode_entity(self, conversation_text: str) -> dict[str, Any]:
        first_line = conversation_text.splitlines()[0] if conversation_text else ""
        return {
            "title": first_line[:80] or "Episode",
            "content": conversation_text,
            "search_fields": [],
        }

    def schema_for_generation(self, *, entity_manager: Any = None) -> list[dict[str, Any]]:
        em = entity_manager or self.entity_manager
        schema = copy.deepcopy(em.get_all_dicts())
        return strip_for_generation(schema)

    def prepare_raw_memory(self, raw_memory: dict[str, Any], dialogue_timestamp: str) -> dict[str, Any]:
        return self.normalizer.normalize(raw_memory, dialogue_timestamp)

    def validate_memory(
        self,
        raw_memory: dict[str, Any],
        *,
        entity_manager: Any = None,
        reference_entity_names: set[str] | None = None,
    ) -> str | None:
        return self.normalizer.validate(
            raw_memory,
            entity_manager=entity_manager,
            reference_entity_names=reference_entity_names,
        )
