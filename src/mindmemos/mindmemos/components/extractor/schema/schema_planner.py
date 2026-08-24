"""Write-plan planner for schema add extraction results."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from ....llm import EmbedClient, LLMClient
from ....logging import get_logger
from ....prompts import AddPromptSet
from ....typing import (
    EntityView,
    EntityWrite,
    FieldCondition,
    GraphRelationship,
    MemoryAddEventItem,
    MemoryDbWritePlan,
    MemoryRequestContext,
    MemoryWrite,
    SearchFilter,
)
from ...memory_modeling.schema import Edge, get_entity_manager
from ...text import SparseVectorEncoder, TextPreprocessor
from ._runtime_clients import resolve_embed_client, resolve_llm_client
from ._schema_utils import (
    base_entity_name,
    base_metadata,
    dedupe_entity_relationships,
    edge_relationships,
    format_candidate_episodes,
    merge_description,
    parse_json_object,
    property_relationships,
    resolve_duplicate_name,
    schema_memory_type,
)
from ._schema_write_plan import SchemaWritePlanBuilder

logger = get_logger(__name__)
ProgressReporter = Callable[[str, str, int | None, dict[str, Any] | None], Awaitable[None]]


async def _report_progress(
    progress: ProgressReporter | None,
    stage: str,
    message: str,
    percent: int | None = None,
    data: dict[str, Any] | None = None,
) -> None:
    if progress is not None:
        await progress(stage, message, percent, data)


class SchemaAddPlanner:
    """Build write plans for schema add extracted entities using rule-based fusion."""

    def __init__(
        self,
        *,
        llm_client: LLMClient | None,
        embed_client: EmbedClient | None,
        db_reader: Any,
        prompt_set: AddPromptSet,
        entity_recall_top_k: int,
        episode_edge_top_k: int,
        text_preprocessor: TextPreprocessor,
        sparse_encoder: SparseVectorEncoder,
        max_entities_per_conversation: int = 200,
        max_properties_per_entity: int = 15,
        description_rewrite_threshold: int = 1000,
        description_max_chars: int = 2000,
        reference_description_max_chars: int = 500,
    ) -> None:
        self.llm_client = llm_client
        self.embed_client = embed_client
        self.db_reader = db_reader
        self.prompt_set = prompt_set
        self.entity_recall_top_k = entity_recall_top_k
        self.episode_edge_top_k = episode_edge_top_k
        self.max_entities_per_conversation = max_entities_per_conversation
        self.max_properties_per_entity = max_properties_per_entity
        self.description_rewrite_threshold = description_rewrite_threshold
        self.description_max_chars = description_max_chars
        self._text_preprocessor = text_preprocessor
        self._sparse_encoder = sparse_encoder
        self._write_plan_builder = SchemaWritePlanBuilder(
            text_preprocessor=self._text_preprocessor,
            sparse_encoder=self._sparse_encoder,
            embed_texts=self._embed_texts,
            entity_description_max_chars=reference_description_max_chars,
        )

    async def recall_reference_entities(
        self,
        *,
        conversation_text: str,
        context: MemoryRequestContext,
    ) -> list[EntityView]:
        """Recall non-episode entities close to the raw conversation text.

        Replaces the previous per-entity candidate recall: a single dense search
        over the whole conversation supplies the reference entities injected into
        the entity-generation prompt and used for rule-based merge decisions.
        """
        query_text = conversation_text[:4000]
        vectors = await self._embed_texts("memory.add.entity", [query_text])
        query_vector = vectors[0] if vectors else []
        if not query_vector:
            return []

        must: list[FieldCondition] = []
        if context.user_id:
            must.append(FieldCondition(field="user_id", op="match", value=context.user_id))
        filters = SearchFilter(
            must=must,
            must_not=[FieldCondition(field="entity_type", op="match", value="episodes")],
        )
        result = await self.db_reader.search_entities_dense(
            context,
            query=query_text,
            query_vector=query_vector,
            filters=filters,
            limit=self.entity_recall_top_k,
        )
        return [hit.entity for hit in result.hits if hit.entity is not None]

    async def plan_episode_edges(
        self,
        *,
        episode_name: str,
        episode_description: str,
        context: MemoryRequestContext,
        prompt_set: AddPromptSet | None = None,
    ) -> list[tuple[EntityView, str]]:
        """Plan edges between the new episode and historical episodes (二.1.c).

        Returns ``(target_episode, relation)`` pairs. The episode EntityWrite is not
        built yet, so graph relationships are materialized later in
        :meth:`build_write_plan`.
        """
        if self.episode_edge_top_k <= 0:
            return []

        query_text = (episode_description or episode_name)[:4000]
        vectors = await self._embed_texts("memory.add.entity", [query_text])
        query_vector = vectors[0] if vectors else []
        if not query_vector:
            return []

        episode_must: list[FieldCondition] = [
            FieldCondition(field="entity_type", op="match", value="episodes"),
        ]
        if context.user_id:
            episode_must.append(FieldCondition(field="user_id", op="match", value=context.user_id))
        try:
            result = await self.db_reader.search_entities_dense(
                context,
                query=query_text,
                query_vector=query_vector,
                filters=SearchFilter(must=episode_must),
                limit=self.episode_edge_top_k,
            )
            candidates = [hit.entity for hit in result.hits if hit.entity and hit.entity.entity_type == "episodes"]
        except Exception:
            logger.warning("episode edge candidate search failed", exc_info=True)
            return []

        if not candidates:
            return []

        prompts = prompt_set or self.prompt_set
        prompt = (
            prompts.episode_edge.replace("{new_episode_name}", episode_name)
            .replace("{new_episode_description}", episode_description or "")
            .replace("{candidate_episodes}", format_candidate_episodes(candidates))
        )
        try:
            response = await resolve_llm_client(self.llm_client).chat(
                task="memory.add.episode_edge",
                messages=[{"role": "user", "content": prompt}],
                format_parser=parse_json_object,
            )
            edges = response.parsed if isinstance(response.parsed, list) else []
        except Exception:
            logger.warning("episode edge LLM failed; no episode edges created", exc_info=True)
            return []

        candidate_by_id = {candidate.entity_id: candidate for candidate in candidates}
        planned: list[tuple[EntityView, str]] = []
        for edge in edges:
            if not isinstance(edge, dict):
                continue
            target = candidate_by_id.get(str(edge.get("target_episode_id") or ""))
            if target is None:
                continue
            planned.append((target, str(edge.get("relation") or "related_to")))
        return planned

    async def build_write_plan(
        self,
        *,
        raw_entities: list[dict[str, Any]],
        raw_edges: list[dict[str, Any]],
        episode_entity: dict[str, Any],
        reference_entities: list[EntityView],
        episode_edges: list[tuple[EntityView, str]],
        context: MemoryRequestContext,
        request_metadata: dict[str, Any],
        created_at: datetime,
        progress: ProgressReporter | None = None,
    ) -> tuple[MemoryDbWritePlan, list[MemoryAddEventItem]]:
        """Build a complete database write plan from extracted schema entities."""
        await _report_progress(
            progress,
            "memory_planning",
            "Planning memory structure.",
            60,
            {"message_i18n": {"zh": "正在整理记忆结构", "en": "Planning memory structure"}},
        )

        reference_by_name: dict[str, EntityView] = {ref.entity_name: ref for ref in reference_entities}

        memories: list[MemoryWrite] = []
        entities: list[EntityWrite] = []
        relationships: list[GraphRelationship] = []
        events: list[MemoryAddEventItem] = []
        entity_by_name: dict[str, EntityWrite] = {}

        raw_entity_list = list(raw_entities)
        if len(raw_entity_list) > self.max_entities_per_conversation:
            logger.warning(
                "entity_count_exceeds_limit",
                count=len(raw_entity_list),
                limit=self.max_entities_per_conversation,
            )
            raw_entity_list = raw_entity_list[: self.max_entities_per_conversation]

        for entity in raw_entity_list:
            original_name = str(entity.get("name") or "")
            entity_write = self._resolve_entity_write(
                entity,
                reference_by_name=reference_by_name,
                context=context,
                created_at=created_at,
                request_metadata=request_metadata,
            )
            entities.append(entity_write)
            entity_by_name[entity_write.entity_name] = entity_write
            if original_name and original_name != entity_write.entity_name:
                entity_by_name[original_name] = entity_write

        episode_write = self._new_entity_write(
            episode_entity,
            context=context,
            created_at=created_at,
            request_metadata=request_metadata,
        )
        entities.append(episode_write)
        entity_by_name[episode_entity["name"]] = episode_write

        # Compress rule-merged descriptions that outgrew the threshold before any
        # embedding text, payload, or event content is derived from them.
        await self._rewrite_oversized_descriptions(entities)

        await _report_progress(
            progress,
            "memory_planning",
            "Processing memory properties.",
            78,
            {"message_i18n": {"zh": "正在处理记忆属性", "en": "Processing memory properties"}},
        )
        for entity_write in entities:
            raw_properties = self._properties_for_entity(raw_entity_list, episode_entity, entity_write)
            prop_memories = self._build_property_memories(
                entity_write=entity_write,
                properties=raw_properties,
                context=context,
                created_at=created_at,
                request_metadata=request_metadata,
            )
            for memory in prop_memories:
                memories.append(memory)
                relationships.extend(property_relationships(context.project_id, entity_write.entity_id, memory))
            if prop_memories:
                events.append(
                    MemoryAddEventItem(
                        operation="add",
                        content=_format_entity_add_content(entity_write, prop_memories),
                        memory_id=entity_write.entity_id,
                        mem_type=schema_memory_type(entity_write.entity_type),
                        related_memory_ids=[memory.memory_id for memory in prop_memories],
                    )
                )

        # Entity-to-entity edges may reference new entities or recalled reference
        # entities. Combine both into one name -> DTO map for resolution.
        dto_by_name: dict[str, EntityView | EntityWrite] = dict(reference_by_name)
        dto_by_name.update(entity_by_name)
        relationships.extend(edge_relationships(raw_edges, dto_by_name, context.project_id))

        # Episode-to-episode edges planned during 二.1.c.
        for target, relation in episode_edges:
            relationship = Edge.from_entity_dtos(
                episode_write,
                target,
                description=relation,
            ).to_graph_relationship(project_id=context.project_id)
            relationship.metadata["edge_source"] = "episode_edge_prompt"
            relationships.append(relationship)

        relationships = dedupe_entity_relationships(relationships)
        await _report_progress(
            progress,
            "embedding",
            "Generating memory vectors.",
            86,
            {"message_i18n": {"zh": "正在生成记忆向量", "en": "Generating memory vectors"}},
        )
        plan = await self._write_plan_builder.build(
            memories=memories,
            entities=entities,
            relationships=relationships,
            project_id=context.project_id,
            entity_context_memories=memories,
        )
        return plan, events

    def _properties_for_entity(
        self,
        raw_entity_list: list[dict[str, Any]],
        episode_entity: dict[str, Any],
        entity_write: EntityWrite,
    ) -> list[dict[str, Any]]:
        if entity_write.entity_type == "episodes":
            return episode_entity.get("properties", [])
        # An alias-update entity stores its own raw name in latest_raw_entity_name;
        # match that first so a same-named create entity earlier in the list cannot
        # steal its properties.
        latest_raw_name = entity_write.metadata.get("latest_raw_entity_name")
        for entity in raw_entity_list:
            if latest_raw_name and entity.get("name") == latest_raw_name:
                raw_properties = entity.get("properties", [])
                if len(raw_properties) > self.max_properties_per_entity:
                    raw_properties = raw_properties[: self.max_properties_per_entity]
                return raw_properties
        for entity in raw_entity_list:
            if entity.get("name") == entity_write.entity_name:
                raw_properties = entity.get("properties", [])
                if len(raw_properties) > self.max_properties_per_entity:
                    raw_properties = raw_properties[: self.max_properties_per_entity]
                return raw_properties
        return []

    def _resolve_entity_write(
        self,
        entity: dict[str, Any],
        *,
        reference_by_name: dict[str, EntityView],
        context: MemoryRequestContext,
        created_at: datetime,
        request_metadata: dict[str, Any],
    ) -> EntityWrite:
        """Resolve whether an extracted entity creates or updates an entity record."""
        operation = entity.get("operation")
        merge_target = str(entity.get("merge_target") or "")

        if operation == "update" and merge_target:
            target = reference_by_name.get(merge_target)
            if target is not None:
                return self._entity_write_from_view(target, entity, context=context, created_at=created_at)

        return self._resolve_create(entity, reference_by_name, context, created_at, request_metadata)

    def _resolve_create(
        self,
        entity: dict[str, Any],
        reference_by_name: dict[str, EntityView],
        context: MemoryRequestContext,
        created_at: datetime,
        request_metadata: dict[str, Any],
    ) -> EntityWrite:
        """Create a new entity, with rule-based same-name disambiguation."""
        entity_name = str(entity.get("name") or "")
        entity_type = entity.get("entity_type", "")

        existing = self._reference_by_base_name(entity_name, entity_type, reference_by_name)
        if existing is not None:
            resolution = resolve_duplicate_name(entity, existing)
            if resolution["action"] == "merge":
                logger.info(
                    "duplicate name in references, converting to UPDATE",
                    entity_name=entity_name,
                    target_id=existing.entity_id,
                )
                return self._entity_write_from_view(existing, entity, context=context, created_at=created_at)
            entity["name"] = resolution["new_name"]

        return self._new_entity_write(
            entity,
            context=context,
            created_at=created_at,
            request_metadata=request_metadata,
        )

    def _reference_by_base_name(
        self,
        name: str,
        entity_type: str | None,
        reference_by_name: dict[str, EntityView],
    ) -> EntityView | None:
        base = base_entity_name(name)
        for ref in reference_by_name.values():
            if base_entity_name(ref.entity_name) == base and (not entity_type or ref.entity_type == entity_type):
                return ref
        return None

    async def _rewrite_oversized_descriptions(self, entities: list[EntityWrite]) -> None:
        """Compress rule-merged entity descriptions that crossed the rewrite threshold.

        The v2 flow merges entity descriptions by plain concatenation (no extra
        LLM call per update). Left unchecked a hot entity's description grows
        without bound, inflating the stored payload, the embedding text, and
        every later reference-entity prompt. Past the threshold one LLM call
        rewrites the accumulated description into a concise single one; on
        failure the hard-capped concatenation from ``merge_description`` stays.
        """
        for entity in entities:
            description = entity.description or ""
            if entity.entity_type == "episodes" or len(description) <= self.description_rewrite_threshold:
                continue
            rewritten = await self._rewrite_description(entity, description)
            if rewritten:
                entity.description = rewritten
                entity.metadata = {**dict(entity.metadata or {}), "description_rewritten": True}

    async def _rewrite_description(self, entity: EntityWrite, description: str) -> str | None:
        prompt = (
            self.prompt_set.entity_description_rewrite.replace("{entity_name}", entity.entity_name)
            .replace("{entity_type}", entity.entity_type or "")
            .replace("{char_limit}", str(self.description_max_chars))
            .replace("{current_description}", description)
        )
        try:
            response = await resolve_llm_client(self.llm_client).chat(
                task="memory.add.description_rewrite",
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception:
            logger.warning("description rewrite LLM failed; keeping capped concatenation", exc_info=True)
            return None
        content = response.content.strip()
        if "<description>" in content:
            content = content.split("<description>")[1]
        if "</description>" in content:
            content = content.split("</description>")[0]
        return content.strip()[: self.description_max_chars] or None

    def _entity_write_from_view(
        self,
        target: EntityView,
        new_entity: dict[str, Any],
        *,
        context: MemoryRequestContext,
        created_at: datetime,
    ) -> EntityWrite:
        """Build an update EntityWrite with a rule-based description merge."""
        em = get_entity_manager(project_id=context.project_id)
        description = merge_description(
            target.description or "", new_entity.get("description") or "", max_chars=self.description_max_chars
        )
        metadata = dict(target.metadata)
        metadata.update(
            {
                "add_algorithm": "schema_add",
                "merge_action": "update",
                "latest_raw_entity_name": new_entity.get("name"),
                "record_time": new_entity.get("record_time"),
            }
        )
        return EntityWrite(
            entity_id=target.entity_id,
            account_id=target.account_id or context.account_id,
            project_id=context.project_id,
            api_key_uuid=target.api_key_uuid or context.api_key_uuid,
            user_id=target.user_id or context.user_id,
            app_id=target.app_id or context.app_id,
            session_id=target.session_id or context.session_id,
            agent_id=target.agent_id or context.agent_id,
            request_id=context.request_id,
            entity_name=target.entity_name,
            entity_type=target.entity_type or new_entity.get("entity_type"),
            description=description,
            schema_version=em.file_path.name,
            metadata=metadata,
            created_at=target.created_at or created_at,
            update_at=created_at,
        )

    def _new_entity_write(
        self,
        entity: dict[str, Any],
        *,
        context: MemoryRequestContext,
        created_at: datetime,
        request_metadata: dict[str, Any],
    ) -> EntityWrite:
        """Build an EntityWrite for a newly created entity."""
        em = get_entity_manager(project_id=context.project_id)
        entity_id = str(uuid4())
        metadata = base_metadata(request_metadata)
        metadata.update(
            {
                "add_algorithm": "schema_add",
                "record_time": entity.get("record_time"),
                "raw_entity_name": entity.get("name"),
            }
        )
        if entity.get("search_fields"):
            metadata["search_fields"] = list(entity["search_fields"])
        return EntityWrite(
            entity_id=entity_id,
            account_id=context.account_id,
            project_id=context.project_id,
            api_key_uuid=context.api_key_uuid,
            user_id=context.user_id,
            app_id=context.app_id,
            session_id=context.session_id,
            agent_id=context.agent_id,
            request_id=context.request_id,
            entity_name=str(entity.get("name") or entity_id),
            entity_type=entity.get("entity_type"),
            description=entity.get("description"),
            schema_version=em.file_path.name,
            metadata=metadata,
            created_at=created_at,
        )

    def _build_property_memories(
        self,
        *,
        entity_write: EntityWrite,
        properties: list[dict[str, Any]],
        context: MemoryRequestContext,
        created_at: datetime,
        request_metadata: dict[str, Any],
    ) -> list[MemoryWrite]:
        """Convert extracted properties into memory writes (no LLM merge/delete)."""
        return [
            self._memory_from_property(
                entity_write=entity_write,
                prop=prop,
                context=context,
                created_at=created_at,
                request_metadata=request_metadata,
            )
            for prop in properties
            if prop.get("operation") != "delete"
        ]

    def _memory_from_property(
        self,
        *,
        entity_write: EntityWrite,
        prop: dict[str, Any],
        context: MemoryRequestContext,
        created_at: datetime,
        request_metadata: dict[str, Any],
    ) -> MemoryWrite:
        """Build a MemoryWrite DTO from one extracted property dictionary."""
        memory_id = str(uuid4())
        property_name = str(prop.get("property_name") or "default_property")
        metadata = base_metadata(request_metadata)
        metadata.update(
            {
                "add_algorithm": "schema_add",
                "property_time": prop.get("time"),
                "property_operation": prop.get("operation", "set"),
                "entity_name": entity_write.entity_name,
            }
        )
        mem_type = schema_memory_type(entity_write.entity_type, property_name)
        return MemoryWrite(
            memory_id=memory_id,
            account_id=context.account_id,
            project_id=context.project_id,
            api_key_uuid=context.api_key_uuid,
            user_id=context.user_id,
            app_id=context.app_id,
            session_id=context.session_id,
            agent_id=context.agent_id,
            request_id=context.request_id,
            content=str(prop.get("value") or ""),
            mem_type=mem_type,
            mem_extract_type="schema",
            mem_extract_version="schema_add",
            metadata=metadata,
            validate_from=_validate_from_property_time(prop.get("time")),
            created_at=created_at,
            parent_ids=[],
            root_id=[memory_id],
            property_name=property_name,
            entity_id=entity_write.entity_id,
            entity_type=entity_write.entity_type,
        )

    async def _embed_texts(self, task: str, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = await resolve_embed_client(self.embed_client).embed(task=task, text=texts)
        return response.embeddings


def _validate_from_property_time(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _format_entity_add_content(entity_write: EntityWrite, prop_memories: list[MemoryWrite]) -> str:
    """Format entity properties into a prompt matching schema search output."""
    block = f"Entity: {entity_write.entity_name} (Type: {entity_write.entity_type})"
    for memory in prop_memories:
        prop_name = memory.property_name or memory.mem_type or ""
        if memory.content:
            block += f"\n   Property '{prop_name}': {memory.content}"
    return block
