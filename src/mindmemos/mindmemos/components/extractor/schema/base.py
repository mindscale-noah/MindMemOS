"""Schema add component protocols."""

from __future__ import annotations

from typing import Any, Protocol

from ....typing import EntityWrite, GraphRelationship, MemoryDbWritePlan, MemoryWrite


class SchemaEpisodeExtractor(Protocol):
    """Protocol for prompt-driven schema episode extraction."""

    async def objectify_conversation(self, conversation_text: str, conversation_timestamp: str) -> str: ...

    async def generate_episode_entity(
        self, conversation_text: str, conversation_timestamp: str, max_fields: int
    ) -> dict[str, Any]: ...


class SchemaExtractionNormalizerProtocol(Protocol):
    """Protocol for schema extraction result normalization."""

    def normalize(self, raw_memory: dict[str, Any], dialogue_timestamp: str) -> dict[str, Any]: ...

    def validate(
        self,
        raw_memory: dict[str, Any],
        *,
        entity_manager: Any = None,
        reference_entity_names: set[str] | None = None,
    ) -> str | None: ...


class SchemaWritePlanBuilderProtocol(Protocol):
    """Protocol for turning schema-add DTOs into a memory DB write plan."""

    async def build(
        self,
        *,
        memories: list[MemoryWrite],
        entities: list[EntityWrite],
        relationships: list[GraphRelationship],
        project_id: str,
        entity_context_memories: list[MemoryWrite] | None = None,
    ) -> MemoryDbWritePlan: ...
