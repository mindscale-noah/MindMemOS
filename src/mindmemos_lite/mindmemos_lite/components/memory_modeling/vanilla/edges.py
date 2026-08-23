"""Graph relationship builders for Memory / Entity / Source edges.

Produces ``GraphRelationship`` DTOs for the four standard relationship types
used across add, update, merge, and dreaming pipelines. Each builder is a
pure function with no side effects.
"""

from __future__ import annotations

from ....typing import (
    REL_EXTRACTED_FROM,
    REL_HAS_PROPERTY_MEMORY,
    REL_MENTIONED_IN_SOURCE,
    REL_MENTIONS,
    REL_RELATES_TO,
    REL_TASK_EXPERIENCE,
    Entity,
    GraphNodeRef,
    GraphRelationship,
    MemoryRequestContext,
    MemoryWrite,
    SourceRef,
)
from ...chunker import SourceAwareSegment


def build_mentions_edge(
    memory_id: str,
    entity_id: str,
    entity: Entity,
    context: MemoryRequestContext,
) -> GraphRelationship:
    """Build a Memory --MENTIONS--> Entity relationship."""
    return GraphRelationship(
        source=GraphNodeRef(kind="Memory", project_id=context.project_id, node_id=memory_id),
        target=GraphNodeRef(kind="Entity", project_id=context.project_id, node_id=entity_id),
        rel_type=REL_MENTIONS,
        project_id=context.project_id,
        mention_count=1,
        metadata={
            "entity_name": entity.name,
            "canonical_name": entity.canonical_name,
            "entity_type": entity.entity_type,
            "confidence": entity.confidence,
            "extractor": entity.extractor,
            "offsets": entity.offsets,
        },
    )


def build_extracted_from_edge(
    memory_id: str,
    source_ref: SourceRef,
    context: MemoryRequestContext,
    segment: SourceAwareSegment,
) -> GraphRelationship:
    """Build a Memory --EXTRACTED_FROM--> Source relationship."""
    if source_ref.source_id is None:
        raise ValueError("source_ref.source_id is required before writing source relationship")
    return GraphRelationship(
        source=GraphNodeRef(kind="Memory", project_id=context.project_id, node_id=memory_id),
        target=GraphNodeRef(kind="Source", project_id=context.project_id, node_id=source_ref.source_id),
        rel_type=REL_EXTRACTED_FROM,
        project_id=context.project_id,
        extraction_position={
            "message_index": segment.message_index,
            "start_offset": segment.start_offset,
            "end_offset": segment.end_offset,
        },
        metadata={
            "source_type": source_ref.source_type,
            "role": segment.role,
            "timestamp": segment.timestamp,
        },
    )


def build_mentioned_in_source_edge(
    entity_id: str,
    source_ref: SourceRef,
    entity: Entity,
    context: MemoryRequestContext,
) -> GraphRelationship:
    """Build an Entity --MENTIONED_IN_SOURCE--> Source relationship."""
    if source_ref.source_id is None:
        raise ValueError("source_ref.source_id is required before writing source relationship")
    return GraphRelationship(
        source=GraphNodeRef(kind="Entity", project_id=context.project_id, node_id=entity_id),
        target=GraphNodeRef(kind="Source", project_id=context.project_id, node_id=source_ref.source_id),
        rel_type=REL_MENTIONED_IN_SOURCE,
        project_id=context.project_id,
        mention_count=1,
        metadata={
            "entity_name": entity.name,
            "canonical_name": entity.canonical_name,
            "entity_type": entity.entity_type,
            "confidence": entity.confidence,
            "extractor": entity.extractor,
        },
    )


def build_relates_to_edge(
    memory_id: str,
    related_memory_id: str,
    context: MemoryRequestContext,
    *,
    edge_type: str = "related_to",
    source: str = "add_related_recall",
) -> GraphRelationship:
    """Build a Memory --RELATES_TO--> Memory relationship."""
    return GraphRelationship(
        source=GraphNodeRef(kind="Memory", project_id=context.project_id, node_id=memory_id),
        target=GraphNodeRef(kind="Memory", project_id=context.project_id, node_id=related_memory_id),
        rel_type=REL_RELATES_TO,
        project_id=context.project_id,
        edge_type=edge_type,
        metadata={"source": source},
    )


def build_task_experience_edge(
    task_entity_id: str,
    experience_memory_id: str,
    task_text: str,
    context: MemoryRequestContext,
) -> GraphRelationship:
    """Build a task Entity --TASK_EXPERIENCE--> experience Memory relationship.

    ``edge_type`` and ``entity_id`` feed the stable edge identity so re-importing
    the same (task, experience) pair upserts the same graph edge instead of
    duplicating it.
    """
    return GraphRelationship(
        source=GraphNodeRef(kind="Entity", project_id=context.project_id, node_id=task_entity_id),
        target=GraphNodeRef(kind="Memory", project_id=context.project_id, node_id=experience_memory_id),
        rel_type=REL_TASK_EXPERIENCE,
        project_id=context.project_id,
        edge_type="task_experience",
        entity_id=task_entity_id,
        metadata={"task_text": task_text},
    )


def property_relationships(
    project_id: str,
    entity_id: str,
    memory: MemoryWrite,
) -> list[GraphRelationship]:
    """Build the schema-compatible bidirectional property-memory edges."""

    entity_ref = GraphNodeRef(kind="Entity", project_id=project_id, node_id=entity_id)
    memory_ref = GraphNodeRef(kind="Memory", project_id=project_id, node_id=memory.memory_id)
    return [
        GraphRelationship(
            source=entity_ref,
            target=memory_ref,
            rel_type=REL_HAS_PROPERTY_MEMORY,
            project_id=project_id,
            property_name=memory.property_name,
            entity_id=entity_id,
        ),
        GraphRelationship(
            source=memory_ref,
            target=entity_ref,
            rel_type=REL_MENTIONS,
            project_id=project_id,
            entity_id=entity_id,
            mention_count=1,
        ),
    ]
