"""Stable ID generation for entities, sources, and other domain objects.

Produces deterministic UUID5 identifiers from business keys so the same
logical entity or source always maps to the same ID across pipelines.
"""

from __future__ import annotations

from uuid import NAMESPACE_URL, uuid5

from ..typing import Entity, MemoryRequestContext, SourceRef


def generate_source_id(source_ref: SourceRef, context: MemoryRequestContext) -> SourceRef:
    """Return source_ref with a stable source_id derived from context + source metadata.

    If source_id is already set, returns the ref unchanged.
    """
    if source_ref.source_id:
        return source_ref
    key = "|".join(
        [
            context.project_id,
            context.request_id,
            source_ref.source_type,
            source_ref.message_id or "",
            source_ref.file_path or "",
            source_ref.uri or "",
            str(source_ref.metadata.get("message_index", "")),
        ]
    )
    return source_ref.model_copy(update={"source_id": str(uuid5(NAMESPACE_URL, key))})


def generate_entity_id(project_id: str, entity: Entity) -> str:
    """Return a stable entity ID derived from project_id + entity type + canonical name."""
    name = entity.canonical_name or entity.name
    key = f"{project_id}:{entity.entity_type or 'entity'}:{name}"
    return str(uuid5(NAMESPACE_URL, key))


def generate_memory_id(project_id: str, request_id: str, content_hash: str) -> str:
    """Return a stable memory ID derived from project + namespace + content hash.

    Combined with ``content_hash`` this ensures the same request produces the
    same memory ID, so Qdrant upsert naturally deduplicates repeated writes
    within that request.
    """
    key = f"{project_id}:{request_id}:{content_hash}"
    return str(uuid5(NAMESPACE_URL, key))


def generate_experience_id(project_id: str, canonical_content: str) -> str:
    """Return a deterministic experience-memory ID independent of the add request.

    Unlike ``generate_memory_id`` the key deliberately excludes ``request_id`` so
    the exact same (normalized) experience text maps to the same memory node
    across trajectory imports. Cross-import reuse of semantically-similar
    experiences is primarily handled by recall + LLM dedup, not by this ID.
    """
    key = f"{project_id}:experience:{canonical_content}"
    return str(uuid5(NAMESPACE_URL, key))


def generate_trajectory_source_id(project_id: str, message_index: int, content_hash: str) -> str:
    """Return a deterministic trajectory-source node ID.

    Re-importing the same trajectory message must reuse the same Source node, so
    the key is derived from message index + content hash and deliberately omits
    the ad-hoc request ID used by ``generate_source_id``.
    """
    key = f"{project_id}:trajectory:source:{message_index}:{content_hash}"
    return str(uuid5(NAMESPACE_URL, key))
