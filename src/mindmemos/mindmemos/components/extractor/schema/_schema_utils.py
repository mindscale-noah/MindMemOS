"""Pure helpers for schema add extraction components."""

from __future__ import annotations

import copy
import json
import re
from typing import Any

from ....typing import (
    REL_HAS_PROPERTY_MEMORY,
    REL_MENTIONS,
    REL_RELATED_TO,
    EntityView,
    EntityWrite,
    GraphNodeRef,
    GraphRelationship,
    MemoryType,
    MemoryView,
    MemoryWrite,
)
from ...memory_modeling.schema import Edge

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_EPISODIC_SCHEMA_TYPES = frozenset({"episode", "episodes", "episodic"})
_PROFILE_SCHEMA_TYPES = frozenset({"user", "person"})
_EXPERIENCE_SCHEMA_KEYS = frozenset({"task_experience"})


def parse_json_object(content: str) -> Any:
    """Parse a JSON object or array from LLM output."""

    text = content.strip()
    fenced = _JSON_FENCE_RE.search(text)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = min((idx for idx in [text.find("{"), text.find("[")] if idx >= 0), default=-1)
        end = max(text.rfind("}"), text.rfind("]"))
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def schema_memory_type(entity_type: str | None, property_name: str | None = None) -> MemoryType:
    """Map arbitrary schema labels to the standard displayed memory types."""

    normalized_entity_type = _normalize_schema_key(entity_type)
    normalized_property_name = _normalize_schema_key(property_name)
    if normalized_entity_type in _EPISODIC_SCHEMA_TYPES:
        return "episodic"
    if normalized_entity_type in _EXPERIENCE_SCHEMA_KEYS or normalized_property_name in _EXPERIENCE_SCHEMA_KEYS:
        return "experience"
    if normalized_entity_type in _PROFILE_SCHEMA_TYPES:
        return "profile"
    return "fact"


def strip_for_generation(schema: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the schema entries used for entity generation.

    The ``episodes`` pseudo-entity is dropped because the episode entity is built
    by the dedicated episode-entity branch rather than the generic entity prompt.
    """
    return [item for item in schema if item.get("entity_type") != "episodes"]


def format_schema_summary(entity_schema: list[dict[str, Any]]) -> str:
    """Format an entity schema as a concise Markdown summary for prompts."""
    lines: list[str] = []
    for entity in entity_schema:
        entity_type = entity.get("entity_type")
        if entity_type == "episodes":
            continue
        lines.append(f"## {entity_type}: {entity.get('entity_description', '')}")
        dynamic = entity.get("dynamic_property", {})
        if isinstance(dynamic, dict):
            for prop_name, prop_def in dynamic.items():
                if prop_name == "default_property":
                    continue
                desc = prop_def.get("desc", "") if isinstance(prop_def, dict) else str(prop_def)
                lines.append(f"  - {prop_name}: {desc[:80]}")
        lines.append("")
    return "\n".join(lines)


def build_filtered_schema(full_schema: list[dict[str, Any]], selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build a filtered schema from the LLM schema-selection result."""
    selected_map: dict[str, list[str]] = {}
    for item in selected:
        entity_type = item.get("entity_type")
        if entity_type:
            selected_map[entity_type] = list(item.get("relevant_properties") or ["all"])
    filtered: list[dict[str, Any]] = []
    for entity in full_schema:
        entity_type = entity.get("entity_type")
        if entity_type == "episodes" or entity_type not in selected_map:
            continue
        props_filter = selected_map[entity_type]
        entity_copy = copy.deepcopy(entity)
        if props_filter != ["all"]:
            original_dynamic = entity_copy.get("dynamic_property", {})
            if isinstance(original_dynamic, dict):
                new_dynamic = {}
                if "default_property" in original_dynamic:
                    new_dynamic["default_property"] = original_dynamic["default_property"]
                for prop_name in props_filter:
                    if prop_name in original_dynamic:
                        new_dynamic[prop_name] = original_dynamic[prop_name]
                entity_copy["dynamic_property"] = new_dynamic
        filtered.append(entity_copy)
    return filtered


def has_unique_entity_names(raw_memory: dict[str, Any]) -> bool:
    """Return whether extracted entity names are unique."""
    names = [entity.get("name") for entity in raw_memory.get("entities", []) if entity.get("name")]
    return len(names) == len(set(names))


def build_episode_entity(
    *,
    objectified_content: str,
    title: str,
    content: str,
    dialogue_date: str,
    search_fields: list[str] | None = None,
) -> dict[str, Any]:
    """Build the raw episode entity dictionary used by write planning."""
    return {
        "name": title or "Episode",
        "entity_type": "episodes",
        "description": content,
        "record_time": dialogue_date,
        "search_fields": search_fields or [],
        "properties": [
            {
                "property_name": "input_messages",
                "value": objectified_content,
                "time": dialogue_date,
                "operation": "set",
            }
        ],
    }


def entity_write_embedding_text(entity: EntityWrite, memories: list[MemoryWrite] | None = None) -> str:
    """Build the canonical entity indexing text from the final entity write payload."""

    property_text = " ".join(memory.content for memory in (memories or [])[:5])
    search_field_text = " ".join(
        str(field) for field in (entity.metadata or {}).get("search_fields", [])[:5] if isinstance(field, str)
    )
    return " ".join(
        part
        for part in [
            entity.entity_name,
            entity.entity_type or "",
            entity.description or "",
            property_text,
            search_field_text,
        ]
        if part
    )


def memory_embedding_text(memory: MemoryWrite | MemoryView) -> str:
    """Build the canonical property-memory indexing text."""

    entity_name = str((memory.metadata or {}).get("entity_name") or "")
    property_name = str(memory.property_name or "")
    return f"{entity_name}:{property_name}:{memory.content}"


def base_metadata(request_metadata: dict[str, Any]) -> dict[str, Any]:
    """Build the base metadata wrapper for request metadata."""
    return {"request_metadata": dict(request_metadata)}


def merge_description(old: str | None, new: str | None) -> str | None:
    """Merge an existing description with a newly extracted description."""
    if not old:
        return new
    if not new or new in old:
        return old
    return f"{old}\n{new}"


def format_candidate_episodes(candidates: list[EntityView]) -> str:
    """Format recalled episode candidates for the episode-edge prompt."""
    return "\n".join(
        f"- id: {candidate.entity_id}, name: {candidate.entity_name}, description: {candidate.description or ''}"
        for candidate in candidates
    )


def format_reference_entities(entities: list[EntityView]) -> str:
    """Format recalled reference entities for the entity-generation prompt."""
    if not entities:
        return "None (no existing entities in memory)."
    return "\n".join(
        f"- name: {entity.entity_name}, entity_type: {entity.entity_type or ''}, "
        f"description: {entity.description or ''}"
        for entity in entities
    )


def base_entity_name(name: str) -> str:
    return name.split("(", 1)[0].strip().lower()


def resolve_duplicate_name(new_entity: dict[str, Any], existing: EntityView) -> dict[str, Any]:
    """Resolve entity name conflicts by merge or rename."""
    entity_type = new_entity.get("entity_type", "")

    if entity_type == "episodes":
        new_name = f"{new_entity.get('name', '')} ({new_entity.get('record_time', 'latest')})"
        return {"action": "rename", "new_name": new_name}

    if entity_type == existing.entity_type:
        return {"action": "merge"}

    new_name = f"{new_entity.get('name', '')} ({entity_type})"
    return {"action": "rename", "new_name": new_name}


def property_relationships(project_id: str, entity_id: str, memory: MemoryWrite) -> list[GraphRelationship]:
    """Build bidirectional relationships for a property memory."""
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


def edge_relationships(
    raw_edges: list[dict[str, Any]],
    entity_by_name: dict[str, EntityView | EntityWrite],
    project_id: str,
) -> list[GraphRelationship]:
    """Build entity-to-entity relationships from raw edges.

    ``entity_by_name`` maps a referenced entity name to either a newly created
    ``EntityWrite`` or a recalled reference ``EntityView``, so edges may connect
    two new entities, a new entity and a reference entity, or two reference
    entities.
    """
    relationships: list[GraphRelationship] = []
    seen_pairs: set[tuple[str, str]] = set()
    for edge in raw_edges:
        source = entity_by_name.get(str(edge.get("link_entity1_name") or ""))
        target = entity_by_name.get(str(edge.get("link_entity2_name") or ""))
        if source is None or target is None:
            continue
        pair = tuple(sorted((source.entity_id, target.entity_id)))
        if pair[0] == pair[1] or pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        relation = str(edge.get("link_description") or "related_to")
        relationships.append(
            Edge.from_entity_dtos(source, target, description=relation).to_graph_relationship(project_id=project_id)
        )
    return relationships


def dedupe_entity_relationships(relationships: list[GraphRelationship]) -> list[GraphRelationship]:
    """Keep one entity-to-entity edge per unordered entity pair."""

    deduped: list[GraphRelationship] = []
    seen_pairs: set[tuple[str, str]] = set()
    for relationship in relationships:
        if (
            relationship.rel_type == REL_RELATED_TO
            and relationship.source.kind == "Entity"
            and relationship.target.kind == "Entity"
        ):
            pair = tuple(sorted((relationship.source.node_id, relationship.target.node_id)))
            if pair[0] == pair[1] or pair in seen_pairs:
                continue
            seen_pairs.add(pair)
        deduped.append(relationship)
    return deduped


def _normalize_schema_key(value: str | None) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
