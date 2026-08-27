"""Memory-port table declarations for persistence v2."""

from __future__ import annotations

from ...infra.vector_store import FieldType, IndexKind, IndexSpec, TableSpec, VectorFieldSpec
from .base import TableDefinition, column, required, schema_version_column

MEMORY_TABLE = "memory_item_v2"
ENTITY_TABLE = "entity_item_v2"
SOURCE_TABLE = "source_item_v2"


def memory_table_definitions(
    *,
    vector_dimensions: int,
    sparse_hash_dim: int,
) -> tuple[TableDefinition, ...]:
    """Return the memory, entity, and source tables owned by the memory port."""

    semantic = VectorFieldSpec(name="semantic", dimensions=vector_dimensions)
    bm25 = VectorFieldSpec(name="bm25", dimensions=sparse_hash_dim, distance="dot", sparse=True)
    return (
        TableDefinition(
            port="memory",
            spec=TableSpec(
                name=MEMORY_TABLE,
                primary_key="memory_id",
                fields=(
                    schema_version_column(),
                    required("memory_id", FieldType.UUID),
                    column("request_id", FieldType.UUID),
                    required("content", FieldType.TEXT),
                    column("mem_type", FieldType.TEXT, nullable=False, default="fact"),
                    column("memory_mode", FieldType.TEXT, nullable=False, default="vanilla"),
                    column("mem_extract_type", FieldType.TEXT, nullable=False, default="vanilla"),
                    required("mem_extract_version", FieldType.TEXT),
                    column("status", FieldType.TEXT, nullable=False, default="active"),
                    column("validate_from", FieldType.DATETIME),
                    column("validate_to", FieldType.DATETIME),
                    column("reinforcement_count", FieldType.INTEGER, nullable=False, default=0),
                    required("created_at", FieldType.DATETIME),
                    column("update_at", FieldType.DATETIME),
                    column("status_changed_at", FieldType.DATETIME),
                    column("parent_ids", FieldType.UUID_ARRAY, nullable=False, default=()),
                    column("root_id", FieldType.UUID_ARRAY, nullable=False, default=()),
                    column("property_name", FieldType.TEXT),
                    column("entity_id", FieldType.UUID),
                    column("entity_type", FieldType.TEXT),
                    column("metadata", FieldType.JSON, nullable=False, default={}),
                ),
                indexes=(
                    IndexSpec(name="memory_v2_status_idx", fields=("status",)),
                    IndexSpec(name="memory_v2_mode_idx", fields=("memory_mode",)),
                    IndexSpec(name="memory_v2_created_idx", fields=("created_at",)),
                    IndexSpec(name="memory_v2_content_fts_idx", fields=("content",), kind=IndexKind.FULL_TEXT),
                    IndexSpec(name="memory_v2_entity_idx", fields=("entity_id",)),
                ),
                vectors=(semantic, bm25),
            ),
        ),
        TableDefinition(
            port="memory",
            spec=TableSpec(
                name=ENTITY_TABLE,
                primary_key="entity_id",
                fields=(
                    schema_version_column(),
                    required("entity_id", FieldType.UUID),
                    column("request_id", FieldType.UUID),
                    required("entity_name", FieldType.TEXT),
                    column("entity_type", FieldType.TEXT),
                    column("description", FieldType.TEXT),
                    column("status", FieldType.TEXT, nullable=False, default="active"),
                    required("created_at", FieldType.DATETIME),
                    column("update_at", FieldType.DATETIME),
                    column("status_changed_at", FieldType.DATETIME),
                    column("parent_ids", FieldType.UUID_ARRAY, nullable=False, default=()),
                    column("root_id", FieldType.UUID_ARRAY, nullable=False, default=()),
                    column("metadata", FieldType.JSON, nullable=False, default={}),
                ),
                indexes=(
                    IndexSpec(name="entity_v2_name_idx", fields=("entity_name",), kind=IndexKind.HASH),
                    IndexSpec(name="entity_v2_status_idx", fields=("status",)),
                ),
                vectors=(semantic, bm25),
            ),
        ),
        TableDefinition(
            port="memory",
            spec=TableSpec(
                name=SOURCE_TABLE,
                primary_key="source_id",
                fields=(
                    schema_version_column(),
                    required("source_id", FieldType.UUID),
                    column("request_id", FieldType.UUID),
                    required("source_type", FieldType.TEXT),
                    required("file_path", FieldType.TEXT),
                    required("file_name", FieldType.TEXT),
                    column("is_parsed", FieldType.BOOLEAN, nullable=False, default=False),
                    column("parsed_content_path", FieldType.TEXT),
                    column("status", FieldType.TEXT, nullable=False, default="active"),
                    required("created_at", FieldType.DATETIME),
                    column("update_at", FieldType.DATETIME),
                    column("parsed_at", FieldType.DATETIME),
                    column("parsed_cost", FieldType.FLOAT),
                    column("status_changed_at", FieldType.DATETIME),
                    column("parent_ids", FieldType.UUID_ARRAY, nullable=False, default=()),
                    column("root_id", FieldType.UUID_ARRAY, nullable=False, default=()),
                    column("metadata", FieldType.JSON, nullable=False, default={}),
                ),
                indexes=(
                    IndexSpec(name="source_v2_type_idx", fields=("source_type",)),
                    IndexSpec(name="source_v2_status_idx", fields=("status",)),
                ),
                vectors=(semantic,),
            ),
        ),
    )


__all__ = [
    "ENTITY_TABLE",
    "MEMORY_TABLE",
    "SOURCE_TABLE",
    "memory_table_definitions",
]
