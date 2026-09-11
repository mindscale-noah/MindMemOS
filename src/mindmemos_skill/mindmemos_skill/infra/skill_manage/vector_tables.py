"""Vector-store table catalog for ``skill_manage`` embeddings.

The embedding table is bootstrapped separately from the structured tables via
``bootstrap_vector_store()``. It declares every field explicitly — metadata is
not flattened ad hoc — and the dense ``embedding`` vector's physical dimension
is fixed at bootstrap time from the compiled embedding model config.
"""

from __future__ import annotations

from ..vector_store import (
    FieldSpec,
    FieldType,
    IndexSpec,
    TableRegistry,
    TableSpec,
    VectorFieldSpec,
)

SKILL_MANAGE_EMBEDDINGS = "skill_manage_embeddings"
EMBEDDING_VECTOR_NAME = "embedding"


def build_skill_manage_vector_tables(dimension: int) -> TableRegistry:
    if dimension <= 0:
        raise ValueError("embedding dimension must be positive")

    table = TableSpec(
        name=SKILL_MANAGE_EMBEDDINGS,
        primary_key="vector_record_id",
        fields=(
            FieldSpec(name="vector_record_id", field_type=FieldType.TEXT, nullable=False),
            FieldSpec(name="graph_node_id", field_type=FieldType.TEXT, nullable=False),
            FieldSpec(name="skill_id", field_type=FieldType.TEXT),
            FieldSpec(name="version_id", field_type=FieldType.TEXT),
            FieldSpec(name="node_type", field_type=FieldType.TEXT, nullable=False),
            FieldSpec(name="generation_id", field_type=FieldType.TEXT, nullable=False),
            FieldSpec(name="content_hash", field_type=FieldType.TEXT),
            FieldSpec(name="embedding_model", field_type=FieldType.TEXT, nullable=False),
            FieldSpec(name="embedding_profile_version", field_type=FieldType.TEXT),
            FieldSpec(name="metadata", field_type=FieldType.JSON, nullable=False, default={}),
        ),
        indexes=(
            IndexSpec(name="skill_manage_embeddings_node_idx", fields=("graph_node_id",)),
            IndexSpec(name="skill_manage_embeddings_generation_idx", fields=("generation_id",)),
        ),
        vectors=(VectorFieldSpec(name=EMBEDDING_VECTOR_NAME, dimensions=dimension, distance="cosine"),),
        scope_scoped=True,
    )
    registry = TableRegistry((table,))
    registry.freeze()
    return registry


__all__ = [
    "EMBEDDING_VECTOR_NAME",
    "SKILL_MANAGE_EMBEDDINGS",
    "build_skill_manage_vector_tables",
]
