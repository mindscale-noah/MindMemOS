"""Vector-store embedding table catalog (AC-P3-001 vector portion)."""

from __future__ import annotations

import pytest
from mindmemos_skill.infra.skill_manage.vector_tables import (
    EMBEDDING_VECTOR_NAME,
    SKILL_MANAGE_EMBEDDINGS,
    build_skill_manage_vector_tables,
)


def test_vector_tables_declare_embedding_fields_explicitly() -> None:
    registry = build_skill_manage_vector_tables(dimension=3072)
    table = registry.get(SKILL_MANAGE_EMBEDDINGS)

    field_names = {field.name for field in table.fields}
    assert {
        "vector_record_id",
        "graph_node_id",
        "skill_id",
        "version_id",
        "node_type",
        "generation_id",
        "content_hash",
        "embedding_model",
        "embedding_profile_version",
        "metadata",
    } <= field_names
    assert table.scope_scoped is True
    assert table.primary_key == "vector_record_id"


def test_vector_table_dimension_is_fixed_at_bootstrap() -> None:
    registry = build_skill_manage_vector_tables(dimension=1536)
    table = registry.get(SKILL_MANAGE_EMBEDDINGS)

    assert len(table.vectors) == 1
    vector = table.vectors[0]
    assert vector.name == EMBEDDING_VECTOR_NAME
    assert vector.dimensions == 1536
    assert vector.distance == "cosine"


def test_vector_table_rejects_non_positive_dimension() -> None:
    with pytest.raises(ValueError, match="dimension"):
        build_skill_manage_vector_tables(0)
