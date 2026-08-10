from __future__ import annotations

import pytest
from mindmemos_skill.infra.database import (
    FieldSpec,
    FieldType,
    IndexSpec,
    TableRegistry,
    TableSpec,
)
from mindmemos_skill.infra.vector_store import (
    FieldSpec as VectorFieldSpec,
)
from mindmemos_skill.infra.vector_store import (
    FieldType as VectorFieldType,
)
from mindmemos_skill.infra.vector_store import (
    IndexSpec as VectorIndexSpec,
)
from mindmemos_skill.infra.vector_store import (
    TableRegistry as VectorTableRegistry,
)
from mindmemos_skill.infra.vector_store import (
    TableSpec as VectorTableSpec,
)


def _database_table(name: str, *indexes: IndexSpec) -> TableSpec:
    return TableSpec(
        name=name,
        primary_key="record_id",
        fields=(FieldSpec(name="record_id", field_type=FieldType.TEXT, nullable=False),),
        indexes=indexes,
    )


def _vector_table(name: str, *indexes: VectorIndexSpec) -> VectorTableSpec:
    return VectorTableSpec(
        name=name,
        primary_key="record_id",
        fields=(VectorFieldSpec(name="record_id", field_type=VectorFieldType.TEXT, nullable=False),),
        indexes=indexes,
    )


def test_database_table_spec_rejects_duplicate_index_names() -> None:
    duplicate = IndexSpec(name="shared_uq", fields=("record_id",), unique=True)

    with pytest.raises(ValueError, match="contains duplicate indexes: \\['shared_uq'\\]"):
        _database_table("records", duplicate, duplicate)


def test_database_table_registry_rejects_cross_table_duplicate_index_names_atomically() -> None:
    shared = IndexSpec(name="shared_uq", fields=("record_id",), unique=True)
    first = _database_table("first_records", shared)
    second = _database_table("second_records", shared)
    registry = TableRegistry((first,))

    with pytest.raises(
        ValueError,
        match="index 'shared_uq' is already registered for table 'first_records'; "
        "table 'second_records' cannot reuse it",
    ):
        registry.register(second)

    assert registry.specs == (first,)


def test_vector_table_spec_rejects_duplicate_index_names() -> None:
    duplicate = VectorIndexSpec(name="shared_uq", fields=("record_id",), unique=True)

    with pytest.raises(ValueError, match="contains duplicate indexes: \\['shared_uq'\\]"):
        _vector_table("records", duplicate, duplicate)


def test_vector_table_registry_rejects_cross_table_duplicate_index_names_atomically() -> None:
    shared = VectorIndexSpec(name="shared_uq", fields=("record_id",), unique=True)
    first = _vector_table("first_records", shared)
    second = _vector_table("second_records", shared)
    registry = VectorTableRegistry((first,))

    with pytest.raises(
        ValueError,
        match="index 'shared_uq' is already registered for table 'first_records'; "
        "table 'second_records' cannot reuse it",
    ):
        registry.register(second)

    assert registry.specs == (first,)
