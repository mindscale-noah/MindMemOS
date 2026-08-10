from __future__ import annotations

from inspect import isabstract
from typing import Any

import pytest
from mindmemos_skill.infra.vector_store import (
    BackendConfig,
    BackendRegistry,
    DatabaseScope,
    FieldSpec,
    FieldType,
    FilterGroup,
    Predicate,
    Record,
    ScopedVectorStore,
    SparseVector,
    TableRegistry,
    TableSpec,
    VectorFieldSpec,
    VectorHit,
    VectorQuery,
    VectorValue,
)
from mindmemos_skill.infra.vector_store.vector_store_impl.pgvector import (
    PgVectorBackend,
    PgVectorOptions,
    _decode_cursor,
    _dense_literal,
    _encode_cursor,
    _parse_dense_literal,
    _parse_sparse_literal,
    _sparse_literal,
    create_pgvector_backend,
    register_pgvector_backend,
)


def _tables(*, sparse_dimensions: int = 2_000_000) -> TableRegistry:
    registry = TableRegistry(
        (
            TableSpec(
                name="memory",
                primary_key="memory_id",
                fields=(
                    FieldSpec(name="memory_id", field_type=FieldType.TEXT, nullable=False),
                    FieldSpec(name="project_id", field_type=FieldType.TEXT, nullable=False),
                    FieldSpec(name="content", field_type=FieldType.TEXT, nullable=False),
                    FieldSpec(name="metadata", field_type=FieldType.JSON),
                ),
                vectors=(
                    VectorFieldSpec(name="semantic", dimensions=3),
                    VectorFieldSpec(name="bm25", dimensions=sparse_dimensions, distance="dot", sparse=True),
                ),
            ),
        )
    )
    registry.freeze()
    return registry


def _backend(*, sparse_dimensions: int = 2_000_000) -> PgVectorBackend:
    return PgVectorBackend(
        options=PgVectorOptions(dsn="postgresql://unused"),
        tables=_tables(sparse_dimensions=sparse_dimensions),
        pool=object(),
    )


class _AsyncContext:
    def __init__(self, value: Any) -> None:
        self._value = value

    async def __aenter__(self):
        return self._value

    async def __aexit__(self, _exc_type, _exc, _traceback) -> None:
        return None


class _EnsureCursor:
    def __init__(self, row: dict[str, Any] | None = None) -> None:
        self._row = row

    async def fetchone(self) -> dict[str, Any] | None:
        return self._row


class _EnsureConnection:
    def __init__(self) -> None:
        self.statements: list[str] = []

    def transaction(self) -> _AsyncContext:
        return _AsyncContext(self)

    async def execute(self, statement, _params=None) -> _EnsureCursor:
        rendered = statement.as_string() if hasattr(statement, "as_string") else statement
        self.statements.append(rendered)
        if "FROM pg_extension" in rendered:
            return _EnsureCursor({"extversion": "0.8.2"})
        return _EnsureCursor()


class _EnsurePool:
    def __init__(self) -> None:
        self.connection_value = _EnsureConnection()

    def connection(self) -> _AsyncContext:
        return _AsyncContext(self.connection_value)


def test_pgvector_options_require_a_dsn_and_reject_typos() -> None:
    with pytest.raises(ValueError, match="'dsn' is required"):
        PgVectorOptions.from_mapping({})
    with pytest.raises(ValueError, match="unknown pgvector backend options: pool_size"):
        PgVectorOptions.from_mapping({"dsn": "postgresql://unused", "pool_size": 2})


@pytest.mark.asyncio
async def test_python_ensure_schema_enables_extension_and_creates_schema() -> None:
    tables = TableRegistry()
    tables.freeze()
    pool = _EnsurePool()
    backend = PgVectorBackend(
        options=PgVectorOptions(dsn="postgresql://unused"),
        tables=tables,
        pool=pool,
    )

    await backend.ensure_schema(tables)

    assert pool.connection_value.statements[0] == "CREATE EXTENSION IF NOT EXISTS vector"
    assert 'CREATE SCHEMA IF NOT EXISTS "mindmemos"' in pool.connection_value.statements
    assert any(
        'CREATE TABLE IF NOT EXISTS "mindmemos"."__mindmemos_schema"' in statement
        for statement in pool.connection_value.statements
    )


def test_pgvector_factory_registers_capabilities_without_connecting() -> None:
    registry = BackendRegistry()
    register_pgvector_backend(registry)

    backend = registry.create(
        config=BackendConfig(provider="pgvector", options={"dsn": "postgresql://unused"}),
        tables=_tables(),
    )

    assert backend.name == "pgvector"
    assert backend.capabilities.dense_vector is True
    assert backend.capabilities.sparse_vector is True
    assert backend.capabilities.hybrid_search is True
    assert backend.capabilities.atomic_batch_write is True


def test_pgvector_ddl_uses_typed_columns_and_native_vector_types() -> None:
    backend = _backend()
    ddl = backend._table_ddl(backend._table("memory")).as_string()

    assert 'CREATE TABLE IF NOT EXISTS "mindmemos"."memory"' in ddl
    assert '"metadata" jsonb' in ddl
    assert '"semantic" vector(3)' in ddl
    assert '"bm25" sparsevec(2000000)' in ddl
    assert "PRIMARY KEY (_scope_key, _record_id)" in ddl


def test_pgvector_record_preparation_enforces_scope_identity_and_vector_dimensions() -> None:
    backend = _backend(sparse_dimensions=8)
    spec = backend._table("memory")
    record = Record(
        table="memory",
        record_id="memory-1",
        scope=DatabaseScope(project_id="project-a"),
        payload={
            "memory_id": "memory-1",
            "project_id": "project-a",
            "content": "coffee",
            "metadata": {"source": "chat"},
        },
        vectors=VectorValue(
            dense={"semantic": (1.0, 0.0, 0.0)},
            sparse={"bm25": SparseVector(indices=(0, 7), values=(0.5, 2.0))},
        ),
    )

    values = backend._prepare_record(spec, record)

    assert values[0] == '{"project_id":"project-a"}'
    assert values[1].obj == {"project_id": "project-a"}
    assert values[-2:] == ("[1.0,0.0,0.0]", "{1:0.5,8:2.0}/8")

    mismatched = Record(
        table="memory",
        record_id="memory-1",
        scope=DatabaseScope(project_id="project-a"),
        payload={
            "memory_id": "memory-1",
            "project_id": "project-b",
            "content": "coffee",
        },
    )
    with pytest.raises(ValueError, match="differs from the record scope"):
        backend._prepare_record(spec, mismatched)


def test_pgvector_sparse_text_boundary_converts_zero_based_indices() -> None:
    value = SparseVector(indices=(5, 0, 3), values=(0.5, 1.0, -2.0))

    encoded = _sparse_literal(value, 8)

    assert encoded == "{1:1.0,4:-2.0,6:0.5}/8"
    assert _parse_sparse_literal(encoded) == SparseVector(indices=(0, 3, 5), values=(1.0, -2.0, 0.5))
    assert _parse_dense_literal(_dense_literal((1.0, -0.25, 3.0))) == (1.0, -0.25, 3.0)


def test_pgvector_filter_compiler_supports_groups_and_nested_json() -> None:
    backend = _backend()
    spec = backend._table("memory")
    expression = FilterGroup(
        operator="and",
        clauses=(
            Predicate(field="project_id", op="eq", value="project-a"),
            Predicate(field="metadata.business.priority", op="gte", value=2),
            Predicate(field="content", op="icontains", value="coffee"),
        ),
    )

    statement, params = backend._compile_filter(spec, expression)

    rendered = statement.as_string()
    assert '"project_id" IS NOT DISTINCT FROM %s' in rendered
    assert '(("metadata" #>> %s))::numeric >= %s' in rendered
    assert '"content" ILIKE %s' in rendered
    assert params == ["project-a", ["business", "priority"], 2, "%coffee%"]


def test_pgvector_filter_compiler_distinguishes_empty_from_null() -> None:
    backend = _backend()
    spec = backend._table("memory")

    empty_statement, empty_params = backend._compile_filter(
        spec,
        Predicate(field="content", op="is_empty"),
    )
    null_statement, null_params = backend._compile_filter(
        spec,
        Predicate(field="content", op="is_null", value=True),
    )

    assert "IN ('', '[]', '{}', 'null')" in empty_statement.as_string()
    assert empty_params == []
    assert null_statement.as_string() == '"content" IS NULL'
    assert null_params == []


def test_pgvector_cursor_is_table_bound_and_rejects_invalid_values() -> None:
    cursor = _encode_cursor("memory", 50)

    assert _decode_cursor(cursor, "memory") == 50
    with pytest.raises(ValueError, match="invalid record query cursor"):
        _decode_cursor(cursor, "entity")
    with pytest.raises(ValueError, match="invalid record query cursor"):
        _decode_cursor("not-base64", "memory")


def test_pgvector_rrf_fuses_dense_and_sparse_ranks_deterministically() -> None:
    backend = _backend()

    def hit(record_id: str, score: float, source: str) -> VectorHit:
        return VectorHit(
            record=Record(
                table="memory",
                record_id=record_id,
                scope=DatabaseScope(project_id="project-a"),
                payload={},
            ),
            score=score,
            source=source,
        )

    query: Any = type("Query", (), {"score_threshold": None, "top_k": 2})()
    fused = backend._fuse_rrf(
        query,
        [hit("a", 0.9, "dense"), hit("b", 0.8, "dense")],
        [hit("b", 4.0, "sparse"), hit("a", 3.0, "sparse")],
    )

    assert [item.record.record_id for item in fused] == ["a", "b"]
    assert all(item.source == "rrf" for item in fused)
    assert fused[0].debug["ranks"] == {"dense": 1, "sparse": 2}
    assert fused[0].score == pytest.approx(1 / 2 + 1 / 3)


@pytest.mark.asyncio
async def test_pgvector_hybrid_search_honors_independent_channel_limits(monkeypatch) -> None:
    backend = _backend()
    observed: dict[str, int] = {}

    async def record_search(query, _vector, _literal, *, source):
        observed[source] = query.top_k
        return []

    monkeypatch.setattr(backend, "_search_one", record_search)

    result = await backend.search_vectors(
        VectorQuery(
            table="memory",
            scope=DatabaseScope(project_id="project-a"),
            vector_name="semantic",
            dense_vector=(1.0, 0.0, 0.0),
            sparse_indices=(1,),
            sparse_values=(2.0,),
            mode="hybrid",
            top_k=5,
            dense_limit=30,
            sparse_limit=40,
        )
    )

    assert result == []
    assert observed == {"pgvector_dense": 30, "pgvector_sparse": 40}


def test_pgvector_factory_is_compatible_with_backend_registry_signature() -> None:
    backend = create_pgvector_backend({"dsn": "postgresql://unused"}, _tables())

    assert isinstance(backend, PgVectorBackend)
    assert isinstance(backend, ScopedVectorStore)
    assert not isabstract(PgVectorBackend)


@pytest.mark.asyncio
async def test_pgvector_requires_an_extension_version_with_sparsevec_support() -> None:
    class Cursor:
        def __init__(self, row: dict[str, str] | None) -> None:
            self._row = row

        async def fetchone(self) -> dict[str, str] | None:
            return self._row

    class Connection:
        def __init__(self, row: dict[str, str] | None) -> None:
            self._row = row

        async def execute(self, statement: str) -> Cursor:
            return Cursor(self._row)

    backend = _backend()

    await backend._validate_extension(Connection({"extversion": "0.7.0"}))
    with pytest.raises(RuntimeError, match="pgvector >= 0.7.0"):
        await backend._validate_extension(Connection({"extversion": "0.6.2"}))
    with pytest.raises(RuntimeError, match="is not installed"):
        await backend._validate_extension(Connection(None))
