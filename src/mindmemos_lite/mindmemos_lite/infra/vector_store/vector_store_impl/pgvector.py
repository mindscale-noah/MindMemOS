"""PostgreSQL + pgvector implementation of the vector-store backend contract.

The adapter stores every logical table as one PostgreSQL table. Payload fields
remain typed columns, arbitrary database scope is kept in JSONB, and named
dense/sparse vectors use pgvector's ``vector`` and ``sparsevec`` types.

Vector values are sent and read through pgvector's text representation. This
keeps driver-specific values inside this module and avoids exposing pgvector
objects through the public vector-store models.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Any
from uuid import UUID

from psycopg import sql
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from ..models import (
    BackendCapabilities,
    FieldSpec,
    FieldType,
    FilterExpression,
    FilterGroup,
    IndexKind,
    Predicate,
    Record,
    RecordQuery,
    Sort,
    SparseVector,
    TableSpec,
    VectorFieldSpec,
    VectorHit,
    VectorQuery,
    VectorValue,
)
from ..registry import BackendRegistry, TableRegistry
from ..scope import DatabaseScope
from ..vector_store import ScopedVectorStore

_RESERVED_COLUMNS = frozenset({"_scope_key", "_scope", "_record_id", "_schema_fingerprint"})
_SCHEMA_TABLE = "__mindmemos_schema"
_MAX_VECTOR_DIMENSIONS = 16_000
_MAX_SPARSE_NON_ZERO = 1_000


@dataclass(frozen=True, slots=True, kw_only=True)
class PgVectorOptions:
    """Connection and retrieval settings accepted by the pgvector factory.

    ``hybrid_prefetch_factor`` is used only when a hybrid ``VectorQuery`` does
    not provide explicit dense/sparse channel limits.
    """

    dsn: str
    schema: str = "mindmemos"
    min_pool_size: int = 1
    max_pool_size: int = 10
    pool_timeout: float = 30.0
    create_extension: bool = True
    create_schema: bool = True
    hybrid_prefetch_factor: int = 4
    rrf_k: int = 2
    dense_weight: float = 1.0
    sparse_weight: float = 1.0

    @classmethod
    def from_mapping(cls, options: Mapping[str, Any]) -> "PgVectorOptions":
        allowed = set(cls.__dataclass_fields__)
        unknown = set(options) - allowed
        if unknown:
            raise ValueError(f"unknown pgvector backend options: {', '.join(sorted(unknown))}")
        if not options.get("dsn"):
            raise ValueError("pgvector backend option 'dsn' is required")
        result = cls(**dict(options))
        if not result.schema:
            raise ValueError("pgvector backend option 'schema' must not be empty")
        if result.min_pool_size < 0 or result.max_pool_size <= 0:
            raise ValueError("pgvector pool sizes must be non-negative and max_pool_size must be positive")
        if result.min_pool_size > result.max_pool_size:
            raise ValueError("pgvector min_pool_size must not exceed max_pool_size")
        if result.pool_timeout <= 0:
            raise ValueError("pgvector pool_timeout must be positive")
        if result.hybrid_prefetch_factor <= 0 or result.rrf_k <= 0:
            raise ValueError("pgvector hybrid_prefetch_factor and rrf_k must be positive")
        if result.dense_weight < 0 or result.sparse_weight < 0:
            raise ValueError("pgvector RRF weights must be non-negative")
        if result.dense_weight == result.sparse_weight == 0:
            raise ValueError("at least one pgvector RRF weight must be positive")
        return result


class PgVectorBackend(ScopedVectorStore):
    """Concrete PostgreSQL backend using the pgvector extension."""

    _capabilities = BackendCapabilities(
        dense_vector=True,
        sparse_vector=True,
        hybrid_search=True,
        metadata_filtering=True,
        batch_record_io=True,
        atomic_batch_write=True,
        max_vector_dimensions=_MAX_VECTOR_DIMENSIONS,
    )

    def __init__(
        self,
        *,
        options: PgVectorOptions,
        tables: TableRegistry,
        pool: Any | None = None,
    ) -> None:
        self._options = options
        self._tables = {spec.name: spec for spec in tables.specs}
        self._validate_registry()
        self._pool = pool or AsyncConnectionPool(
            conninfo=options.dsn,
            min_size=options.min_pool_size,
            max_size=options.max_pool_size,
            timeout=options.pool_timeout,
            open=False,
            kwargs={"row_factory": dict_row},
        )
        self._pool_open = pool is not None
        self._closed = False
        self._open_lock = asyncio.Lock()

    @property
    def name(self) -> str:
        return "pgvector"

    @property
    def capabilities(self) -> BackendCapabilities:
        return self._capabilities

    async def ensure_schema(self, tables: TableRegistry) -> None:
        """Create extension/schema/tables/indexes and reject schema drift."""

        requested = {spec.name: spec for spec in tables.specs}
        if requested != self._tables:
            raise ValueError("ensure_schema registry differs from the registry used to construct the pgvector backend")
        await self._ensure_open()
        async with self._pool.connection() as connection:
            async with connection.transaction():
                if self._options.create_extension:
                    await connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
                await self._validate_extension(connection)
                if self._options.create_schema:
                    await connection.execute(
                        sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(self._options.schema))
                    )
                await connection.execute(self._schema_metadata_ddl())
                for spec in self._tables.values():
                    await self._ensure_table(connection, spec)

    async def upsert_records(self, table: str, records: Sequence[Record]) -> None:
        if not records:
            return
        spec = self._table(table)
        prepared = [self._prepare_record(spec, record) for record in records]
        statement = self._upsert_statement(spec)
        await self._ensure_open()
        async with self._pool.connection() as connection:
            async with connection.transaction():
                async with connection.cursor() as cursor:
                    await cursor.executemany(statement, prepared)

    async def get_records(
        self,
        table: str,
        scope: DatabaseScope,
        record_ids: Sequence[str],
        *,
        with_vectors: bool = False,
    ) -> list[Record]:
        if not record_ids:
            return []
        spec = self._table(table)
        select = self._select_columns(spec, with_vectors=with_vectors)
        identity = self._exact_scope_clause(spec)
        statement = sql.SQL(
            "SELECT {select} FROM {table} WHERE {identity} AND _record_id = ANY(%s) "
            "ORDER BY array_position(%s::text[], _record_id)"
        ).format(select=select, table=self._qualified_table(spec), identity=identity)
        params: list[Any] = []
        if spec.scope_scoped:
            params.append(_scope_key(scope))
        params.extend((list(record_ids), list(record_ids)))
        rows = await self._fetchall(statement, params)
        return [self._row_to_record(spec, row, with_vectors=with_vectors) for row in rows]

    async def patch_record(
        self,
        table: str,
        scope: DatabaseScope,
        record_id: str,
        changes: Mapping[str, Any],
    ) -> None:
        if not changes:
            return
        spec = self._table(table)
        fields = {field.name: field for field in spec.fields}
        unknown = set(changes) - set(fields)
        if unknown:
            raise ValueError(f"unknown fields for table {table!r}: {', '.join(sorted(unknown))}")
        if spec.primary_key in changes:
            raise ValueError(f"cannot patch primary key {spec.primary_key!r}")
        assignments: list[sql.Composable] = []
        params: list[Any] = []
        for name, value in changes.items():
            field = fields[name]
            if value is None and not field.nullable:
                raise ValueError(f"field {name!r} on table {table!r} is not nullable")
            assignments.append(sql.SQL("{} = %s").format(sql.Identifier(name)))
            params.append(_adapt_field_value(field, value))
        identity = self._exact_scope_clause(spec)
        statement = sql.SQL("UPDATE {table} SET {assignments} WHERE {identity} AND _record_id = %s").format(
            table=self._qualified_table(spec),
            assignments=sql.SQL(", ").join(assignments),
            identity=identity,
        )
        if spec.scope_scoped:
            params.append(_scope_key(scope))
        params.append(record_id)
        await self._execute(statement, params)

    async def delete_records(self, table: str, scope: DatabaseScope, record_ids: Sequence[str]) -> None:
        if not record_ids:
            return
        spec = self._table(table)
        identity = self._exact_scope_clause(spec)
        statement = sql.SQL("DELETE FROM {table} WHERE {identity} AND _record_id = ANY(%s)").format(
            table=self._qualified_table(spec),
            identity=identity,
        )
        params: list[Any] = []
        if spec.scope_scoped:
            params.append(_scope_key(scope))
        params.append(list(record_ids))
        await self._execute(statement, params)

    async def query_records(self, table: str, query: RecordQuery) -> tuple[list[Record], str | None]:
        return await self._query_records(table, query, with_vectors=False)

    async def scroll(
        self,
        table: str,
        query: RecordQuery,
        *,
        with_vectors: bool = False,
    ) -> tuple[list[Record], str | None]:
        """Scroll records using the portable query and cursor contract."""

        return await self._query_records(table, query, with_vectors=with_vectors)

    async def _query_records(
        self,
        table: str,
        query: RecordQuery,
        *,
        with_vectors: bool,
    ) -> tuple[list[Record], str | None]:
        spec = self._table(table)
        offset = _decode_cursor(query.page.cursor, table)
        where, where_params = self._where_clause(spec, query.scope, query.filters)
        order, order_params = self._order_clause(spec, query.sort)
        statement = sql.SQL("SELECT {select} FROM {table} WHERE {where} ORDER BY {order} LIMIT %s OFFSET %s").format(
            select=self._select_columns(spec, with_vectors=with_vectors),
            table=self._qualified_table(spec),
            where=where,
            order=order,
        )
        limit = query.page.limit
        rows = await self._fetchall(statement, [*where_params, *order_params, limit + 1, offset])
        has_more = len(rows) > limit
        records = [self._row_to_record(spec, row, with_vectors=with_vectors) for row in rows[:limit]]
        cursor = _encode_cursor(table, offset + limit) if has_more else None
        return records, cursor

    async def search_vectors(self, query: VectorQuery) -> list[VectorHit]:
        spec = self._table(query.table)
        if query.mode == "dense":
            vector = self._vector_spec(spec, query.vector_name, sparse=False)
            self._validate_dense(vector, query.dense_vector or ())
            return await self._search_one(
                query,
                vector,
                _dense_literal(query.dense_vector or ()),
                source=vector.name,
            )
        if query.mode == "sparse":
            vector = self._sparse_query_spec(spec, query.vector_name)
            sparse = SparseVector(indices=query.sparse_indices or (), values=query.sparse_values or ())
            self._validate_sparse(vector, sparse)
            return await self._search_one(
                query,
                vector,
                _sparse_literal(sparse, vector.dimensions),
                source=vector.name,
            )

        dense_spec = self._vector_spec(spec, query.vector_name, sparse=False)
        sparse_spec = self._sparse_query_spec(spec, query.vector_name)
        dense_values = query.dense_vector or ()
        sparse_values = SparseVector(indices=query.sparse_indices or (), values=query.sparse_values or ())
        self._validate_dense(dense_spec, dense_values)
        self._validate_sparse(sparse_spec, sparse_values)
        default_candidate_count = max(query.top_k, query.top_k * self._options.hybrid_prefetch_factor)
        dense_candidate_query = VectorQuery(
            table=query.table,
            scope=query.scope,
            vector_name=query.vector_name,
            dense_vector=dense_values,
            sparse_indices=sparse_values.indices,
            sparse_values=sparse_values.values,
            mode="hybrid",
            filters=query.filters,
            top_k=query.dense_limit or default_candidate_count,
        )
        sparse_candidate_query = VectorQuery(
            table=query.table,
            scope=query.scope,
            vector_name=query.vector_name,
            dense_vector=dense_values,
            sparse_indices=sparse_values.indices,
            sparse_values=sparse_values.values,
            mode="hybrid",
            filters=query.filters,
            top_k=query.sparse_limit or default_candidate_count,
        )
        dense_hits, sparse_hits = await asyncio.gather(
            self._search_one(
                dense_candidate_query,
                dense_spec,
                _dense_literal(dense_values),
                source="pgvector_dense",
            ),
            self._search_one(
                sparse_candidate_query,
                sparse_spec,
                _sparse_literal(sparse_values, sparse_spec.dimensions),
                source="pgvector_sparse",
            ),
        )
        return self._fuse_rrf(query, dense_hits, sparse_hits)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._pool_open:
            await self._pool.close()

    async def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("pgvector backend is closed")
        if self._pool_open:
            return
        async with self._open_lock:
            if not self._pool_open:
                await self._pool.open()
                self._pool_open = True

    async def _execute(self, statement: sql.Composable, params: Sequence[Any]) -> None:
        await self._ensure_open()
        async with self._pool.connection() as connection:
            await connection.execute(statement, params)

    async def _fetchall(self, statement: sql.Composable, params: Sequence[Any]) -> list[dict[str, Any]]:
        await self._ensure_open()
        async with self._pool.connection() as connection:
            cursor = await connection.execute(statement, params)
            return list(await cursor.fetchall())

    async def _ensure_table(self, connection: Any, spec: TableSpec) -> None:
        fingerprint = _schema_fingerprint(spec)
        cursor = await connection.execute(
            sql.SQL("SELECT fingerprint FROM {}.{} WHERE logical_table = %s").format(
                sql.Identifier(self._options.schema),
                sql.Identifier(_SCHEMA_TABLE),
            ),
            (spec.name,),
        )
        existing = await cursor.fetchone()
        if existing is not None and existing["fingerprint"] != fingerprint:
            raise ValueError(
                f"pgvector schema drift for table {spec.name!r}; use an explicit database migration "
                "instead of changing TableSpec in place"
            )
        if existing is None:
            cursor = await connection.execute(
                "SELECT EXISTS ("
                "SELECT 1 FROM pg_catalog.pg_class AS c "
                "JOIN pg_catalog.pg_namespace AS n ON n.oid = c.relnamespace "
                "WHERE n.nspname = %s AND c.relname = %s"
                ") AS table_exists",
                (self._options.schema, spec.name),
            )
            unmanaged = await cursor.fetchone()
            if unmanaged is not None and unmanaged["table_exists"]:
                raise ValueError(
                    f"PostgreSQL table {self._options.schema}.{spec.name} exists without a vector-store "
                    "schema fingerprint; adopt it through an explicit migration"
                )
        await connection.execute(self._table_ddl(spec))
        for index in self._index_ddl(spec):
            await connection.execute(index)
        if existing is None:
            await connection.execute(
                sql.SQL("INSERT INTO {}.{} (logical_table, fingerprint) VALUES (%s, %s)").format(
                    sql.Identifier(self._options.schema),
                    sql.Identifier(_SCHEMA_TABLE),
                ),
                (spec.name, fingerprint),
            )

    async def _validate_extension(self, connection: Any) -> None:
        cursor = await connection.execute(
            "SELECT extversion FROM pg_extension WHERE extname = 'vector'",
        )
        row = await cursor.fetchone()
        if row is None:
            raise RuntimeError(
                "PostgreSQL extension 'vector' is not installed; install pgvector or set "
                "create_extension=true for an account allowed to create it"
            )
        parsed = [int(value) for value in re.findall(r"\d+", row["extversion"])[:3]]
        version = tuple([*parsed, 0, 0][:3])
        if version < (0, 7, 0):
            raise RuntimeError(f"pgvector >= 0.7.0 is required for sparsevec support; found {row['extversion']}")

    def _schema_metadata_ddl(self) -> sql.Composable:
        return sql.SQL(
            "CREATE TABLE IF NOT EXISTS {}.{} ("
            "logical_table text PRIMARY KEY, fingerprint text NOT NULL, "
            "created_at timestamptz NOT NULL DEFAULT now())"
        ).format(sql.Identifier(self._options.schema), sql.Identifier(_SCHEMA_TABLE))

    def _table_ddl(self, spec: TableSpec) -> sql.Composable:
        columns: list[sql.Composable] = [
            sql.SQL("_scope_key text NOT NULL"),
            sql.SQL("_scope jsonb NOT NULL"),
            sql.SQL("_record_id text NOT NULL"),
        ]
        for field in spec.fields:
            nullable = sql.SQL("") if field.nullable else sql.SQL(" NOT NULL")
            columns.append(
                sql.SQL("{} {}{}").format(
                    sql.Identifier(field.name),
                    sql.SQL(_POSTGRES_TYPES[field.field_type]),
                    nullable,
                )
            )
        for vector in spec.vectors:
            vector_type = "sparsevec" if vector.sparse else "vector"
            columns.append(
                sql.SQL("{} {}({})").format(
                    sql.Identifier(vector.name),
                    sql.SQL(vector_type),
                    sql.Literal(vector.dimensions),
                )
            )
        primary = (
            sql.SQL("PRIMARY KEY (_scope_key, _record_id)")
            if spec.scope_scoped
            else sql.SQL("PRIMARY KEY (_record_id)")
        )
        columns.append(primary)
        return sql.SQL("CREATE TABLE IF NOT EXISTS {table} ({columns})").format(
            table=self._qualified_table(spec),
            columns=sql.SQL(", ").join(columns),
        )

    def _index_ddl(self, spec: TableSpec) -> list[sql.Composable]:
        statements = [
            sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {} USING gin (_scope)").format(
                sql.Identifier(f"{spec.name}__scope_gin"),
                self._qualified_table(spec),
            )
        ]
        fields = {field.name for field in spec.fields}
        for index in spec.indexes:
            index_fields = [
                sql.Identifier(field) if field in fields else sql.SQL("_record_id") for field in index.fields
            ]
            if index.kind == IndexKind.FULL_TEXT:
                text = sql.SQL(" || ' ' || ").join(
                    sql.SQL("coalesce({}::text, '')").format(field) for field in index_fields
                )
                statements.append(
                    sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {} USING gin (to_tsvector('simple', {}))").format(
                        sql.Identifier(index.name),
                        self._qualified_table(spec),
                        text,
                    )
                )
                continue
            if index.kind == IndexKind.HASH:
                if index.unique:
                    raise ValueError("hash indexes do not support UNIQUE constraints")
                if len(index.fields) != 1:
                    raise ValueError("postgres hash indexes support exactly one column")
                statements.append(
                    sql.SQL("CREATE INDEX IF NOT EXISTS {name} ON {table} USING hash ({fields})").format(
                        name=sql.Identifier(index.name),
                        table=self._qualified_table(spec),
                        fields=sql.SQL(", ").join(index_fields),
                    )
                )
                continue
            if index.unique and spec.scope_scoped:
                index_fields.insert(0, sql.SQL("_scope_key"))
            unique = sql.SQL("UNIQUE ") if index.unique else sql.SQL("")
            statements.append(
                sql.SQL("CREATE {unique}INDEX IF NOT EXISTS {name} ON {table} ({fields})").format(
                    unique=unique,
                    name=sql.Identifier(index.name),
                    table=self._qualified_table(spec),
                    fields=sql.SQL(", ").join(index_fields),
                )
            )
        return statements

    def _upsert_statement(self, spec: TableSpec) -> sql.Composable:
        columns: list[sql.Composable] = [sql.SQL("_scope_key"), sql.SQL("_scope"), sql.SQL("_record_id")]
        values: list[sql.Composable] = [sql.SQL("%s"), sql.SQL("%s"), sql.SQL("%s")]
        for field in spec.fields:
            columns.append(sql.Identifier(field.name))
            values.append(sql.SQL("%s"))
        for vector in spec.vectors:
            columns.append(sql.Identifier(vector.name))
            vector_type = "sparsevec" if vector.sparse else "vector"
            values.append(sql.SQL("%s::{}").format(sql.SQL(vector_type)))
        conflict = sql.SQL("_scope_key, _record_id") if spec.scope_scoped else sql.SQL("_record_id")
        updates = [
            sql.SQL("{} = EXCLUDED.{}").format(column, column)
            for column in [
                sql.SQL("_scope"),
                *[sql.Identifier(field.name) for field in spec.fields],
                *[sql.Identifier(vector.name) for vector in spec.vectors],
            ]
        ]
        return sql.SQL(
            "INSERT INTO {table} ({columns}) VALUES ({values}) ON CONFLICT ({conflict}) DO UPDATE SET {updates}"
        ).format(
            table=self._qualified_table(spec),
            columns=sql.SQL(", ").join(columns),
            values=sql.SQL(", ").join(values),
            conflict=conflict,
            updates=sql.SQL(", ").join(updates),
        )

    def _prepare_record(self, spec: TableSpec, record: Record) -> tuple[Any, ...]:
        if record.table != spec.name:
            raise ValueError(f"record targets table {record.table!r}, expected {spec.name!r}")
        fields = {field.name: field for field in spec.fields}
        allowed = set(fields) | {spec.primary_key}
        unknown = set(record.payload) - allowed
        if unknown:
            raise ValueError(f"unknown fields for table {spec.name!r}: {', '.join(sorted(unknown))}")
        payload = dict(record.payload)
        if spec.primary_key in fields:
            supplied = payload.get(spec.primary_key, record.record_id)
            if str(supplied) != record.record_id:
                raise ValueError(
                    f"record_id {record.record_id!r} differs from payload primary key {spec.primary_key!r}"
                )
            payload[spec.primary_key] = supplied
        for scope_name, scope_value in record.scope.items():
            if scope_name in payload and str(payload[scope_name]) != str(scope_value):
                raise ValueError(f"payload field {scope_name!r} differs from the record scope")

        values: list[Any] = [_scope_key(record.scope), Jsonb(dict(record.scope.items())), record.record_id]
        for field in spec.fields:
            value = payload.get(field.name, field.default)
            if value is None and not field.nullable:
                raise ValueError(f"field {field.name!r} on table {spec.name!r} is required")
            values.append(_adapt_field_value(field, value))

        vectors = record.vectors or VectorValue()
        dense_specs = {vector.name: vector for vector in spec.vectors if not vector.sparse}
        sparse_specs = {vector.name: vector for vector in spec.vectors if vector.sparse}
        unknown_dense = set(vectors.dense) - set(dense_specs)
        unknown_sparse = set(vectors.sparse) - set(sparse_specs)
        if unknown_dense or unknown_sparse:
            unknown_vectors = sorted(unknown_dense | unknown_sparse)
            raise ValueError(f"unknown vectors for table {spec.name!r}: {', '.join(unknown_vectors)}")
        for vector in spec.vectors:
            if vector.sparse:
                sparse_value = vectors.sparse.get(vector.name)
                if sparse_value is None:
                    values.append(None)
                else:
                    self._validate_sparse(vector, sparse_value)
                    values.append(_sparse_literal(sparse_value, vector.dimensions))
            else:
                dense_value = vectors.dense.get(vector.name)
                if dense_value is None:
                    values.append(None)
                else:
                    self._validate_dense(vector, dense_value)
                    values.append(_dense_literal(dense_value))
        return tuple(values)

    def _select_columns(self, spec: TableSpec, *, with_vectors: bool) -> sql.Composable:
        columns: list[sql.Composable] = [sql.SQL("_scope"), sql.SQL("_record_id")]
        columns.extend(sql.Identifier(field.name) for field in spec.fields)
        if with_vectors:
            for index, vector in enumerate(spec.vectors):
                prefix = "__sparse" if vector.sparse else "__dense"
                columns.append(
                    sql.SQL("{}::text AS {}").format(
                        sql.Identifier(vector.name),
                        sql.Identifier(f"{prefix}_{index}"),
                    )
                )
        return sql.SQL(", ").join(columns)

    def _row_to_record(self, spec: TableSpec, row: Mapping[str, Any], *, with_vectors: bool) -> Record:
        payload = {field.name: row[field.name] for field in spec.fields}
        vectors: VectorValue | None = None
        if with_vectors:
            dense: dict[str, tuple[float, ...]] = {}
            sparse: dict[str, SparseVector] = {}
            for index, vector in enumerate(spec.vectors):
                key = f"{'__sparse' if vector.sparse else '__dense'}_{index}"
                raw = row.get(key)
                if raw is None:
                    continue
                if vector.sparse:
                    sparse[vector.name] = _parse_sparse_literal(raw)
                else:
                    dense[vector.name] = _parse_dense_literal(raw)
            vectors = VectorValue(dense=dense, sparse=sparse)
        return Record(
            table=spec.name,
            record_id=row["_record_id"],
            scope=DatabaseScope(row["_scope"]),
            payload=payload,
            vectors=vectors,
        )

    def _where_clause(
        self,
        spec: TableSpec,
        scope: DatabaseScope,
        filters: FilterExpression | None,
    ) -> tuple[sql.Composable, list[Any]]:
        clauses: list[sql.Composable] = [sql.SQL("_scope @> %s")]
        params: list[Any] = [Jsonb(dict(scope.items()))]
        if filters is not None:
            expression, expression_params = self._compile_filter(spec, filters)
            clauses.append(expression)
            params.extend(expression_params)
        return sql.SQL(" AND ").join(sql.SQL("({})").format(clause) for clause in clauses), params

    def _compile_filter(
        self,
        spec: TableSpec,
        expression: FilterExpression,
    ) -> tuple[sql.Composable, list[Any]]:
        if isinstance(expression, FilterGroup):
            compiled = [self._compile_filter(spec, clause) for clause in expression.clauses]
            if expression.operator == "not":
                if not compiled:
                    return sql.SQL("TRUE"), []
                parts = [item[0] for item in compiled]
                params = [value for item in compiled for value in item[1]]
                return sql.SQL("NOT ({})").format(sql.SQL(" OR ").join(parts)), params
            if not compiled:
                return (sql.SQL("TRUE"), []) if expression.operator == "and" else (sql.SQL("FALSE"), [])
            joiner = sql.SQL(" AND ") if expression.operator == "and" else sql.SQL(" OR ")
            return joiner.join(sql.SQL("({})").format(item[0]) for item in compiled), [
                value for item in compiled for value in item[1]
            ]
        return self._compile_predicate(spec, expression)

    def _compile_predicate(self, spec: TableSpec, predicate: Predicate) -> tuple[sql.Composable, list[Any]]:
        resolved = self._resolve_field(spec, predicate.field)
        field, prefix_params, field_type, nested = resolved
        value = predicate.value
        if predicate.op == "is_null":
            expected = True if value is None else bool(value)
            operator = sql.SQL("IS NULL") if expected else sql.SQL("IS NOT NULL")
            return sql.SQL("{} {}").format(field, operator), prefix_params
        if predicate.op == "is_empty":
            if nested or field_type == FieldType.TEXT:
                return (
                    sql.SQL("({field} IS NULL OR {field} IN ('', '[]', '{{}}', 'null'))").format(field=field),
                    [*prefix_params, *prefix_params],
                )
            if field_type in {FieldType.TEXT_ARRAY, FieldType.UUID_ARRAY}:
                return (
                    sql.SQL("({field} IS NULL OR cardinality({field}) = 0)").format(field=field),
                    [*prefix_params, *prefix_params],
                )
            if field_type == FieldType.JSON:
                return (
                    sql.SQL(
                        "({field} IS NULL OR {field} IN "
                        "('null'::jsonb, '\"\"'::jsonb, '[]'::jsonb, '{{}}'::jsonb))"
                    ).format(field=field),
                    [*prefix_params, *prefix_params],
                )
            return sql.SQL("{} IS NULL").format(field), prefix_params
        if predicate.op in {"in", "not_in"}:
            if not isinstance(value, (list, tuple, set, frozenset)):
                raise TypeError(f"filter operator {predicate.op!r} requires a sequence value")
            values = list(value)
            if not values:
                return (sql.SQL("FALSE"), []) if predicate.op == "in" else (sql.SQL("TRUE"), [])
            comparisons: list[sql.Composable] = []
            params: list[Any] = []
            for item in values:
                comparison, comparison_params = self._comparison(
                    field,
                    prefix_params,
                    field_type,
                    nested,
                    "eq",
                    item,
                )
                comparisons.append(comparison)
                params.extend(comparison_params)
            result = sql.SQL(" OR ").join(sql.SQL("({})").format(item) for item in comparisons)
            if predicate.op == "not_in":
                result = sql.SQL("NOT ({})").format(result)
            return result, params
        return self._comparison(field, prefix_params, field_type, nested, predicate.op, value)

    def _comparison(
        self,
        field: sql.Composable,
        prefix_params: list[Any],
        field_type: FieldType,
        nested: bool,
        operator: str,
        value: Any,
    ) -> tuple[sql.Composable, list[Any]]:
        if operator in {"eq", "ne"}:
            sql_operator = sql.SQL("IS NOT DISTINCT FROM") if operator == "eq" else sql.SQL("IS DISTINCT FROM")
            if field_type == FieldType.JSON and not nested:
                adapted = Jsonb(value)
            elif nested:
                adapted = _nested_scalar(value)
            else:
                adapted = value
            return sql.SQL("{} {} %s").format(field, sql_operator), [*prefix_params, adapted]
        if operator in {"gt", "gte", "lt", "lte"}:
            if field_type in {FieldType.JSON} and nested:
                field = _cast_nested_field(field, value)
            sql_operator = sql.SQL({"gt": ">", "gte": ">=", "lt": "<", "lte": "<="}[operator])
            return sql.SQL("{} {} %s").format(field, sql_operator), [*prefix_params, value]
        if operator in {"contains", "icontains"}:
            if field_type == FieldType.TEXT or nested:
                comparison = sql.SQL("{} ILIKE %s") if operator == "icontains" else sql.SQL("{} LIKE %s")
                return comparison.format(field), [*prefix_params, f"%{value}%"]
            if field_type in {FieldType.TEXT_ARRAY, FieldType.UUID_ARRAY}:
                if operator == "icontains":
                    raise ValueError(f"operator {operator!r} is not supported for array fields")
                return sql.SQL("%s = ANY({})").format(field), [*prefix_params, value]
            if field_type == FieldType.JSON:
                return sql.SQL("{} @> %s").format(field), [*prefix_params, Jsonb(value)]
            raise ValueError(f"operator {operator!r} is not supported for field type {field_type.value!r}")
        raise ValueError(f"unsupported filter operator {operator!r}")

    def _resolve_field(
        self,
        spec: TableSpec,
        name: str,
    ) -> tuple[sql.Composable, list[Any], FieldType, bool]:
        fields = {field.name: field for field in spec.fields}
        if name in fields:
            field = fields[name]
            return sql.Identifier(name), [], field.field_type, False
        if name == spec.primary_key:
            return sql.SQL("_record_id"), [], FieldType.TEXT, False
        root, separator, path = name.partition(".")
        if separator and root in fields and fields[root].field_type == FieldType.JSON:
            return (
                sql.SQL("({} #>> %s)").format(sql.Identifier(root)),
                [path.split(".")],
                FieldType.JSON,
                True,
            )
        raise ValueError(f"unknown filter/sort field {name!r} for table {spec.name!r}")

    def _order_clause(self, spec: TableSpec, sorts: Sequence[Sort]) -> tuple[sql.Composable, list[Any]]:
        expressions: list[sql.Composable] = []
        params: list[Any] = []
        for item in sorts:
            field, field_params, _, _ = self._resolve_field(spec, item.field)
            direction = sql.SQL("DESC") if item.direction == "desc" else sql.SQL("ASC")
            expressions.append(sql.SQL("{} {} NULLS LAST").format(field, direction))
            params.extend(field_params)
        if spec.scope_scoped:
            expressions.append(sql.SQL("_scope_key ASC"))
        expressions.append(sql.SQL("_record_id ASC"))
        return sql.SQL(", ").join(expressions), params

    async def _search_one(
        self,
        query: VectorQuery,
        vector: VectorFieldSpec,
        vector_literal: str,
        *,
        source: str,
    ) -> list[VectorHit]:
        spec = self._table(query.table)
        where, params = self._where_clause(spec, query.scope, query.filters)
        operator = sql.SQL({"cosine": "<=>", "euclidean": "<->", "dot": "<#>"}[vector.distance])
        vector_type = sql.SQL("sparsevec" if vector.sparse else "vector")
        distance = sql.SQL("{} {} %s::{}").format(sql.Identifier(vector.name), operator, vector_type)
        score = _score_expression(vector.distance, sql.SQL("_distance"))
        threshold = sql.SQL("WHERE {} >= %s").format(score) if query.score_threshold is not None else sql.SQL("")
        statement = sql.SQL(
            "SELECT ranked.*, {score} AS _score FROM ("
            "SELECT {select}, {distance} AS _distance FROM {table} "
            "WHERE {where} AND {vector} IS NOT NULL"
            ") AS ranked {threshold} ORDER BY _distance ASC, _record_id ASC LIMIT %s"
        ).format(
            score=score,
            select=self._select_columns(spec, with_vectors=True),
            distance=distance,
            table=self._qualified_table(spec),
            where=where,
            vector=sql.Identifier(vector.name),
            threshold=threshold,
        )
        statement_params: list[Any] = [vector_literal, *params]
        if query.score_threshold is not None:
            statement_params.append(query.score_threshold)
        statement_params.append(query.top_k)
        rows = await self._fetchall(statement, statement_params)
        return [
            VectorHit(
                record=self._row_to_record(spec, row, with_vectors=True),
                score=float(row["_score"]),
                source=source,
                debug={"distance": float(row["_distance"]), "vector_name": vector.name},
            )
            for row in rows
        ]

    def _fuse_rrf(
        self,
        query: VectorQuery,
        dense_hits: Sequence[VectorHit],
        sparse_hits: Sequence[VectorHit],
    ) -> list[VectorHit]:
        by_id: dict[tuple[tuple[tuple[str, Any], ...], str], dict[str, Any]] = {}
        for source_name, weight, hits in (
            ("dense", self._options.dense_weight, dense_hits),
            ("sparse", self._options.sparse_weight, sparse_hits),
        ):
            for rank, hit in enumerate(hits, start=1):
                key = (hit.record.scope.items(), hit.record.record_id)
                state = by_id.setdefault(
                    key,
                    {"record": hit.record, "score": 0.0, "ranks": {}, "native_scores": {}},
                )
                state["score"] += weight / (self._options.rrf_k + rank - 1)
                state["ranks"][source_name] = rank
                state["native_scores"][source_name] = hit.score
        ranked = sorted(
            by_id.values(),
            key=lambda item: (-item["score"], item["record"].record_id),
        )
        results: list[VectorHit] = []
        for item in ranked:
            score = float(item["score"])
            if query.score_threshold is not None and score < query.score_threshold:
                continue
            results.append(
                VectorHit(
                    record=item["record"],
                    score=score,
                    source="rrf",
                    debug={
                        "ranks": item["ranks"],
                        "native_scores": item["native_scores"],
                        "rrf_k": self._options.rrf_k,
                        "dense_weight": self._options.dense_weight,
                        "sparse_weight": self._options.sparse_weight,
                    },
                )
            )
            if len(results) == query.top_k:
                break
        return results

    def _sparse_query_spec(self, spec: TableSpec, requested_name: str) -> VectorFieldSpec:
        sparse = [vector for vector in spec.vectors if vector.sparse]
        for vector in sparse:
            if vector.name == requested_name:
                return vector
        if len(sparse) == 1:
            return sparse[0]
        raise ValueError(f"table {spec.name!r} cannot infer a sparse vector from dense vector name {requested_name!r}")

    def _vector_spec(self, spec: TableSpec, name: str, *, sparse: bool) -> VectorFieldSpec:
        for vector in spec.vectors:
            if vector.name == name and vector.sparse == sparse:
                return vector
        kind = "sparse" if sparse else "dense"
        raise ValueError(f"unknown {kind} vector {name!r} for table {spec.name!r}")

    def _validate_dense(self, spec: VectorFieldSpec, values: Sequence[float]) -> None:
        if len(values) != spec.dimensions:
            raise ValueError(
                f"dense vector {spec.name!r} requires {spec.dimensions} dimensions, received {len(values)}"
            )
        if any(not math.isfinite(value) for value in values):
            raise ValueError(f"dense vector {spec.name!r} contains a non-finite value")

    def _validate_sparse(self, spec: VectorFieldSpec, value: SparseVector) -> None:
        if len(value.indices) > _MAX_SPARSE_NON_ZERO:
            raise ValueError(f"sparse vector {spec.name!r} exceeds {_MAX_SPARSE_NON_ZERO} non-zero values")
        if len(set(value.indices)) != len(value.indices):
            raise ValueError(f"sparse vector {spec.name!r} contains duplicate indices")
        if any(index >= spec.dimensions for index in value.indices):
            raise ValueError(f"sparse vector {spec.name!r} contains an index outside its dimensions")
        if any(not math.isfinite(item) for item in value.values):
            raise ValueError(f"sparse vector {spec.name!r} contains a non-finite value")

    def _exact_scope_clause(self, spec: TableSpec) -> sql.Composable:
        return sql.SQL("_scope_key = %s") if spec.scope_scoped else sql.SQL("TRUE")

    def _qualified_table(self, spec: TableSpec) -> sql.Composable:
        return sql.Identifier(self._options.schema, spec.name)

    def _table(self, name: str) -> TableSpec:
        try:
            return self._tables[name]
        except KeyError as exc:
            raise KeyError(f"unknown logical table {name!r}") from exc

    def _validate_registry(self) -> None:
        for spec in self._tables.values():
            names = {field.name for field in spec.fields} | {vector.name for vector in spec.vectors}
            reserved = names & _RESERVED_COLUMNS
            if reserved:
                raise ValueError(f"table {spec.name!r} uses reserved pgvector columns: {sorted(reserved)}")
            duplicates = {field.name for field in spec.fields} & {vector.name for vector in spec.vectors}
            if duplicates:
                raise ValueError(f"table {spec.name!r} reuses fields as vectors: {sorted(duplicates)}")
            for vector in spec.vectors:
                if not vector.sparse and vector.dimensions > _MAX_VECTOR_DIMENSIONS:
                    raise ValueError(
                        f"vector {spec.name}.{vector.name} has {vector.dimensions} dimensions; "
                        f"pgvector supports at most {_MAX_VECTOR_DIMENSIONS}"
                    )


_POSTGRES_TYPES = {
    FieldType.UUID: "uuid",
    FieldType.TEXT: "text",
    FieldType.INTEGER: "bigint",
    FieldType.FLOAT: "double precision",
    FieldType.BOOLEAN: "boolean",
    FieldType.DATETIME: "timestamptz",
    FieldType.TEXT_ARRAY: "text[]",
    FieldType.UUID_ARRAY: "uuid[]",
    FieldType.JSON: "jsonb",
}


def create_pgvector_backend(options: Mapping[str, Any], tables: TableRegistry) -> PgVectorBackend:
    """Backend factory compatible with :class:`BackendRegistry`."""

    return PgVectorBackend(options=PgVectorOptions.from_mapping(options), tables=tables)


def register_pgvector_backend(registry: BackendRegistry) -> None:
    """Register the stable ``pgvector`` provider name."""

    registry.register("pgvector", create_pgvector_backend)


def _scope_key(scope: DatabaseScope) -> str:
    return json.dumps(dict(scope.items()), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _adapt_field_value(field: FieldSpec, value: Any) -> Any:
    if value is None:
        return None
    if field.field_type == FieldType.JSON:
        return Jsonb(value)
    if field.field_type in {FieldType.TEXT_ARRAY, FieldType.UUID_ARRAY}:
        return list(value)
    return value


def _dense_literal(values: Sequence[float]) -> str:
    return "[" + ",".join(_format_float(value) for value in values) + "]"


def _sparse_literal(value: SparseVector, dimensions: int) -> str:
    pairs = sorted(zip(value.indices, value.values, strict=True))
    body = ",".join(f"{index + 1}:{_format_float(item)}" for index, item in pairs)
    return "{" + body + f"}}/{dimensions}"


def _parse_dense_literal(value: str) -> tuple[float, ...]:
    body = value.strip()[1:-1]
    return tuple(float(item) for item in body.split(",")) if body else ()


def _parse_sparse_literal(value: str) -> SparseVector:
    match = re.fullmatch(r"\{(.*)\}/\d+", value.strip())
    if match is None:
        raise ValueError(f"invalid pgvector sparsevec value: {value!r}")
    body = match.group(1)
    if not body:
        return SparseVector(indices=(), values=())
    pairs = [item.split(":", maxsplit=1) for item in body.split(",")]
    return SparseVector(
        indices=tuple(int(index) - 1 for index, _ in pairs),
        values=tuple(float(item) for _, item in pairs),
    )


def _format_float(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError("pgvector values must be finite")
    return repr(float(value))


def _schema_fingerprint(spec: TableSpec) -> str:
    encoded = json.dumps(asdict(spec), sort_keys=True, separators=(",", ":"), default=_json_default)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, date, UUID)):
        return str(value)
    raise TypeError(f"cannot encode schema default {value!r}")


def _score_expression(distance: str, value: sql.Composable) -> sql.Composable:
    if distance == "cosine":
        return sql.SQL("(1.0 - {})").format(value)
    if distance == "euclidean":
        return sql.SQL("(1.0 / (1.0 + {}))").format(value)
    return sql.SQL("(-{})").format(value)


def _cast_nested_field(field: sql.Composable, value: Any) -> sql.Composable:
    if isinstance(value, bool):
        return sql.SQL("({})::boolean").format(field)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return sql.SQL("({})::numeric").format(field)
    return field


def _nested_scalar(value: Any) -> Any:
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _encode_cursor(table: str, offset: int) -> str:
    raw = json.dumps({"table": table, "offset": offset}, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _decode_cursor(cursor: str | None, table: str) -> int:
    if cursor is None:
        return 0
    try:
        padding = "=" * (-len(cursor) % 4)
        value = json.loads(base64.urlsafe_b64decode(cursor + padding))
        if value["table"] != table or not isinstance(value["offset"], int) or value["offset"] < 0:
            raise ValueError
        return value["offset"]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("invalid record query cursor") from exc


__all__ = [
    "PgVectorBackend",
    "PgVectorOptions",
    "create_pgvector_backend",
    "register_pgvector_backend",
]
