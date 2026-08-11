"""PostgreSQL adapter for backend-neutral structured records."""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any, AsyncIterator

from psycopg import sql
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from ..database import DatabaseUnitOfWork, ScopedDatabase
from ..models import (
    DatabaseCapabilities,
    FieldSpec,
    FieldType,
    FilterExpression,
    FilterGroup,
    Record,
    RecordQuery,
    SchemaMigration,
    Sort,
    TableSpec,
)
from ..registry import DatabaseRegistry, TableRegistry
from ..scope import DatabaseScope

_SCHEMA_TABLE = "__mindmemos_schema"
_MIGRATION_TABLE = "__mindmemos_migrations"


@dataclass(frozen=True, slots=True, kw_only=True)
class PostgresOptions:
    dsn: str
    pool_size: int = 10
    min_size: int = 1
    timeout: float = 30.0

    @classmethod
    def from_mapping(cls, options: Mapping[str, Any]) -> PostgresOptions:
        allowed = set(cls.__dataclass_fields__)
        unknown = set(options) - allowed
        if unknown:
            raise ValueError(f"unknown postgres backend options: {', '.join(sorted(unknown))}")
        result = cls(**dict(options))
        if not result.dsn:
            raise ValueError("postgres backend option 'dsn' must not be empty")
        if result.min_size < 0 or result.pool_size <= 0 or result.min_size > result.pool_size:
            raise ValueError("invalid postgres pool sizes")
        return result


class PostgresBackend(ScopedDatabase):
    _capabilities = DatabaseCapabilities(
        metadata_filtering=True,
        batch_record_io=True,
        atomic_batch_write=True,
        transactions=True,
        compare_and_swap=True,
    )

    def __init__(self, *, options: PostgresOptions, tables: TableRegistry) -> None:
        self._options = options
        self._registry = tables
        self._tables = {spec.name: spec for spec in tables.specs}
        self._pool = AsyncConnectionPool(
            conninfo=options.dsn,
            min_size=options.min_size,
            max_size=options.pool_size,
            timeout=options.timeout,
            open=False,
            kwargs={"row_factory": dict_row},
        )
        self._opened = False

    @property
    def name(self) -> str:
        return "postgres"

    @property
    def capabilities(self) -> DatabaseCapabilities:
        return self._capabilities

    async def _open(self) -> None:
        if not self._opened:
            await self._pool.open()
            self._opened = True

    async def ensure_schema(self, tables: TableRegistry) -> None:
        if tables.specs != self._registry.specs or tables.migrations != self._registry.migrations:
            raise ValueError("ensure_schema registry differs from the configured postgres registry")
        await self._open()
        async with self._pool.connection() as connection, connection.transaction():
            await connection.execute(
                sql.SQL(
                    "CREATE TABLE IF NOT EXISTS {} (table_name TEXT PRIMARY KEY, schema_fingerprint TEXT NOT NULL)"
                ).format(sql.Identifier(_SCHEMA_TABLE))
            )
            await connection.execute(
                sql.SQL(
                    "CREATE TABLE IF NOT EXISTS {} (namespace TEXT NOT NULL, version INTEGER NOT NULL, "
                    "name TEXT NOT NULL, checksum TEXT NOT NULL, applied_at TIMESTAMPTZ NOT NULL, "
                    "PRIMARY KEY (namespace, version))"
                ).format(sql.Identifier(_MIGRATION_TABLE))
            )
            if tables.migrations:
                await self._apply_migrations(connection, tables.migrations)
            else:
                for spec in self._tables.values():
                    await self._ensure_table(connection, spec)

    async def _apply_migrations(self, connection, migrations: tuple[SchemaMigration, ...]) -> None:
        covered_tables = {table for migration in migrations for table in migration.tables}
        missing = set(self._tables) - covered_tables
        if missing:
            raise RuntimeError(f"registered tables are missing an explicit schema migration: {sorted(missing)}")

        ledger_rows = await (
            await connection.execute(
                sql.SQL("SELECT namespace, version, name, checksum FROM {}").format(sql.Identifier(_MIGRATION_TABLE))
            )
        ).fetchall()
        registered = {(item.namespace, item.version): item for item in migrations}
        unexpected = sorted(
            (str(row["namespace"]), int(row["version"]))
            for row in ledger_rows
            if (str(row["namespace"]), int(row["version"])) not in registered
        )
        if unexpected:
            raise RuntimeError(f"postgres schema ledger contains unknown migrations: {unexpected}")

        applied: set[tuple[str, int]] = set()
        for row in ledger_rows:
            identity = (str(row["namespace"]), int(row["version"]))
            migration = registered[identity]
            checksum = _migration_checksum(migration)
            if row["name"] != migration.name or row["checksum"] != checksum:
                raise RuntimeError(
                    f"postgres migration definition drift for {migration.namespace!r} version {migration.version}"
                )
            applied.add(identity)

        namespaces = {migration.namespace for migration in migrations}
        for namespace in namespaces:
            applied_versions = sorted(
                version for applied_namespace, version in applied if applied_namespace == namespace
            )
            if applied_versions != list(range(1, len(applied_versions) + 1)):
                raise RuntimeError(
                    f"postgres schema ledger for namespace {namespace!r} is not a contiguous prefix: {applied_versions}"
                )

        schema_row = await (
            await connection.execute(sql.SQL("SELECT COUNT(*) AS count FROM {}").format(sql.Identifier(_SCHEMA_TABLE)))
        ).fetchone()
        existing_registered_tables = set()
        for table_name in self._tables:
            row = await (await connection.execute("SELECT to_regclass(%s) AS table_name", (table_name,))).fetchone()
            if row["table_name"] is not None:
                existing_registered_tables.add(table_name)
        fresh_database = not ledger_rows and int(schema_row["count"]) == 0 and not existing_registered_tables
        if not ledger_rows and not fresh_database:
            raise RuntimeError(
                "postgres database contains registered schema objects without a migration ledger; "
                "recreate this unpublished database or adopt it through an explicit migration"
            )

        if fresh_database:
            for spec in self._tables.values():
                await self._ensure_table(connection, spec)
            for migration in migrations:
                await self._record_migration(connection, migration)
            return

        touched_tables: set[str] = set()
        for migration in migrations:
            if (migration.namespace, migration.version) in applied:
                continue
            for statement in migration.statements_for("postgres"):
                await connection.execute(statement)
            touched_tables.update(migration.tables)
            await self._record_migration(connection, migration)

        for spec in self._tables.values():
            await self._ensure_table(
                connection,
                spec,
                allow_migrated_fingerprint=spec.name in touched_tables,
            )

    async def _record_migration(self, connection, migration: SchemaMigration) -> None:
        await connection.execute(
            sql.SQL(
                "INSERT INTO {} (namespace, version, name, checksum, applied_at) VALUES (%s, %s, %s, %s, %s)"
            ).format(sql.Identifier(_MIGRATION_TABLE)),
            (
                migration.namespace,
                migration.version,
                migration.name,
                _migration_checksum(migration),
                datetime.now(UTC),
            ),
        )

    async def _ensure_table(
        self,
        connection,
        spec: TableSpec,
        *,
        allow_migrated_fingerprint: bool = False,
    ) -> None:
        fingerprint = _schema_fingerprint(spec)
        row = await (
            await connection.execute(
                sql.SQL("SELECT schema_fingerprint FROM {} WHERE table_name = %s").format(
                    sql.Identifier(_SCHEMA_TABLE)
                ),
                (spec.name,),
            )
        ).fetchone()
        if row is not None and row["schema_fingerprint"] != fingerprint and not allow_migrated_fingerprint:
            raise RuntimeError(f"postgres schema drift for table {spec.name!r}; apply an explicit migration")
        await connection.execute(_table_ddl(spec))
        for statement in _index_ddl(spec):
            await connection.execute(statement)
        await connection.execute(
            sql.SQL(
                "INSERT INTO {} (table_name, schema_fingerprint) VALUES (%s, %s) "
                "ON CONFLICT (table_name) DO UPDATE SET schema_fingerprint = EXCLUDED.schema_fingerprint"
            ).format(sql.Identifier(_SCHEMA_TABLE)),
            (spec.name, fingerprint),
        )

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[DatabaseUnitOfWork]:
        await self._open()
        async with self._pool.connection() as connection, connection.transaction():
            yield _PostgresUnitOfWork(self, connection)

    async def upsert_records(self, table: str, records: Sequence[Record]) -> None:
        await self._open()
        async with self._pool.connection() as connection, connection.transaction():
            await self._upsert(connection, table, records)

    async def get_records(self, table: str, scope: DatabaseScope, record_ids: Sequence[str]) -> list[Record]:
        await self._open()
        async with self._pool.connection() as connection:
            return await self._get(connection, table, scope, record_ids)

    async def patch_record(self, table: str, scope: DatabaseScope, record_id: str, changes: Mapping[str, Any]) -> None:
        await self._open()
        async with self._pool.connection() as connection, connection.transaction():
            await self._patch(connection, table, scope, record_id, changes)

    async def compare_and_swap_record(
        self,
        table: str,
        scope: DatabaseScope,
        record_id: str,
        *,
        expected: Mapping[str, Any],
        changes: Mapping[str, Any],
    ) -> bool:
        await self._open()
        async with self._pool.connection() as connection, connection.transaction():
            return await self._cas(connection, table, scope, record_id, expected, changes)

    async def delete_records(self, table: str, scope: DatabaseScope, record_ids: Sequence[str]) -> None:
        await self._open()
        async with self._pool.connection() as connection, connection.transaction():
            await self._delete(connection, table, scope, record_ids)

    async def query_records(self, table: str, query: RecordQuery) -> tuple[list[Record], str | None]:
        await self._open()
        async with self._pool.connection() as connection:
            return await self._query(connection, table, query)

    async def close(self) -> None:
        if self._opened:
            await self._pool.close()
            self._opened = False

    async def _upsert(self, connection, table: str, records: Sequence[Record]) -> None:
        if not records:
            return
        spec = self._table(table)
        columns = ["_scope_key", "_scope", "_record_id", *[field.name for field in spec.fields]]
        conflict = ["_scope_key", "_record_id"] if spec.scope_scoped else ["_record_id"]
        updates = ["_scope", *[field.name for field in spec.fields]]
        statement = sql.SQL("INSERT INTO {} ({}) VALUES ({}) ON CONFLICT ({}) DO UPDATE SET {}").format(
            sql.Identifier(spec.name),
            sql.SQL(", ").join(map(sql.Identifier, columns)),
            sql.SQL(", ").join(sql.Placeholder() for _ in columns),
            sql.SQL(", ").join(map(sql.Identifier, conflict)),
            sql.SQL(", ").join(
                sql.SQL("{} = EXCLUDED.{}").format(sql.Identifier(name), sql.Identifier(name)) for name in updates
            ),
        )
        async with connection.cursor() as cursor:
            await cursor.executemany(statement, [_prepare_record(spec, record) for record in records])

    async def _get(self, connection, table: str, scope: DatabaseScope, record_ids: Sequence[str]) -> list[Record]:
        if not record_ids:
            return []
        spec = self._table(table)
        condition = sql.SQL("_scope_key = %s AND ") if spec.scope_scoped else sql.SQL("")
        statement = sql.SQL("SELECT * FROM {} WHERE {}_record_id = ANY(%s)").format(
            sql.Identifier(spec.name), condition
        )
        params: list[Any] = []
        if spec.scope_scoped:
            params.append(_scope_key(scope))
        params.append(list(record_ids))
        rows = await (await connection.execute(statement, params)).fetchall()
        by_id = {row["_record_id"]: row for row in rows}
        return [_row_to_record(spec, by_id[item]) for item in record_ids if item in by_id]

    async def _patch(self, connection, table, scope, record_id, changes) -> None:
        spec = self._table(table)
        fields = {field.name: field for field in spec.fields}
        _validate_changes(spec, fields, changes)
        assignments = sql.SQL(", ").join(sql.SQL("{} = %s").format(sql.Identifier(name)) for name in changes)
        condition = sql.SQL("_scope_key = %s AND ") if spec.scope_scoped else sql.SQL("")
        statement = sql.SQL("UPDATE {} SET {} WHERE {}_record_id = %s").format(
            sql.Identifier(spec.name), assignments, condition
        )
        params = [_adapt(fields[name], value) for name, value in changes.items()]
        if spec.scope_scoped:
            params.append(_scope_key(scope))
        params.append(record_id)
        await connection.execute(statement, params)

    async def _cas(self, connection, table, scope, record_id, expected, changes) -> bool:
        if not expected or not changes:
            raise ValueError("compare-and-swap requires expected and changed fields")
        spec = self._table(table)
        fields = {field.name: field for field in spec.fields}
        _validate_changes(spec, fields, {**expected, **changes})
        assignments = sql.SQL(", ").join(sql.SQL("{} = %s").format(sql.Identifier(name)) for name in changes)
        comparisons = sql.SQL(" AND ").join(
            sql.SQL("{} IS NOT DISTINCT FROM %s").format(sql.Identifier(name)) for name in expected
        )
        scope_condition = sql.SQL("_scope_key = %s AND ") if spec.scope_scoped else sql.SQL("")
        statement = sql.SQL("UPDATE {} SET {} WHERE {}_record_id = %s AND {}").format(
            sql.Identifier(spec.name), assignments, scope_condition, comparisons
        )
        params = [_adapt(fields[name], value) for name, value in changes.items()]
        if spec.scope_scoped:
            params.append(_scope_key(scope))
        params.append(record_id)
        params.extend(_adapt(fields[name], value) for name, value in expected.items())
        cursor = await connection.execute(statement, params)
        return cursor.rowcount == 1

    async def _delete(self, connection, table, scope, record_ids) -> None:
        if not record_ids:
            return
        spec = self._table(table)
        condition = sql.SQL("_scope_key = %s AND ") if spec.scope_scoped else sql.SQL("")
        statement = sql.SQL("DELETE FROM {} WHERE {}_record_id = ANY(%s)").format(sql.Identifier(spec.name), condition)
        params: list[Any] = []
        if spec.scope_scoped:
            params.append(_scope_key(scope))
        params.append(list(record_ids))
        await connection.execute(statement, params)

    async def _query(self, connection, table: str, query: RecordQuery) -> tuple[list[Record], str | None]:
        spec = self._table(table)
        offset = _decode_cursor(query.page.cursor, table)
        where, params = _where(spec, query.scope, query.filters)
        order, order_params = _order(spec, query.sort)
        statement = sql.SQL("SELECT * FROM {} WHERE {} ORDER BY {} LIMIT %s OFFSET %s").format(
            sql.Identifier(spec.name), where, order
        )
        rows = await (
            await connection.execute(statement, [*params, *order_params, query.page.limit + 1, offset])
        ).fetchall()
        has_more = len(rows) > query.page.limit
        records = [_row_to_record(spec, row) for row in rows[: query.page.limit]]
        next_cursor = _encode_cursor(table, offset + query.page.limit) if has_more else None
        return records, next_cursor

    def _table(self, name: str) -> TableSpec:
        try:
            return self._tables[name]
        except KeyError as exc:
            raise KeyError(f"unknown logical table {name!r}") from exc


class _PostgresUnitOfWork(DatabaseUnitOfWork):
    def __init__(self, backend: PostgresBackend, connection) -> None:
        self._backend = backend
        self._connection = connection

    async def upsert_records(self, table, records):
        await self._backend._upsert(self._connection, table, records)

    async def get_records(self, table, scope, record_ids):
        return await self._backend._get(self._connection, table, scope, record_ids)

    async def patch_record(self, table, scope, record_id, changes):
        await self._backend._patch(self._connection, table, scope, record_id, changes)

    async def compare_and_swap_record(self, table, scope, record_id, *, expected, changes):
        return await self._backend._cas(self._connection, table, scope, record_id, expected, changes)

    async def delete_records(self, table, scope, record_ids):
        await self._backend._delete(self._connection, table, scope, record_ids)

    async def query_records(self, table, query):
        return await self._backend._query(self._connection, table, query)


def _table_ddl(spec: TableSpec):
    columns = [
        sql.SQL("_scope_key TEXT NOT NULL"),
        sql.SQL("_scope JSONB NOT NULL"),
        sql.SQL("_record_id TEXT NOT NULL"),
    ]
    for field in spec.fields:
        declaration = sql.SQL("{} {}").format(sql.Identifier(field.name), sql.SQL(_PG_TYPES[field.field_type]))
        if not field.nullable:
            declaration += sql.SQL(" NOT NULL")
        columns.append(declaration)
    primary = (
        sql.SQL("PRIMARY KEY (_scope_key, _record_id)") if spec.scope_scoped else sql.SQL("PRIMARY KEY (_record_id)")
    )
    columns.append(primary)
    return sql.SQL("CREATE TABLE IF NOT EXISTS {} ({})").format(sql.Identifier(spec.name), sql.SQL(", ").join(columns))


def _index_ddl(spec: TableSpec):
    statements = []
    for index in spec.indexes:
        columns = [*(["_scope_key"] if spec.scope_scoped else []), *index.fields]
        translated = ["_record_id" if item == spec.primary_key else item for item in columns]
        statements.append(
            sql.SQL("CREATE {}INDEX IF NOT EXISTS {} ON {} ({})").format(
                sql.SQL("UNIQUE ") if index.unique else sql.SQL(""),
                sql.Identifier(index.name),
                sql.Identifier(spec.name),
                sql.SQL(", ").join(map(sql.Identifier, translated)),
            )
        )
    return statements


def _prepare_record(spec: TableSpec, record: Record):
    if record.table != spec.name:
        raise ValueError(f"record targets {record.table!r}, expected {spec.name!r}")
    payload = dict(record.payload)
    if str(payload.get(spec.primary_key, record.record_id)) != record.record_id:
        raise ValueError("record primary key differs from record_id")
    payload[spec.primary_key] = record.record_id
    fields = {field.name: field for field in spec.fields}
    unknown = set(payload) - set(fields)
    if unknown:
        raise ValueError(f"unknown fields for table {spec.name!r}: {sorted(unknown)}")
    values = [_scope_key(record.scope), Jsonb(dict(record.scope.items())), record.record_id]
    for field in spec.fields:
        value = payload.get(field.name, field.default)
        if value is None and not field.nullable:
            raise ValueError(f"field {field.name!r} is required")
        values.append(_adapt(field, value))
    return tuple(values)


def _adapt(field: FieldSpec, value: Any):
    if value is None:
        return None
    if field.field_type == FieldType.JSON:
        return Jsonb(value)
    if field.field_type in {FieldType.TEXT_ARRAY, FieldType.UUID_ARRAY}:
        return list(value)
    if field.field_type == FieldType.DATETIME and isinstance(value, str):
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    if hasattr(value, "value"):
        return value.value
    return value


def _row_to_record(spec: TableSpec, row: Mapping[str, Any]):
    payload = {field.name: row[field.name] for field in spec.fields}
    return Record(table=spec.name, record_id=row["_record_id"], scope=DatabaseScope(row["_scope"]), payload=payload)


def _where(spec: TableSpec, scope: DatabaseScope, filters: FilterExpression | None):
    clauses = []
    params: list[Any] = []
    for name, value in scope.items():
        clauses.append(sql.SQL("_scope ->> %s = %s"))
        params.extend((name, str(value)))
    if filters is not None:
        expression, values = _compile_filter(spec, filters)
        clauses.append(expression)
        params.extend(values)
    return (sql.SQL(" AND ").join(clauses) if clauses else sql.SQL("TRUE")), params


def _compile_filter(spec: TableSpec, expression: FilterExpression):
    if isinstance(expression, FilterGroup):
        compiled = [_compile_filter(spec, item) for item in expression.clauses]
        if not compiled:
            return sql.SQL("TRUE" if expression.operator in {"and", "not"} else "FALSE"), []
        joiner = sql.SQL(" AND " if expression.operator == "and" else " OR ")
        combined = joiner.join(sql.SQL("({})").format(item[0]) for item in compiled)
        if expression.operator == "not":
            combined = sql.SQL("NOT ({})").format(combined)
        return combined, [value for _, values in compiled for value in values]
    fields = {field.name: field for field in spec.fields}
    field = fields.get(expression.field)
    if field is None:
        raise ValueError(f"unknown filter field {expression.field!r}")
    identifier = sql.Identifier(expression.field)
    value = expression.value
    if expression.op == "is_null":
        return sql.SQL("{} IS {}").format(identifier, sql.SQL("NULL" if value is not False else "NOT NULL")), []
    if expression.op in {"in", "not_in"}:
        values = list(value)
        if not values:
            return sql.SQL("FALSE" if expression.op == "in" else "TRUE"), []
        operator = sql.SQL("IN" if expression.op == "in" else "NOT IN")
        return sql.SQL("{} {} ({})").format(
            identifier, operator, sql.SQL(", ").join(sql.Placeholder() for _ in values)
        ), [_adapt(field, item) for item in values]
    if expression.op in {"eq", "ne", "gt", "gte", "lt", "lte"}:
        operator = {"eq": "=", "ne": "<>", "gt": ">", "gte": ">=", "lt": "<", "lte": "<="}[expression.op]
        return sql.SQL("{} {} %s").format(identifier, sql.SQL(operator)), [_adapt(field, value)]
    if expression.op in {"contains", "icontains"}:
        if field.field_type == FieldType.JSON:
            return sql.SQL("{} @> %s").format(identifier), [Jsonb(value)]
        operator = "ILIKE" if expression.op == "icontains" else "LIKE"
        return sql.SQL("{} {} %s").format(identifier, sql.SQL(operator)), [f"%{value}%"]
    if expression.op == "is_empty":
        return sql.SQL("({} IS NULL OR {} = '')").format(identifier, identifier), []
    raise ValueError(f"unsupported postgres filter operator: {expression.op}")


def _order(spec: TableSpec, sorts: Sequence[Sort]):
    expressions = [
        sql.SQL("{} {} NULLS LAST").format(
            sql.Identifier(item.field), sql.SQL("DESC" if item.direction == "desc" else "ASC")
        )
        for item in sorts
    ]
    if spec.scope_scoped:
        expressions.append(sql.SQL("_scope_key ASC"))
    expressions.append(sql.SQL("_record_id ASC"))
    return sql.SQL(", ").join(expressions), []


def _validate_changes(spec, fields, changes):
    unknown = set(changes) - set(fields)
    if unknown:
        raise ValueError(f"unknown fields for table {spec.name!r}: {sorted(unknown)}")
    if spec.primary_key in changes:
        raise ValueError("cannot patch a primary key")


def _scope_key(scope: DatabaseScope):
    return json.dumps(dict(scope.items()), sort_keys=True, separators=(",", ":"))


def _schema_fingerprint(spec: TableSpec):
    payload = {
        "name": spec.name,
        "primary_key": spec.primary_key,
        "scope_scoped": spec.scope_scoped,
        "fields": [asdict(item) for item in spec.fields],
        "indexes": [asdict(item) for item in spec.indexes],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _migration_checksum(migration: SchemaMigration):
    payload = {
        "namespace": migration.namespace,
        "version": migration.version,
        "name": migration.name,
        "tables": migration.tables,
        "sqlite_statements": migration.sqlite_statements,
        "postgres_statements": migration.postgres_statements,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def _encode_cursor(table: str, offset: int):
    return base64.urlsafe_b64encode(json.dumps({"table": table, "offset": offset}).encode()).decode()


def _decode_cursor(cursor: str | None, table: str):
    if cursor is None:
        return 0
    try:
        payload = json.loads(base64.urlsafe_b64decode(cursor))
        if payload["table"] != table:
            raise ValueError("cursor table mismatch")
        return int(payload["offset"])
    except Exception as exc:
        raise ValueError("invalid database cursor") from exc


_PG_TYPES = {
    FieldType.UUID: "UUID",
    FieldType.TEXT: "TEXT",
    FieldType.INTEGER: "INTEGER",
    FieldType.FLOAT: "DOUBLE PRECISION",
    FieldType.BOOLEAN: "BOOLEAN",
    FieldType.DATETIME: "TIMESTAMPTZ",
    FieldType.TEXT_ARRAY: "TEXT[]",
    FieldType.UUID_ARRAY: "UUID[]",
    FieldType.JSON: "JSONB",
}


def create_postgres_backend(options: Mapping[str, Any], tables: TableRegistry) -> PostgresBackend:
    return PostgresBackend(options=PostgresOptions.from_mapping(options), tables=tables)


def register_postgres_backend(registry: DatabaseRegistry) -> None:
    registry.register("postgres", create_postgres_backend)
    registry.register("postgresql", create_postgres_backend)


__all__ = [
    "PostgresBackend",
    "PostgresOptions",
    "create_postgres_backend",
    "register_postgres_backend",
]
