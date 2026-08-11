"""SQLite adapter for the backend-neutral structured database contract.

This module contains only generic table, record, query, and transaction
mechanics. It never imports Skill persistence models or vector-store types.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import math
import re
import sqlite3
import threading
from collections.abc import Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, AsyncIterator
from uuid import UUID

from ..database import DatabaseUnitOfWork, ScopedDatabase
from ..models import (
    DatabaseCapabilities,
    FieldSpec,
    FieldType,
    FilterExpression,
    FilterGroup,
    Predicate,
    Record,
    RecordQuery,
    SchemaMigration,
    Sort,
    TableSpec,
)
from ..registry import DatabaseRegistry, TableRegistry
from ..scope import DatabaseScope

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_RESERVED_COLUMNS = frozenset({"_scope_key", "_scope", "_record_id", "_schema_fingerprint"})
_SCHEMA_TABLE = "__mindmemos_schema"
_MIGRATION_TABLE = "__mindmemos_migrations"


@dataclass(frozen=True, slots=True, kw_only=True)
class SqliteOptions:
    """Connection settings accepted by the ``sqlite`` backend factory."""

    path: str = ":memory:"
    timeout: float = 30.0
    uri: bool = False
    create_parent_dirs: bool = True
    journal_mode: str = "WAL"
    synchronous: str = "NORMAL"
    foreign_keys: bool = True
    busy_timeout_ms: int = 30_000

    @classmethod
    def from_mapping(cls, options: Mapping[str, Any]) -> "SqliteOptions":
        allowed = set(cls.__dataclass_fields__)
        unknown = set(options) - allowed
        if unknown:
            raise ValueError(f"unknown sqlite backend options: {', '.join(sorted(unknown))}")
        result = cls(**dict(options))
        if not result.path:
            raise ValueError("sqlite backend option 'path' must not be empty")
        if result.timeout <= 0:
            raise ValueError("sqlite backend option 'timeout' must be positive")
        if result.busy_timeout_ms < 0:
            raise ValueError("sqlite backend option 'busy_timeout_ms' must not be negative")
        if result.journal_mode.upper() not in {"DELETE", "TRUNCATE", "PERSIST", "MEMORY", "WAL", "OFF"}:
            raise ValueError("unsupported sqlite journal_mode")
        if result.synchronous.upper() not in {"OFF", "NORMAL", "FULL", "EXTRA"}:
            raise ValueError("unsupported sqlite synchronous mode")
        return result


class SqliteBackend(ScopedDatabase):
    """SQLite adapter implementing scoped record CRUD, filtering, and paging."""

    _capabilities = DatabaseCapabilities(
        metadata_filtering=True,
        batch_record_io=True,
        atomic_batch_write=True,
        transactions=True,
        compare_and_swap=True,
    )

    def __init__(self, *, options: SqliteOptions, tables: TableRegistry) -> None:
        self._options = options
        self._tables = {spec.name: spec for spec in tables.specs}
        self._migrations = tables.migrations
        self._validate_registry()
        self._connection: sqlite3.Connection | None = None
        self._closed = False
        self._connection_lock = threading.RLock()
        self._open_lock = asyncio.Lock()
        self._operation_lock = asyncio.Lock()

    @property
    def name(self) -> str:
        return "sqlite"

    @property
    def capabilities(self) -> DatabaseCapabilities:
        return self._capabilities

    async def ensure_schema(self, tables: TableRegistry) -> None:
        requested = {spec.name: spec for spec in tables.specs}
        if requested != self._tables or tables.migrations != self._migrations:
            raise ValueError("ensure_schema registry differs from the registry used to construct the sqlite backend")
        await self._ensure_open()
        async with self._operation_lock:
            await asyncio.to_thread(self._ensure_schema_sync)

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[DatabaseUnitOfWork]:
        """Open one SQLite ``BEGIN IMMEDIATE`` unit of work.

        The yielded object must be used for all operations inside the context.
        SQLite's database lock, rather than the in-process asyncio lock, is the
        cross-process serialization boundary.
        """

        await self._ensure_open()
        await self._operation_lock.acquire()
        try:
            await asyncio.to_thread(self._begin_transaction_sync)
            unit_of_work = _SqliteUnitOfWork(self)
            try:
                yield unit_of_work
            except BaseException:
                await asyncio.to_thread(self._rollback_transaction_sync)
                raise
            else:
                try:
                    await asyncio.to_thread(self._commit_transaction_sync)
                except BaseException:
                    await asyncio.to_thread(self._rollback_transaction_sync)
                    raise
        finally:
            self._operation_lock.release()

    async def upsert_records(self, table: str, records: Sequence[Record]) -> None:
        await self._upsert_records(table, records, transactional=False)

    async def _upsert_records(
        self,
        table: str,
        records: Sequence[Record],
        *,
        transactional: bool,
    ) -> None:
        if not records:
            return
        spec = self._table(table)
        rows = [self._prepare_record(spec, record) for record in records]
        columns = ["_scope_key", "_scope", "_record_id", *[field.name for field in spec.fields]]
        conflict = ["_scope_key", "_record_id"] if spec.scope_scoped else ["_record_id"]
        updates = ["_scope", *[field.name for field in spec.fields]]
        statement = (
            f"INSERT INTO {_quote(spec.name)} ({', '.join(_quote(item) for item in columns)}) "
            f"VALUES ({', '.join('?' for _ in columns)}) "
            f"ON CONFLICT ({', '.join(_quote(item) for item in conflict)}) DO UPDATE SET "
            + ", ".join(f"{_quote(item)} = excluded.{_quote(item)}" for item in updates)
        )
        await self._execute_write(
            lambda connection: connection.executemany(statement, rows),
            transactional=transactional,
        )

    async def get_records(
        self,
        table: str,
        scope: DatabaseScope,
        record_ids: Sequence[str],
    ) -> list[Record]:
        return await self._get_records(table, scope, record_ids, transactional=False)

    async def _get_records(
        self,
        table: str,
        scope: DatabaseScope,
        record_ids: Sequence[str],
        *,
        transactional: bool,
    ) -> list[Record]:
        if not record_ids:
            return []
        spec = self._table(table)
        placeholders = ", ".join("?" for _ in record_ids)
        identity = "_scope_key = ? AND " if spec.scope_scoped else ""
        statement = f"SELECT * FROM {_quote(spec.name)} WHERE {identity}_record_id IN ({placeholders})"
        params: list[Any] = []
        if spec.scope_scoped:
            params.append(_scope_key(scope))
        params.extend(record_ids)
        rows = await self._execute_fetchall(statement, params, transactional=transactional)
        by_id = {row["_record_id"]: row for row in rows}
        return [self._row_to_record(spec, by_id[record_id]) for record_id in record_ids if record_id in by_id]

    async def patch_record(
        self,
        table: str,
        scope: DatabaseScope,
        record_id: str,
        changes: Mapping[str, Any],
    ) -> None:
        await self._patch_record(table, scope, record_id, changes, transactional=False)

    async def _patch_record(
        self,
        table: str,
        scope: DatabaseScope,
        record_id: str,
        changes: Mapping[str, Any],
        *,
        transactional: bool,
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
        assignments: list[str] = []
        params: list[Any] = []
        for name, value in changes.items():
            field = fields[name]
            if value is None and not field.nullable:
                raise ValueError(f"field {name!r} on table {table!r} is not nullable")
            assignments.append(f"{_quote(name)} = ?")
            params.append(_adapt_field_value(field, value))
        identity = "_scope_key = ? AND " if spec.scope_scoped else ""
        statement = f"UPDATE {_quote(spec.name)} SET {', '.join(assignments)} WHERE {identity}_record_id = ?"
        if spec.scope_scoped:
            params.append(_scope_key(scope))
        params.append(record_id)
        await self._execute_write(
            lambda connection: connection.execute(statement, params),
            transactional=transactional,
        )

    async def compare_and_swap_record(
        self,
        table: str,
        scope: DatabaseScope,
        record_id: str,
        *,
        expected: Mapping[str, Any],
        changes: Mapping[str, Any],
    ) -> bool:
        return await self._compare_and_swap_record(
            table,
            scope,
            record_id,
            expected=expected,
            changes=changes,
            transactional=False,
        )

    async def _compare_and_swap_record(
        self,
        table: str,
        scope: DatabaseScope,
        record_id: str,
        *,
        expected: Mapping[str, Any],
        changes: Mapping[str, Any],
        transactional: bool,
    ) -> bool:
        if not expected:
            raise ValueError("compare-and-swap requires at least one expected field")
        if not changes:
            raise ValueError("compare-and-swap requires at least one changed field")
        spec = self._table(table)
        fields = {field.name: field for field in spec.fields}
        unknown = (set(expected) | set(changes)) - set(fields)
        if unknown:
            raise ValueError(f"unknown fields for table {table!r}: {', '.join(sorted(unknown))}")
        if spec.primary_key in expected or spec.primary_key in changes:
            raise ValueError(f"cannot compare or patch primary key {spec.primary_key!r}")

        assignments: list[str] = []
        params: list[Any] = []
        for name, value in changes.items():
            field = fields[name]
            if value is None and not field.nullable:
                raise ValueError(f"field {name!r} on table {table!r} is not nullable")
            assignments.append(f"{_quote(name)} = ?")
            params.append(_adapt_field_value(field, value))

        identity = "_scope_key = ? AND " if spec.scope_scoped else ""
        conditions = [f"{_quote(name)} IS ?" for name in expected]
        statement = (
            f"UPDATE {_quote(spec.name)} SET {', '.join(assignments)} "
            f"WHERE {identity}_record_id = ? AND {' AND '.join(conditions)}"
        )
        if spec.scope_scoped:
            params.append(_scope_key(scope))
        params.append(record_id)
        params.extend(_adapt_field_value(fields[name], value) for name, value in expected.items())

        def execute(connection: sqlite3.Connection) -> bool:
            return connection.execute(statement, params).rowcount == 1

        return await self._execute_write(execute, transactional=transactional)

    async def delete_records(self, table: str, scope: DatabaseScope, record_ids: Sequence[str]) -> None:
        await self._delete_records(table, scope, record_ids, transactional=False)

    async def _delete_records(
        self,
        table: str,
        scope: DatabaseScope,
        record_ids: Sequence[str],
        *,
        transactional: bool,
    ) -> None:
        if not record_ids:
            return
        spec = self._table(table)
        placeholders = ", ".join("?" for _ in record_ids)
        identity = "_scope_key = ? AND " if spec.scope_scoped else ""
        statement = f"DELETE FROM {_quote(spec.name)} WHERE {identity}_record_id IN ({placeholders})"
        params: list[Any] = []
        if spec.scope_scoped:
            params.append(_scope_key(scope))
        params.extend(record_ids)
        await self._execute_write(
            lambda connection: connection.execute(statement, params),
            transactional=transactional,
        )

    async def query_records(self, table: str, query: RecordQuery) -> tuple[list[Record], str | None]:
        return await self._query_records(table, query, transactional=False)

    async def close(self) -> None:
        if self._closed:
            return
        async with self._operation_lock:
            self._closed = True
            connection = self._connection
            self._connection = None
            if connection is not None:
                await asyncio.to_thread(connection.close)

    async def _query_records(
        self,
        table: str,
        query: RecordQuery,
        *,
        transactional: bool,
    ) -> tuple[list[Record], str | None]:
        spec = self._table(table)
        offset = _decode_cursor(query.page.cursor, table)
        where, params = self._where_clause(spec, query.scope, query.filters)
        order, order_params = self._order_clause(spec, query.sort)
        statement = f"SELECT * FROM {_quote(spec.name)} WHERE {where} ORDER BY {order} LIMIT ? OFFSET ?"
        limit = query.page.limit
        rows = await self._execute_fetchall(
            statement,
            [*params, *order_params, limit + 1, offset],
            transactional=transactional,
        )
        has_more = len(rows) > limit
        records = [self._row_to_record(spec, row) for row in rows[:limit]]
        cursor = _encode_cursor(table, offset + limit) if has_more else None
        return records, cursor

    async def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("sqlite backend is closed")
        if self._connection is not None:
            return
        async with self._open_lock:
            if self._connection is not None:
                return
            if self._options.create_parent_dirs and self._options.path != ":memory:" and not self._options.uri:
                Path(self._options.path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
            self._connection = await asyncio.to_thread(self._open_connection)

    def _open_connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self._options.path,
            timeout=self._options.timeout,
            uri=self._options.uri,
            check_same_thread=False,
        )
        connection.row_factory = sqlite3.Row
        connection.create_function("mindmemos_json_contains", 2, _sqlite_json_contains, deterministic=True)
        connection.execute(f"PRAGMA busy_timeout = {self._options.busy_timeout_ms}")
        connection.execute(f"PRAGMA journal_mode = {self._options.journal_mode.upper()}")
        connection.execute(f"PRAGMA synchronous = {self._options.synchronous.upper()}")
        connection.execute(f"PRAGMA foreign_keys = {'ON' if self._options.foreign_keys else 'OFF'}")
        return connection

    def _ensure_schema_sync(self) -> None:
        connection = self._require_connection()
        with self._connection_lock:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    f"CREATE TABLE IF NOT EXISTS {_quote(_SCHEMA_TABLE)} ("
                    "table_name TEXT PRIMARY KEY, schema_fingerprint TEXT NOT NULL)"
                )
                connection.execute(
                    f"CREATE TABLE IF NOT EXISTS {_quote(_MIGRATION_TABLE)} ("
                    "namespace TEXT NOT NULL, version INTEGER NOT NULL, name TEXT NOT NULL, "
                    "checksum TEXT NOT NULL, applied_at TEXT NOT NULL, PRIMARY KEY (namespace, version))"
                )
                if self._migrations:
                    self._apply_migrations_sync(connection)
                else:
                    for spec in self._tables.values():
                        self._ensure_table_sync(connection, spec)
            except BaseException:
                connection.rollback()
                raise
            else:
                connection.commit()

    def _apply_migrations_sync(self, connection: sqlite3.Connection) -> None:
        by_namespace: dict[str, list[SchemaMigration]] = {}
        covered_tables: set[str] = set()
        for migration in self._migrations:
            by_namespace.setdefault(migration.namespace, []).append(migration)
            covered_tables.update(migration.tables)

        missing = set(self._tables) - covered_tables
        if missing:
            raise RuntimeError(f"registered tables are missing an explicit schema migration: {sorted(missing)}")

        ledger_rows = connection.execute(
            f"SELECT namespace, version, name, checksum FROM {_quote(_MIGRATION_TABLE)}"
        ).fetchall()
        registered = {(item.namespace, item.version): item for item in self._migrations}
        unexpected = sorted(
            (str(row["namespace"]), int(row["version"]))
            for row in ledger_rows
            if (str(row["namespace"]), int(row["version"])) not in registered
        )
        if unexpected:
            raise RuntimeError(f"sqlite schema ledger contains unknown migrations: {unexpected}")

        applied: set[tuple[str, int]] = set()
        for row in ledger_rows:
            identity = (str(row["namespace"]), int(row["version"]))
            migration = registered[identity]
            checksum = self._migration_checksum(migration)
            if row["name"] != migration.name or row["checksum"] != checksum:
                raise RuntimeError(
                    f"sqlite migration definition drift for {migration.namespace!r} version {migration.version}"
                )
            applied.add(identity)

        for namespace, migrations in by_namespace.items():
            applied_versions = sorted(
                version for applied_namespace, version in applied if applied_namespace == namespace
            )
            expected_prefix = list(range(1, len(applied_versions) + 1))
            if applied_versions != expected_prefix:
                raise RuntimeError(
                    f"sqlite schema ledger for namespace {namespace!r} is not a contiguous prefix: {applied_versions}"
                )

        schema_row_count = int(connection.execute(f"SELECT COUNT(*) FROM {_quote(_SCHEMA_TABLE)}").fetchone()[0])
        existing_registered_tables = {
            name
            for name in self._tables
            if connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
                (name,),
            ).fetchone()
            is not None
        }
        fresh_database = not ledger_rows and schema_row_count == 0 and not existing_registered_tables
        if not ledger_rows and not fresh_database:
            raise RuntimeError(
                "sqlite database contains registered schema objects without a migration ledger; "
                "recreate this unpublished database or adopt it through an explicit migration"
            )

        if fresh_database:
            for spec in self._tables.values():
                self._ensure_table_sync(connection, spec)
            for migration in self._migrations:
                self._record_migration_sync(connection, migration)
            return

        touched_tables: set[str] = set()
        for migration in self._migrations:
            identity = (migration.namespace, migration.version)
            if identity in applied:
                continue
            for statement in migration.statements_for("sqlite"):
                connection.execute(statement)
            touched_tables.update(migration.tables)
            self._record_migration_sync(connection, migration)

        for spec in self._tables.values():
            self._ensure_table_sync(
                connection,
                spec,
                allow_migrated_fingerprint=spec.name in touched_tables,
            )

    def _record_migration_sync(self, connection: sqlite3.Connection, migration: SchemaMigration) -> None:
        connection.execute(
            f"INSERT INTO {_quote(_MIGRATION_TABLE)} "
            "(namespace, version, name, checksum, applied_at) VALUES (?, ?, ?, ?, ?)",
            (
                migration.namespace,
                migration.version,
                migration.name,
                self._migration_checksum(migration),
                datetime.now(UTC).isoformat(),
            ),
        )

    def _migration_checksum(self, migration: SchemaMigration) -> str:
        payload = {
            "namespace": migration.namespace,
            "version": migration.version,
            "name": migration.name,
            "tables": migration.tables,
            "sqlite_statements": migration.sqlite_statements,
            "postgres_statements": migration.postgres_statements,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _ensure_table_sync(
        self,
        connection: sqlite3.Connection,
        spec: TableSpec,
        *,
        allow_migrated_fingerprint: bool = False,
    ) -> None:
        fingerprint = _schema_fingerprint(spec)
        existing_table = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
            (spec.name,),
        ).fetchone()
        row = connection.execute(
            f"SELECT schema_fingerprint FROM {_quote(_SCHEMA_TABLE)} WHERE table_name = ?",
            (spec.name,),
        ).fetchone()
        if existing_table is not None and row is None and not allow_migrated_fingerprint:
            raise RuntimeError(
                f"sqlite table {spec.name!r} exists without MindMemOS schema metadata; "
                "adopt it through an explicit database migration"
            )
        if row is not None and row["schema_fingerprint"] != fingerprint and not allow_migrated_fingerprint:
            raise RuntimeError(
                f"sqlite schema drift for table {spec.name!r}; use an explicit database migration "
                "instead of changing a registered TableSpec in place"
            )
        connection.execute(self._table_ddl(spec))
        for statement in self._index_ddl(spec):
            connection.execute(statement)
        self._validate_physical_table_sync(connection, spec)
        connection.execute(
            f"INSERT INTO {_quote(_SCHEMA_TABLE)} (table_name, schema_fingerprint) VALUES (?, ?) "
            "ON CONFLICT(table_name) DO UPDATE SET schema_fingerprint = excluded.schema_fingerprint",
            (spec.name, fingerprint),
        )

    def _validate_physical_table_sync(self, connection: sqlite3.Connection, spec: TableSpec) -> None:
        rows = connection.execute(f"PRAGMA table_info({_quote(spec.name)})").fetchall()
        expected_columns = {
            "_scope_key": ("TEXT", True, 1 if spec.scope_scoped else 0),
            "_scope": ("TEXT", True, 0),
            "_record_id": ("TEXT", True, 2 if spec.scope_scoped else 1),
            **{field.name: (_SQLITE_TYPES[field.field_type], not field.nullable, 0) for field in spec.fields},
        }
        actual_columns = {
            str(row["name"]): (str(row["type"]).upper(), bool(row["notnull"]), int(row["pk"])) for row in rows
        }
        if actual_columns != expected_columns:
            raise RuntimeError(
                f"sqlite physical schema for table {spec.name!r} does not match the registered TableSpec"
            )

        actual_indexes: dict[str, tuple[bool, tuple[str, ...]]] = {}
        for row in connection.execute(f"PRAGMA index_list({_quote(spec.name)})").fetchall():
            name = str(row["name"])
            if name.startswith("sqlite_autoindex_"):
                continue
            fields = tuple(
                str(item["name"]) for item in connection.execute(f"PRAGMA index_info({_quote(name)})").fetchall()
            )
            actual_indexes[name] = (bool(row["unique"]), fields)
        expected_indexes = {}
        for index in spec.indexes:
            fields = [*(["_scope_key"] if spec.scope_scoped else []), *index.fields]
            expected_indexes[index.name] = (
                index.unique,
                tuple("_record_id" if field == spec.primary_key else field for field in fields),
            )
        if actual_indexes != expected_indexes:
            raise RuntimeError(f"sqlite physical indexes for table {spec.name!r} do not match the registered TableSpec")

    def _table_ddl(self, spec: TableSpec) -> str:
        columns = ["_scope_key TEXT NOT NULL", "_scope TEXT NOT NULL", "_record_id TEXT NOT NULL"]
        for field in spec.fields:
            declaration = f"{_quote(field.name)} {_SQLITE_TYPES[field.field_type]}"
            if not field.nullable:
                declaration += " NOT NULL"
            columns.append(declaration)
        primary = "PRIMARY KEY (_scope_key, _record_id)" if spec.scope_scoped else "PRIMARY KEY (_record_id)"
        columns.append(primary)
        return f"CREATE TABLE IF NOT EXISTS {_quote(spec.name)} ({', '.join(columns)})"

    def _index_ddl(self, spec: TableSpec) -> list[str]:
        statements: list[str] = []
        for index in spec.indexes:
            columns = [*(["_scope_key"] if spec.scope_scoped else []), *index.fields]
            translated = ["_record_id" if item == spec.primary_key else item for item in columns]
            unique = "UNIQUE " if index.unique else ""
            statements.append(
                f"CREATE {unique}INDEX IF NOT EXISTS {_quote(index.name)} ON {_quote(spec.name)} "
                f"({', '.join(_quote(item) for item in translated)})"
            )
        return statements

    async def _execute_write(self, operation: Any, *, transactional: bool) -> Any:
        if transactional:
            return await asyncio.to_thread(self._run_transaction_operation_sync, operation)
        return await self._run_write(operation)

    async def _execute_fetchall(
        self,
        statement: str,
        params: Sequence[Any],
        *,
        transactional: bool,
    ) -> list[sqlite3.Row]:
        if transactional:
            return await asyncio.to_thread(self._fetchall_transaction_sync, statement, params)
        return await self._fetchall(statement, params)

    async def _run_write(self, operation: Any) -> Any:
        await self._ensure_open()

        def run() -> Any:
            connection = self._require_connection()
            with self._connection_lock, connection:
                return operation(connection)

        async with self._operation_lock:
            return await asyncio.to_thread(run)

    async def _fetchall(self, statement: str, params: Sequence[Any]) -> list[sqlite3.Row]:
        await self._ensure_open()

        def run() -> list[sqlite3.Row]:
            connection = self._require_connection()
            with self._connection_lock:
                return list(connection.execute(statement, tuple(params)).fetchall())

        async with self._operation_lock:
            return await asyncio.to_thread(run)

    def _run_transaction_operation_sync(self, operation: Any) -> Any:
        connection = self._require_connection()
        if not connection.in_transaction:
            raise RuntimeError("sqlite unit of work is no longer active")
        with self._connection_lock:
            return operation(connection)

    def _fetchall_transaction_sync(self, statement: str, params: Sequence[Any]) -> list[sqlite3.Row]:
        return self._run_transaction_operation_sync(
            lambda connection: list(connection.execute(statement, tuple(params)).fetchall())
        )

    def _begin_transaction_sync(self) -> None:
        connection = self._require_connection()
        with self._connection_lock:
            if connection.in_transaction:
                raise RuntimeError("sqlite connection already has an active transaction")
            connection.execute("BEGIN IMMEDIATE")

    def _commit_transaction_sync(self) -> None:
        connection = self._require_connection()
        with self._connection_lock:
            connection.commit()

    def _rollback_transaction_sync(self) -> None:
        connection = self._require_connection()
        with self._connection_lock:
            connection.rollback()

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
        values: list[Any] = [_scope_key(record.scope), _json_dump(dict(record.scope.items())), record.record_id]
        for field in spec.fields:
            value = payload.get(field.name, field.default)
            if value is None and not field.nullable:
                raise ValueError(f"field {field.name!r} on table {spec.name!r} is required")
            values.append(_adapt_field_value(field, value))
        return tuple(values)

    def _row_to_record(self, spec: TableSpec, row: Mapping[str, Any]) -> Record:
        payload = {field.name: _restore_field_value(field, row[field.name]) for field in spec.fields}
        return Record(
            table=spec.name,
            record_id=row["_record_id"],
            scope=DatabaseScope(json.loads(row["_scope"])),
            payload=payload,
        )

    def _where_clause(
        self,
        spec: TableSpec,
        scope: DatabaseScope,
        filters: FilterExpression | None,
    ) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        for name, value in scope.items():
            clauses.append("json_extract(_scope, ?) = ?")
            params.extend((_json_path(name), value))
        if filters is not None:
            expression, expression_params = self._compile_filter(spec, filters)
            clauses.append(f"({expression})")
            params.extend(expression_params)
        return (" AND ".join(clauses) if clauses else "1"), params

    def _compile_filter(self, spec: TableSpec, expression: FilterExpression) -> tuple[str, list[Any]]:
        if isinstance(expression, FilterGroup):
            compiled = [self._compile_filter(spec, clause) for clause in expression.clauses]
            if expression.operator == "not":
                if not compiled:
                    return "1", []
                return "NOT (" + " OR ".join(f"({sql})" for sql, _ in compiled) + ")", [
                    value for _, values in compiled for value in values
                ]
            if not compiled:
                return ("1", []) if expression.operator == "and" else ("0", [])
            joiner = " AND " if expression.operator == "and" else " OR "
            return joiner.join(f"({sql})" for sql, _ in compiled), [value for _, values in compiled for value in values]
        return self._compile_predicate(spec, expression)

    def _compile_predicate(self, spec: TableSpec, predicate: Predicate) -> tuple[str, list[Any]]:
        field, field_params, field_type, nested = self._resolve_field(spec, predicate.field)
        value = predicate.value
        if predicate.op == "is_null":
            expected = True if value is None else bool(value)
            return f"{field} IS {'NULL' if expected else 'NOT NULL'}", field_params
        if predicate.op == "is_empty":
            if nested or field_type == FieldType.TEXT:
                return f"({field} IS NULL OR {field} = '')", [*field_params, *field_params]
            if field_type in {FieldType.TEXT_ARRAY, FieldType.UUID_ARRAY, FieldType.JSON}:
                return (
                    f"({field} IS NULL OR {field} IN ('[]', '{{}}', 'null', '\"\"'))",
                    [*field_params, *field_params],
                )
            return f"{field} IS NULL", field_params
        if predicate.op in {"in", "not_in"}:
            if not isinstance(value, (list, tuple, set, frozenset)):
                raise TypeError(f"filter operator {predicate.op!r} requires a sequence value")
            values = [_adapt_comparison_value(field_type, nested, item) for item in value]
            if not values:
                return ("0", []) if predicate.op == "in" else ("1", [])
            operator = "IN" if predicate.op == "in" else "NOT IN"
            return f"{field} {operator} ({', '.join('?' for _ in values)})", [*field_params, *values]
        if predicate.op in {"eq", "ne", "gt", "gte", "lt", "lte"}:
            operator = {"eq": "IS", "ne": "IS NOT", "gt": ">", "gte": ">=", "lt": "<", "lte": "<="}[predicate.op]
            return f"{field} {operator} ?", [
                *field_params,
                _adapt_comparison_value(field_type, nested, value),
            ]
        if predicate.op in {"contains", "icontains"}:
            if nested or field_type == FieldType.TEXT:
                if predicate.op == "icontains":
                    return f"LOWER({field}) LIKE LOWER(?)", [*field_params, f"%{value}%"]
                return f"{field} LIKE ?", [*field_params, f"%{value}%"]
            if field_type in {FieldType.TEXT_ARRAY, FieldType.UUID_ARRAY}:
                if predicate.op == "icontains":
                    return f"EXISTS (SELECT 1 FROM json_each({field}) WHERE LOWER(value) = LOWER(?))", [
                        *field_params,
                        str(value),
                    ]
                return f"EXISTS (SELECT 1 FROM json_each({field}) WHERE value = ?)", [
                    *field_params,
                    _adapt_array_member(field_type, value),
                ]
            if field_type == FieldType.JSON:
                if predicate.op == "icontains":
                    return f"EXISTS (SELECT 1 FROM json_each({field}) WHERE LOWER(value) = LOWER(?))", [
                        *field_params,
                        str(value),
                    ]
                return f"mindmemos_json_contains({field}, ?) = 1", [*field_params, _json_dump(value)]
            raise ValueError(f"operator {predicate.op!r} is not supported for field type {field_type.value!r}")
        raise ValueError(f"unsupported filter operator {predicate.op!r}")

    def _resolve_field(self, spec: TableSpec, name: str) -> tuple[str, list[Any], FieldType, bool]:
        fields = {field.name: field for field in spec.fields}
        if name in fields:
            return _quote(name), [], fields[name].field_type, False
        if name == spec.primary_key:
            return "_record_id", [], FieldType.TEXT, False
        root, separator, path = name.partition(".")
        if separator and root in fields and fields[root].field_type == FieldType.JSON:
            return f"json_extract({_quote(root)}, ?)", [_json_path(*path.split("."))], FieldType.JSON, True
        raise ValueError(f"unknown filter/sort field {name!r} for table {spec.name!r}")

    def _order_clause(self, spec: TableSpec, sorts: Sequence[Sort]) -> tuple[str, list[Any]]:
        expressions: list[str] = []
        params: list[Any] = []
        for item in sorts:
            field, field_params, _, _ = self._resolve_field(spec, item.field)
            expressions.append(f"{field} {'DESC' if item.direction == 'desc' else 'ASC'} NULLS LAST")
            params.extend(field_params)
        if spec.scope_scoped:
            expressions.append("_scope_key ASC")
        expressions.append("_record_id ASC")
        return ", ".join(expressions), params

    def _table(self, name: str) -> TableSpec:
        try:
            return self._tables[name]
        except KeyError as exc:
            raise KeyError(f"unknown logical table {name!r}") from exc

    def _validate_registry(self) -> None:
        for spec in self._tables.values():
            _validate_identifier(spec.name)
            _validate_identifier(spec.primary_key)
            names = {field.name for field in spec.fields}
            reserved = names & _RESERVED_COLUMNS
            if reserved:
                raise ValueError(f"table {spec.name!r} uses reserved sqlite columns: {sorted(reserved)}")
            for field in spec.fields:
                _validate_identifier(field.name)
            for index in spec.indexes:
                _validate_identifier(index.name)

    def _require_connection(self) -> sqlite3.Connection:
        if self._connection is None:
            raise RuntimeError("sqlite backend is not open")
        return self._connection


class _SqliteUnitOfWork(DatabaseUnitOfWork):
    """Transaction-bound view over one :class:`SqliteBackend` connection."""

    def __init__(self, backend: SqliteBackend) -> None:
        self._backend = backend

    async def upsert_records(self, table: str, records: Sequence[Record]) -> None:
        await self._backend._upsert_records(table, records, transactional=True)

    async def get_records(
        self,
        table: str,
        scope: DatabaseScope,
        record_ids: Sequence[str],
    ) -> list[Record]:
        return await self._backend._get_records(table, scope, record_ids, transactional=True)

    async def patch_record(
        self,
        table: str,
        scope: DatabaseScope,
        record_id: str,
        changes: Mapping[str, Any],
    ) -> None:
        await self._backend._patch_record(table, scope, record_id, changes, transactional=True)

    async def compare_and_swap_record(
        self,
        table: str,
        scope: DatabaseScope,
        record_id: str,
        *,
        expected: Mapping[str, Any],
        changes: Mapping[str, Any],
    ) -> bool:
        return await self._backend._compare_and_swap_record(
            table,
            scope,
            record_id,
            expected=expected,
            changes=changes,
            transactional=True,
        )

    async def delete_records(self, table: str, scope: DatabaseScope, record_ids: Sequence[str]) -> None:
        await self._backend._delete_records(table, scope, record_ids, transactional=True)

    async def query_records(self, table: str, query: RecordQuery) -> tuple[list[Record], str | None]:
        return await self._backend._query_records(table, query, transactional=True)


_SQLITE_TYPES = {
    FieldType.UUID: "TEXT",
    FieldType.TEXT: "TEXT",
    FieldType.INTEGER: "INTEGER",
    FieldType.FLOAT: "REAL",
    FieldType.BOOLEAN: "INTEGER",
    FieldType.DATETIME: "TEXT",
    FieldType.TEXT_ARRAY: "TEXT",
    FieldType.UUID_ARRAY: "TEXT",
    FieldType.JSON: "TEXT",
}


def create_sqlite_backend(options: Mapping[str, Any], tables: TableRegistry) -> SqliteBackend:
    """Backend factory compatible with :class:`DatabaseRegistry`."""

    return SqliteBackend(options=SqliteOptions.from_mapping(options), tables=tables)


def register_sqlite_backend(registry: DatabaseRegistry) -> None:
    """Register the stable ``sqlite`` provider name."""

    registry.register("sqlite", create_sqlite_backend)


def _scope_key(scope: DatabaseScope) -> str:
    return _json_dump(dict(scope.items()))


def _json_dump(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=_json_default,
    )


def _json_default(value: Any) -> str:
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, datetime):
        return _normalize_datetime(value)
    raise TypeError(f"value of type {type(value).__name__} is not JSON serializable")


def _adapt_field_value(field: FieldSpec, value: Any) -> Any:
    if value is None:
        return None
    context = f"field {field.name!r}"
    if field.field_type == FieldType.TEXT:
        if not isinstance(value, str):
            raise TypeError(f"{context} requires a string value")
        return value
    if field.field_type == FieldType.INTEGER:
        if type(value) is not int:
            raise TypeError(f"{context} requires an integer value")
        return value
    if field.field_type == FieldType.FLOAT:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{context} requires a numeric value")
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"{context} requires a finite numeric value")
        return result
    if field.field_type == FieldType.BOOLEAN:
        if type(value) is not bool:
            raise TypeError(f"{context} requires a boolean value")
        return int(value)
    if field.field_type == FieldType.UUID:
        return str(_coerce_uuid(value, context=context))
    if field.field_type == FieldType.DATETIME:
        return _normalize_datetime(_coerce_datetime(value, context=context))
    if field.field_type == FieldType.TEXT_ARRAY:
        if not isinstance(value, (list, tuple)) or any(not isinstance(item, str) for item in value):
            raise TypeError(f"{context} requires an array of strings")
        return _json_dump(list(value))
    if field.field_type == FieldType.UUID_ARRAY:
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"{context} requires an array of UUID values")
        return _json_dump([str(_coerce_uuid(item, context=context)) for item in value])
    if field.field_type == FieldType.JSON:
        return _json_dump(value)
    raise ValueError(f"unsupported field type {field.field_type!r}")


def _restore_field_value(field: FieldSpec, value: Any) -> Any:
    if value is None:
        return None
    if field.field_type == FieldType.BOOLEAN:
        return bool(value)
    if field.field_type == FieldType.UUID:
        return UUID(value)
    if field.field_type == FieldType.DATETIME:
        return datetime.fromisoformat(value)
    if field.field_type in {FieldType.TEXT_ARRAY, FieldType.UUID_ARRAY, FieldType.JSON}:
        restored = json.loads(value)
        if field.field_type == FieldType.UUID_ARRAY:
            return [UUID(item) for item in restored]
        return restored
    return value


def _adapt_comparison_value(field_type: FieldType, nested: bool, value: Any) -> Any:
    if value is None:
        return None
    if nested:
        return value
    comparison_field = FieldSpec(name="<comparison>", field_type=field_type)
    return _adapt_field_value(comparison_field, value)


def _adapt_array_member(field_type: FieldType, value: Any) -> str:
    if field_type == FieldType.TEXT_ARRAY:
        if not isinstance(value, str):
            raise TypeError("TEXT_ARRAY contains requires a string value")
        return value
    if field_type == FieldType.UUID_ARRAY:
        return str(_coerce_uuid(value, context="UUID_ARRAY contains"))
    raise ValueError(f"unsupported array field type {field_type!r}")


def _coerce_uuid(value: Any, *, context: str) -> UUID:
    if isinstance(value, UUID):
        return value
    if isinstance(value, str):
        try:
            return UUID(value)
        except ValueError as exc:
            raise ValueError(f"{context} requires a valid UUID value") from exc
    raise TypeError(f"{context} requires a UUID or UUID string")


def _coerce_datetime(value: Any, *, context: str) -> datetime:
    if isinstance(value, datetime):
        result = value
    elif isinstance(value, str):
        try:
            result = datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"{context} requires a valid ISO-8601 datetime") from exc
    else:
        raise TypeError(f"{context} requires a datetime or ISO-8601 string")
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError(f"{context} requires a timezone-aware datetime")
    return result


def _normalize_datetime(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("datetime values must include timezone information")
    return value.astimezone(UTC).isoformat()


def _sqlite_json_contains(document: str | None, expected: str | None) -> int:
    if document is None or expected is None:
        return 0
    try:
        actual_value = json.loads(document)
        expected_value = json.loads(expected)
    except (TypeError, json.JSONDecodeError):
        return 0
    return int(_json_value_contains(actual_value, expected_value))


def _json_value_contains(actual: Any, expected: Any) -> bool:
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and _json_value_contains(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return isinstance(actual, list) and all(
            any(_json_value_contains(candidate, item) for candidate in actual) for item in expected
        )
    if isinstance(actual, list):
        return any(_json_value_contains(item, expected) for item in actual)
    if isinstance(actual, bool) or isinstance(expected, bool):
        return type(actual) is type(expected) and actual == expected
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return actual == expected
    return type(actual) is type(expected) and actual == expected


def _schema_fingerprint(spec: TableSpec) -> str:
    payload = json.dumps(asdict(spec), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _encode_cursor(table: str, offset: int) -> str:
    raw = json.dumps({"table": table, "offset": offset}, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii")


def _decode_cursor(cursor: str | None, table: str) -> int:
    if cursor is None:
        return 0
    try:
        value = json.loads(base64.urlsafe_b64decode(cursor.encode("ascii")))
        if value.get("table") != table or not isinstance(value.get("offset"), int) or value["offset"] < 0:
            raise ValueError
        return value["offset"]
    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        raise ValueError("invalid or cross-table sqlite cursor") from exc


def _quote(identifier: str) -> str:
    _validate_identifier(identifier)
    return f'"{identifier}"'


def _validate_identifier(identifier: str) -> None:
    if not _IDENTIFIER.fullmatch(identifier):
        raise ValueError(f"invalid sqlite identifier: {identifier!r}")


def _json_path(*parts: str) -> str:
    return "$" + "".join(f'."{part.replace(chr(34), chr(34) * 2)}"' for part in parts)


__all__ = [
    "SqliteBackend",
    "SqliteOptions",
    "create_sqlite_backend",
    "register_sqlite_backend",
]
