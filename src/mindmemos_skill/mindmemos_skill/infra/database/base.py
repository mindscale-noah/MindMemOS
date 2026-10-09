"""Backend-independent operations on registered, validated tables."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from contextlib import contextmanager
from copy import copy
from datetime import timedelta
from itertools import groupby
from typing import Any

from sqlalchemy import Date, UniqueConstraint, and_, case, cast, func, or_, select

from ..base import Infrastructure
from .tables import TableRegistry
from .typing import DailyCount, Scope, SearchHit
from .validation import column_for, filter_clause, validate_batch
from .vector import Vector


class Database(Infrastructure, ABC):
    """Share CRUD and validation while delegating engine/vector details.

    Args:
        config: Validated backend-specific connection settings.
        tables: Instance-scoped business table definitions.
    """

    def __init__(self, config, *, tables: TableRegistry | None = None):
        super().__init__(config)
        self.tables = tables if tables is not None else TableRegistry()
        self.engine = self._create_engine(config)
        self._connection = None
        self._closed = False

    @abstractmethod
    def _create_engine(self, config):
        """Construct a lazy engine for this backend."""
        raise NotImplementedError

    @abstractmethod
    def _insert_statement(self, table):
        """Create the backend's conflict-aware insert statement."""
        raise NotImplementedError

    @abstractmethod
    def _distance(self, column, vector):
        """Build a bound cosine-distance expression."""
        raise NotImplementedError

    @contextmanager
    def _connect(self):
        if self._closed:
            raise RuntimeError("Database or transaction is closed")
        if self._connection is not None:
            if not self._connection.in_transaction():
                raise RuntimeError("Transaction is no longer active")
            with self._connection.begin_nested():
                yield self._connection
        else:
            with self.engine.begin() as connection:
                yield connection

    @contextmanager
    def transaction(self):
        """Bind operations to one connection and commit or roll back together.

        Yields:
            A transaction-scoped database handle; invalid after context exit.

        Raises:
            RuntimeError: Nested transactions or closed handles are used.
        """
        if self._connection is not None:
            raise RuntimeError("Nested transactions are not supported")
        with self._connect() as connection:
            transaction = copy(self)
            transaction._connection = connection
            try:
                yield transaction
            finally:
                transaction._closed = True

    @contextmanager
    def scoped(self, scope: Scope | Mapping[str, Any]):
        """Yield a field-bound handle and invalidate it on context exit.

        Args:
            scope: Immutable scope or arbitrary column-value mapping.

        Yields:
            Database operations restricted to the supplied scope.
        """
        from .scoped import ScopedDatabase

        validated = Scope(fields=scope) if isinstance(scope, Mapping) else Scope.model_validate(scope)
        handle = ScopedDatabase(self, validated)
        try:
            yield handle
        finally:
            handle._active = False

    def create_tables(self):
        """Explicitly create registered tables in foreign-key dependency order.

        Existing tables are left intact; schema migration is a deployment concern.
        """
        with self._connect() as connection:
            catalogs = {}
            for spec in self.tables.values():
                catalogs.setdefault(spec.table.metadata, []).append(spec.table)
            for metadata, tables in catalogs.items():
                # create_all adds named cycle-breaking foreign keys after their tables.
                metadata.create_all(connection, tables=tables, checkfirst=True)
                for table in tables:
                    for index in table.indexes:
                        index.create(connection, checkfirst=True)

    def close(self):
        """Release this database's pool; active transactions must finish first."""
        if self._connection is not None:
            raise RuntimeError("Use the transaction context to close its connection")
        self.engine.dispose()
        self._closed = True

    def insert(self, table: str, records: list[dict]) -> int:
        """Insert a fully validated batch atomically.

        Args:
            table: Registered table name.
            records: Records to insert; a single record is a one-element list.

        Returns:
            Number of inserted records. Constraint violations raise database errors.
        """
        spec = self.tables.get(table)
        rows = validate_batch(spec, records)
        with self._connect() as connection:
            # Adjacent groups preserve input order and database-default semantics.
            for _, group in groupby(rows, key=lambda row: tuple(sorted(row))):
                connection.execute(spec.table.insert(), list(group))
        return len(rows)

    def upsert(self, table: str, records: list[dict], *, conflict_fields=None, update_fields=None) -> int:
        """Atomically insert or update using an existing unique constraint.

        Args:
            table: Registered table name.
            records: Complete insertable records.
            conflict_fields: Unique key columns; defaults to the primary key.
            update_fields: Explicit columns to replace on conflict.

        Returns:
            Number of inserted or updated records (not skipped conflicts).
        """
        spec = self.tables.get(table)
        rows = validate_batch(spec, records)
        keys = list(conflict_fields) if conflict_fields is not None else list(spec.table.primary_key.columns.keys())
        unique_keys = [set(spec.table.primary_key.columns.keys())]
        unique_keys += [set(c.columns.keys()) for c in spec.table.constraints if isinstance(c, UniqueConstraint)]
        if not keys or len(set(keys)) != len(keys) or set(keys) not in unique_keys:
            raise ValueError("conflict_fields must identify a declared primary or unique constraint")
        updates = list(update_fields or [])
        for name in updates:
            column_for(spec, name)
        if any(not set(keys + updates) <= row.keys() for row in rows):
            raise ValueError("Each upsert record must supply conflict and update fields")
        count = 0
        with self._connect() as connection:
            for _, group in groupby(rows, key=lambda row: tuple(sorted(row))):
                statement = self._insert_statement(spec.table)
                if updates:
                    statement = statement.on_conflict_do_update(
                        index_elements=keys, set_={name: statement.excluded[name] for name in updates}
                    )
                else:
                    statement = statement.on_conflict_do_nothing(index_elements=keys)
                count += len(connection.execute(statement.returning(*spec.table.primary_key), list(group)).all())
        return count

    def update(self, table: str, values: dict, *, filters) -> int:
        """Apply the same partial values to all matching records.

        Args:
            table: Registered table name.
            values: Explicit fields to change; omitted fields are untouched.
            filters: Required conditions; an empty mapping explicitly targets all rows.

        Returns:
            Number of matched records, including unchanged values.
        """
        if filters is None:
            raise ValueError("Supply filters; use {} explicitly to update every row")
        spec = self.tables.get(table)
        row = validate_batch(spec, [values], partial=True)[0]
        if not row:
            raise ValueError("Updates must contain fields")
        statement = (
            spec.table.update().where(filter_clause(spec, filters)).values(row).returning(*spec.table.primary_key)
        )
        with self._connect() as connection:
            return len(connection.execute(statement).all())

    def update_many(self, table: str, records: list[dict]) -> int:
        """Apply per-record changes, locating rows by their complete primary key.

        Args:
            table: Registered table name.
            records: Primary key values together with fields to update.

        Returns:
            Total matched rows; absent keys are not inserted.
        """
        spec = self.tables.get(table)
        keys = set(spec.table.primary_key.columns.keys())
        prepared = []
        for index, record in enumerate(records):
            if not isinstance(record, dict) or not keys <= record.keys():
                raise ValueError(f"Table {table!r}, record {index}: missing primary key")
            identity = {key: record[key] for key in keys}
            if any(value is None for value in identity.values()):
                raise ValueError(f"Table {table!r}, record {index}: NULL primary key")
            condition = filter_clause(spec, identity)
            prepared.append((condition, {key: value for key, value in record.items() if key not in keys}))
        rows = validate_batch(spec, [row for _, row in prepared], partial=True)
        if any(not row for row in rows):
            raise ValueError("Each update must include at least one changed field")
        count = 0
        with self._connect() as connection:
            for (condition, _), row in zip(prepared, rows):
                statement = spec.table.update().where(condition).values(row).returning(*spec.table.primary_key)
                count += len(connection.execute(statement).all())
        return count

    def delete(self, table: str, *, filters) -> int:
        """Delete matching rows atomically.

        Args:
            table: Registered table name.
            filters: Required conditions; an empty mapping explicitly targets all rows.

        Returns:
            Number of deleted records.
        """
        if filters is None:
            raise ValueError("Supply filters; use {} explicitly to delete every row")
        spec = self.tables.get(table)
        statement = spec.table.delete().where(filter_clause(spec, filters)).returning(*spec.table.primary_key)
        with self._connect() as connection:
            return len(connection.execute(statement).all())

    def get(self, table: str, key):
        """Read a record by its primary key.

        Args:
            table: Registered table name.
            key: Scalar for a single key, or a mapping for a composite key.

        Returns:
            A record dictionary, or None if absent.
        """
        spec = self.tables.get(table)
        keys = list(spec.table.primary_key.columns.keys())
        if not isinstance(key, dict):
            if len(keys) != 1:
                raise ValueError("Composite primary keys require a mapping")
            key = {keys[0]: key}
        if set(key) != set(keys) or any(value is None for value in key.values()):
            raise ValueError("Supply exactly the non-null primary key fields")
        rows = self.select(table, filters=key, limit=1)
        return rows[0] if rows else None

    def select(
        self,
        table: str,
        *,
        filters=None,
        any_filters=None,
        fields=None,
        order_by=None,
        limit: int | None = 100,
        offset: int = 0,
    ):
        """Read records using deterministic ordering and bounded pagination.

        Args:
            table: Registered table name.
            filters: Equality mapping or AND-combined Filter objects.
            any_filters: Optional OR-combined Filter objects, AND-combined with filters.
            fields: Columns to return; None returns all columns.
            order_by: Column names, prefixed with '-' for descending order.
            limit: Maximum records to return, including zero; None has no limit.
            offset: Number of sorted records to skip.

        Returns:
            A list of record dictionaries with native Python values.
        """
        spec = self.tables.get(table)
        if (limit is not None and (type(limit) is not int or limit < 0)) or type(offset) is not int or offset < 0:
            raise ValueError("limit must be a nonnegative integer or None; offset must be a nonnegative integer")
        columns = self._columns(spec, fields)
        ordering = []
        for name in order_by or []:
            column = column_for(spec, name.removeprefix("-"))
            ordering.append(column.desc() if name.startswith("-") else column.asc())
        ordering.extend(spec.table.primary_key.columns)
        conditions = [filter_clause(spec, filters)]
        if any_filters is not None:
            choices = list(any_filters)
            if not choices:
                raise ValueError("any_filters must contain at least one condition")
            conditions.append(or_(*(filter_clause(spec, [condition]) for condition in choices)))
        statement = select(*columns).where(*conditions).order_by(*ordering).limit(limit).offset(offset)
        with self._connect() as connection:
            return [dict(row) for row in connection.execute(statement).mappings()]

    def select_distinct(self, table: str, field: str, *, filters=None) -> list[Any]:
        """Return distinct values of one registered column matching the filters.

        Args:
            table: Registered table name.
            field: Column whose unique values are returned.
            filters: Equality mapping or AND-combined Filter objects.

        Returns:
            Matching values without duplicates.
        """
        spec = self.tables.get(table)
        column = column_for(spec, field)
        statement = select(column).where(filter_clause(spec, filters)).distinct()
        with self._connect() as connection:
            return list(connection.execute(statement).scalars())

    def count(self, table: str, *, filters=None) -> int:
        """Count matching rows without fetching records.

        Args:
            table: Registered table name.
            filters: Equality mapping or AND-combined Filter objects.

        Returns:
            Number of matching records.
        """
        spec = self.tables.get(table)
        statement = select(func.count()).select_from(spec.table).where(filter_clause(spec, filters))
        with self._connect() as connection:
            return connection.execute(statement).scalar_one()

    def count_by_day(
        self,
        table: str,
        *,
        time_field: str,
        filters,
        matching_table: str,
        join_fields: list[str],
        matching_filters=None,
        utc_offset_minutes: int = 0,
    ) -> list[DailyCount]:
        """Aggregate primary rows and unique related-row membership in SQL.

        Args:
            table: Registered primary table.
            time_field: UTC timestamp used for calendar grouping.
            filters: Primary conditions, including the caller's time bounds.
            matching_table: Registered related table marking matching rows.
            join_fields: Shared columns forming the entire related primary key.
            matching_filters: Related conditions, including ownership scope.
            utc_offset_minutes: Fixed offset from UTC, in minutes.

        Returns:
            Daily counts without loading primary or related record payloads.

        Raises:
            ValueError: Join fields do not identify unique related rows.
        """
        primary = self.tables.get(table)
        related = self.tables.get(matching_table)
        if not join_fields or set(join_fields) != set(related.table.primary_key.columns.keys()):
            raise ValueError("Daily-count joins must use the complete related primary key")
        for field in join_fields:
            column_for(primary, field)
            column_for(related, field)
        timestamp = column_for(primary, time_field)
        if self.engine.dialect.name == "sqlite":
            day = func.date(timestamp, f"{utc_offset_minutes:+d} minutes")
        else:
            day = cast(func.timezone("UTC", timestamp) + timedelta(minutes=utc_offset_minutes), Date)
        matches = (
            select(*(related.table.c[field] for field in join_fields))
            .where(filter_clause(related, matching_filters))
            .subquery()
        )
        membership = matches.c[join_fields[0]].is_not(None)
        statement = (
            select(
                day.label("day"), func.count().label("total"), func.sum(case((membership, 1), else_=0)).label("matched")
            )
            .select_from(
                primary.table.outerjoin(
                    matches, and_(*(primary.table.c[field] == matches.c[field] for field in join_fields))
                )
            )
            .where(filter_clause(primary, filters))
            .group_by(day)
            .order_by(day)
        )
        with self._connect() as connection:
            return [DailyCount.model_validate(dict(row)) for row in connection.execute(statement).mappings()]

    def search(self, table: str, *, vector_field: str, query_vector, filters=None, fields=None, top_k: int = 10):
        """Return exact cosine neighbors among rows matching the filters.

        Args:
            table: Registered table name.
            vector_field: Registered Vector column to search.
            query_vector: Finite, nonzero vector with the declared dimensions.
            filters: Equality mapping or AND-combined Filter objects.
            fields: Returned record fields; None returns all fields.
            top_k: Positive maximum number of neighbors.

        Returns:
            SearchHit objects sorted by ascending distance, then primary key.
        """
        spec = self.tables.get(table)
        column = column_for(spec, vector_field)
        if not isinstance(column.type, Vector):
            raise ValueError("vector_field must identify a Vector column")
        if type(top_k) is not int or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        vector = column.type.validate(query_vector)
        columns = self._columns(spec, fields)
        distance = self._distance(column, vector)
        statement = (
            select(*columns, distance)
            .where(filter_clause(spec, filters), column.is_not(None))
            .order_by(distance, *spec.table.primary_key.columns)
            .limit(top_k)
        )
        with self._connect() as connection:
            return [
                SearchHit(record={c.key: row[i] for i, c in enumerate(columns)}, distance=row[-1])
                for row in connection.execute(statement)
            ]

    def _columns(self, spec, fields):
        if fields is None:
            return list(spec.table.c)
        if not fields or len(set(fields)) != len(fields):
            raise ValueError("fields must contain unique column names")
        return [column_for(spec, name) for name in fields]
