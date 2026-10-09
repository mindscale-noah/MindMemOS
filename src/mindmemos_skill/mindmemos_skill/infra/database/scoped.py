"""Fail-closed database operations bound to arbitrary scope fields."""

from contextlib import contextmanager

from .typing import Filter, Scope


class ScopedDatabase:
    """Apply scope to every operation without exposing the unscoped database."""

    def __init__(self, database, scope: Scope, *, in_transaction: bool = False):
        self._database = database
        self.scope = scope
        self.tables = database.tables
        self._active = True
        self._in_transaction = in_transaction

    def _required(self, table: str) -> dict[str, object]:
        """Resolve dimensions declared by a registered resource."""
        if not self._active:
            raise RuntimeError("Scoped database is closed")
        columns = self.tables.get(table).table.c
        required = dict(self.scope.fields)
        missing = required.keys() - columns.keys()
        if missing:
            raise ValueError(f"Table {table!r} lacks scope fields: {sorted(missing)}")
        return required

    def _filters(self, table: str, filters=None) -> list[Filter]:
        """Append mandatory equality predicates to caller conditions."""
        required = self._required(table)
        conditions = (
            [Filter(field=name, value=value) for name, value in filters.items()]
            if isinstance(filters, dict)
            else list(filters or [])
        )
        if any(
            condition.field in required and (condition.op != "eq" or condition.value != required[condition.field])
            for condition in conditions
        ):
            raise ValueError("Query ownership conflicts with the active scope")
        return conditions + [Filter(field=name, value=value) for name, value in required.items()]

    def _records(self, table: str, records: list[dict]) -> list[dict]:
        """Fill missing scope fields and reject conflicting ownership."""
        required = self._required(table)
        result = []
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("Records must be dictionaries")
            if any(name in record and record[name] != value for name, value in required.items()):
                raise ValueError("Record ownership conflicts with the active scope")
            result.append({**record, **required})
        return result

    def insert(self, table: str, records: list[dict]) -> int:
        """Insert records after binding their ownership to the active scope."""
        return self._database.insert(table, self._records(table, records))

    def upsert(self, table: str, records: list[dict], *, conflict_fields=None, update_fields=None) -> int:
        """Upsert only through a conflict key containing every scope dimension."""
        required = self._required(table)
        spec = self.tables.get(table)
        keys = list(conflict_fields) if conflict_fields is not None else list(spec.table.primary_key.columns.keys())
        if not required.keys() <= set(keys):
            raise ValueError("Scoped upsert conflict_fields must include every active scope field")
        if update_fields is not None and required.keys() & set(update_fields):
            raise ValueError("Scope fields cannot be updated")
        return self._database.upsert(
            table, self._records(table, records), conflict_fields=conflict_fields, update_fields=update_fields
        )

    def update(self, table: str, values: dict, *, filters) -> int:
        """Update only matching rows while preserving ownership fields."""
        if self._required(table).keys() & values.keys():
            raise ValueError("Scope fields cannot be updated")
        return self._database.update(table, values, filters=self._filters(table, filters))

    def update_many(self, table: str, records: list[dict]) -> int:
        """Update primary-key records with an added ownership condition."""
        keys = set(self.tables.get(table).table.primary_key.columns.keys())
        if self._in_transaction:
            return self._update_many_rows(table, records, keys)
        with self.transaction() as transaction:
            return transaction._update_many_rows(table, records, keys)

    def _update_many_rows(self, table: str, records: list[dict], keys: set[str]) -> int:
        """Apply per-key updates on the current transaction handle."""
        count = 0
        for record in records:
            if not isinstance(record, dict) or not keys <= record.keys():
                raise ValueError("Each update must contain the complete primary key")
            identity = {key: record[key] for key in keys}
            values = {key: value for key, value in record.items() if key not in keys}
            count += self.update(table, values, filters=identity)
        return count

    def delete(self, table: str, *, filters) -> int:
        """Delete only rows inside the active scope."""
        return self._database.delete(table, filters=self._filters(table, filters))

    def get(self, table: str, key):
        """Read a primary-key record only when it belongs to the scope."""
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
        """Read a bounded page while enforcing the active scope on all alternatives.

        Args:
            table: Registered table name.
            filters: Conditions combined with the active scope using AND.
            any_filters: Optional OR conditions inside the active scope.
            fields: Columns to return; None returns all columns.
            order_by: Columns used for deterministic ordering.
            limit: Maximum number of returned records; None has no limit.
            offset: Number of matching records to skip.

        Returns:
            Matching records inside the active scope.
        """
        return self._database.select(
            table,
            filters=self._filters(table, filters),
            any_filters=any_filters,
            fields=fields,
            order_by=order_by,
            limit=limit,
            offset=offset,
        )

    def count(self, table: str, *, filters=None) -> int:
        """Count rows inside the active scope."""
        return self._database.count(table, filters=self._filters(table, filters))

    def select_distinct(self, table: str, field: str, *, filters=None) -> list:
        """Return unique field values inside the active scope."""
        return self._database.select_distinct(table, field, filters=self._filters(table, filters))

    def search(self, table: str, *, vector_field: str, query_vector, filters=None, fields=None, top_k: int = 10):
        """Search vectors only inside the active scope."""
        return self._database.search(
            table,
            vector_field=vector_field,
            query_vector=query_vector,
            filters=self._filters(table, filters),
            fields=fields,
            top_k=top_k,
        )

    @contextmanager
    def transaction(self):
        """Yield a transaction that retains the same immutable scope."""
        if not self._active:
            raise RuntimeError("Scoped database is closed")
        with self._database.transaction() as database:
            transaction = ScopedDatabase(database, self.scope, in_transaction=True)
            try:
                yield transaction
            finally:
                transaction._active = False
