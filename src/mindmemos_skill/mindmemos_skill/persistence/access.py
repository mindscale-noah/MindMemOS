"""Restrict repository operations to declared tables and the active database scope."""

from contextlib import asynccontextmanager, contextmanager

from ..infra.database import Database, TableRegistry
from ..infra.database.scoped import ScopedDatabase
from .typing import TransactionState


def _project_id_from_database(database: Database | ScopedDatabase) -> str:
    """Read project ownership from the active parent scope without caching it."""
    scope = None
    while isinstance(database, ScopedDatabase):
        if not database._active:
            raise RuntimeError("Persistence scope is closed")
        scope = database.scope
        database = database._database
    if database._closed:
        raise RuntimeError("Database is closed")
    if scope is None:
        raise RuntimeError("A project scope is required")
    project_id = scope.require("project_id")
    if not isinstance(project_id, str):
        raise ValueError("Project scope must contain a string project_id")
    return project_id


class RepositoryDatabase(ScopedDatabase):
    """Borrow a scoped handle with explicit writable and read-only table sets."""

    def __init__(
        self,
        database: Database | ScopedDatabase,
        *,
        tables: frozenset[str],
        read_tables: frozenset[str] = frozenset(),
        state: TransactionState | None = None,
    ):
        """Bind a repository without connecting or widening its parent handle.

        Args:
            database: Active scoped, transactional, or explicit maintenance handle.
            tables: Tables this repository may read and write.
            read_tables: Additional tables this repository may only read.
            state: Shared failure state when participating in a transaction.
        """
        self._database = database
        self.scope = database.scope if isinstance(database, ScopedDatabase) else None
        self._write_tables = tables
        self._read_tables = read_tables
        self.tables = TableRegistry(database.tables.get(name) for name in sorted(tables | read_tables))
        self._active = True
        self._in_transaction = state is not None
        self._state = state

    @property
    def project_id(self) -> str:
        """Read the project bound to this repository's active authorization scope.

        Returns:
            Current project identity, derived from the parent scope.

        Raises:
            RuntimeError: The handle is closed or has no project scope.
            ValueError: The project scope is missing or has a non-string identity.
        """
        return _project_id_from_database(self)

    def _required(self, table):
        """Check both the repository boundary and every parent lease."""
        if not self._active:
            raise RuntimeError("Repository session is closed")
        if table not in self._write_tables | self._read_tables:
            raise PermissionError(f"Repository cannot access table {table!r}")
        self.tables.get(table)
        if isinstance(self._database, ScopedDatabase):
            return self._database._required(table)
        if self._database._closed:
            raise RuntimeError("Database is closed")
        return {}

    def _writable(self, table):
        """Reject writes to a dependency's read-only tables."""
        self._required(table)
        if table not in self._write_tables:
            raise PermissionError(f"Repository cannot write table {table!r}")

    async def insert(self, table, records):
        """Insert records only into owned tables.

        Args:
            table: Declared writable table.
            records: Complete records validated by the registered table model.

        Returns:
            Number of inserted records.
        """
        self._writable(table)
        return await super().insert(table, records)

    async def upsert(self, table, records, *, conflict_fields=None, update_fields=None):
        """Upsert records within the table and ownership boundary.

        Args:
            table: Declared writable table.
            records: Complete records validated by the table model.
            conflict_fields: Unique-key columns; None selects the primary key.
            update_fields: Columns to replace on conflict; None uses backend defaults.

        Returns:
            Number of affected records.
        """
        self._writable(table)
        return await super().upsert(table, records, conflict_fields=conflict_fields, update_fields=update_fields)

    async def update(self, table, values, *, filters):
        """Update owned rows without changing their authorized scope.

        Args:
            table: Declared writable table.
            values: Column-value replacements validated by the table schema.
            filters: Conditions further narrowing the authorized scope.

        Returns:
            Number of affected records.
        """
        self._writable(table)
        return await super().update(table, values, filters=filters)

    async def delete(self, table, *, filters):
        """Delete owned rows within the active scope.

        Args:
            table: Declared writable table.
            filters: Conditions further narrowing the authorized scope.

        Returns:
            Number of deleted records.
        """
        self._writable(table)
        return await super().delete(table, filters=filters)

    @contextmanager
    def dependency(self, repository: "RepositoryDatabase"):
        """Borrow a granted read dependency on this transaction's connection.

        Args:
            repository: Active repository handle granted by the same scoped session.

        Yields:
            Read-only dependency retaining its original tables and ownership scope.

        Raises:
            ValueError: The dependency has a different scope or database resource.
            RuntimeError: This handle does not belong to an active transaction.
        """
        if not self._active or not self._in_transaction:
            raise RuntimeError("Dependencies can only be bound inside a repository transaction")
        if repository.scope != self.scope:
            raise ValueError("Repository dependency has a different ownership scope")
        # Both grants must originate from the same resource, not merely matching project values.
        source = repository._database
        target = self._database
        while isinstance(source, ScopedDatabase):
            source = source._database
        while isinstance(target, ScopedDatabase):
            target = target._database
        if source.engine is not target.engine:
            raise ValueError("Repository dependency uses a different database resource")
        tables = repository._write_tables | repository._read_tables
        for table in tables:
            repository._required(table)
        handle = RepositoryDatabase(self._database, tables=frozenset(), read_tables=tables, state=self._state)
        try:
            yield handle
        finally:
            handle._active = False

    @asynccontextmanager
    async def transaction(self):
        """Join an existing repository transaction or start a short atomic operation.

        Yields:
            A handle retaining this repository's table and row restrictions.
        """
        if not self._active:
            raise RuntimeError("Repository session is closed")
        if self._in_transaction:
            try:
                yield self
            except BaseException:
                self._state.failed = True
                raise
            return
        async with self._database.transaction() as database:
            state = TransactionState()
            transaction = RepositoryDatabase(
                database, tables=self._write_tables, read_tables=self._read_tables, state=state
            )
            try:
                yield transaction
                if state.failed:
                    raise RuntimeError("Repository transaction contains a failed operation")
            finally:
                transaction._active = False
