"""Instance-scoped lookup of business-owned table definitions."""

from collections.abc import Iterable

from .typing import TableSpec


class TableRegistry:
    """Collect table definitions without creating database resources.

    Args:
        tables: Initial definitions to register.
    """

    def __init__(self, tables: Iterable[TableSpec] = ()):
        self._tables: dict[str, TableSpec] = {}
        for table in tables:
            self.register(table)

    def register(self, spec: TableSpec) -> None:
        """Register one validated definition.

        Args:
            spec: Business table and its validation models.

        Raises:
            ValueError: The public name or physical table is already registered.
        """
        if spec.name in self._tables or any(s.table.fullname == spec.table.fullname for s in self._tables.values()):
            raise ValueError(f"Duplicate table registration: {spec.name!r}")
        self._tables[spec.name] = spec

    def get(self, name: str) -> TableSpec:
        """Resolve a public name.

        Args:
            name: Exact registered name.

        Raises:
            ValueError: The name has not been registered.
        """
        if name not in self._tables:
            raise ValueError(f"Unregistered table: {name!r}")
        return self._tables[name]

    def values(self) -> tuple[TableSpec, ...]:
        """Return a stable snapshot of registered definitions."""
        return tuple(self._tables.values())
