"""Independent registration for structured database backends and schemas."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from types import MappingProxyType
from typing import Any

from .database import ScopedDatabase
from .models import DatabaseConfig, SchemaMigration, TableSpec


class TableRegistry:
    def __init__(
        self,
        specs: Iterable[TableSpec] = (),
        *,
        migrations: Iterable[SchemaMigration] = (),
    ) -> None:
        self._specs: dict[str, TableSpec] = {}
        self._migrations: list[SchemaMigration] = []
        self._frozen = False
        for spec in specs:
            self.register(spec)
        for migration in migrations:
            self.register_migration(migration)

    def register(self, spec: TableSpec) -> None:
        if self._frozen:
            raise RuntimeError("table registry is frozen")
        if spec.name in self._specs:
            raise ValueError(f"table {spec.name!r} is already registered")
        index_owners = {index.name: table.name for table in self._specs.values() for index in table.indexes}
        for index in spec.indexes:
            owner = index_owners.get(index.name)
            if owner is not None:
                raise ValueError(
                    f"index {index.name!r} is already registered for table {owner!r}; "
                    f"table {spec.name!r} cannot reuse it"
                )
        self._specs[spec.name] = spec

    def register_migration(self, migration: SchemaMigration) -> None:
        if self._frozen:
            raise RuntimeError("table registry is frozen")
        identity = (migration.namespace, migration.version)
        if any((item.namespace, item.version) == identity for item in self._migrations):
            raise ValueError(
                f"schema migration {migration.namespace!r} version {migration.version} is already registered"
            )
        unknown = set(migration.tables) - set(self._specs)
        if unknown:
            raise ValueError(f"schema migration references unknown tables: {sorted(unknown)}")
        previous_versions = [item.version for item in self._migrations if item.namespace == migration.namespace]
        expected_version = previous_versions[-1] + 1 if previous_versions else 1
        if migration.version != expected_version:
            raise ValueError(
                f"schema migrations for namespace {migration.namespace!r} must be contiguous; "
                f"expected version {expected_version}, got {migration.version}"
            )
        self._migrations.append(migration)

    def get(self, name: str) -> TableSpec:
        try:
            return self._specs[name]
        except KeyError as exc:
            raise KeyError(f"unknown logical table {name!r}") from exc

    def freeze(self) -> Mapping[str, TableSpec]:
        self._frozen = True
        return MappingProxyType(self._specs)

    @property
    def specs(self) -> tuple[TableSpec, ...]:
        return tuple(self._specs.values())

    @property
    def migrations(self) -> tuple[SchemaMigration, ...]:
        return tuple(self._migrations)


DatabaseFactory = Callable[[Mapping[str, Any], TableRegistry], ScopedDatabase]


class DatabaseRegistry:
    """Provider registry isolated from vector-store providers."""

    def __init__(self) -> None:
        self._factories: dict[str, DatabaseFactory] = {}

    def register(self, name: str, factory: DatabaseFactory) -> None:
        normalized = name.strip().lower()
        if not normalized:
            raise ValueError("database provider name must not be empty")
        if normalized in self._factories:
            raise ValueError(f"database provider {normalized!r} is already registered")
        self._factories[normalized] = factory

    def create(self, config: DatabaseConfig, tables: TableRegistry) -> ScopedDatabase:
        name = config.provider.strip().lower()
        try:
            factory = self._factories[name]
        except KeyError as exc:
            raise ValueError(f"unsupported database backend {config.provider!r}") from exc
        backend = factory(MappingProxyType(dict(config.options)), tables)
        missing = config.required.missing_from(backend.capabilities)
        if missing:
            raise ValueError(f"database backend {name!r} is missing required capabilities: {', '.join(missing)}")
        return backend

    @property
    def providers(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))


__all__ = ["DatabaseFactory", "DatabaseRegistry", "TableRegistry"]
