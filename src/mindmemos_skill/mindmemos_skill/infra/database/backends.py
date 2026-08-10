"""Composition helpers for structured persistence databases."""

from __future__ import annotations

from .database import ScopedDatabase
from .models import DatabaseConfig
from .registry import DatabaseRegistry, TableRegistry


def register_builtin_databases(registry: DatabaseRegistry) -> None:
    from .database_impl import register_sqlite_backend

    register_sqlite_backend(registry)
    try:
        from .database_impl.postgres import register_postgres_backend
    except ImportError:
        return
    register_postgres_backend(registry)


def create_database(
    config: DatabaseConfig,
    tables: TableRegistry,
    *,
    registry: DatabaseRegistry | None = None,
) -> ScopedDatabase:
    resolved = registry or DatabaseRegistry()
    if registry is None:
        register_builtin_databases(resolved)
    return resolved.create(config, tables)


async def bootstrap_database(
    config: DatabaseConfig,
    tables: TableRegistry,
    *,
    registry: DatabaseRegistry | None = None,
) -> ScopedDatabase:
    database = create_database(config, tables, registry=registry)
    try:
        await database.ensure_schema(tables)
    except BaseException:
        await database.close()
        raise
    return database


__all__ = ["bootstrap_database", "create_database", "register_builtin_databases"]
