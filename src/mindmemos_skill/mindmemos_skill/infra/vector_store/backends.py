"""Composition helpers for optional vector-store backends."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ...errors import SkillCapabilityUnavailableError
from .models import BackendConfig
from .registry import BackendRegistry, TableRegistry
from .vector_store import ScopedVectorStore


def register_builtin_vector_stores(registry: BackendRegistry) -> None:
    """Register built-in providers without importing their optional drivers."""

    registry.register("pgvector", _create_pgvector_backend)


def _create_pgvector_backend(options: Mapping[str, Any], tables: TableRegistry) -> ScopedVectorStore:
    """Load PostgreSQL dependencies only when a vector store is constructed."""

    try:
        from .vector_store_impl import create_pgvector_backend
    except ModuleNotFoundError as exc:
        missing_root = (exc.name or "").split(".", 1)[0]
        if missing_root not in {"psycopg", "psycopg_pool"}:
            raise
        raise SkillCapabilityUnavailableError(
            "pgvector capability is unavailable because its PostgreSQL driver is not installed. "
            "Install it with `pip install 'mindmemos-skill[pgvector]'`."
        ) from exc

    return create_pgvector_backend(options, tables)


def create_vector_store(
    config: BackendConfig,
    tables: TableRegistry,
    *,
    registry: BackendRegistry | None = None,
) -> ScopedVectorStore:
    """Create a configured vector store, allowing custom providers."""

    resolved_registry = registry or BackendRegistry()
    if registry is None:
        register_builtin_vector_stores(resolved_registry)
    return resolved_registry.create(config, tables)


async def bootstrap_vector_store(
    config: BackendConfig,
    tables: TableRegistry,
    *,
    registry: BackendRegistry | None = None,
) -> ScopedVectorStore:
    """Create a vector store and initialize its registered indexes."""

    backend = create_vector_store(config, tables, registry=registry)
    try:
        await backend.ensure_schema(tables)
    except BaseException:
        await backend.close()
        raise
    return backend


__all__ = ["bootstrap_vector_store", "create_vector_store", "register_builtin_vector_stores"]
