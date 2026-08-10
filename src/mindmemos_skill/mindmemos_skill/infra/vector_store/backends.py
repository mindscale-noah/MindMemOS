"""Composition helpers for optional vector-store backends."""

from __future__ import annotations

from ...errors import SkillCapabilityUnavailableError
from .models import BackendConfig
from .registry import BackendRegistry, TableRegistry
from .vector_store import ScopedVectorStore


def register_builtin_vector_stores(registry: BackendRegistry) -> None:
    """Register every backend shipped by ``mindmemos-skill``.

    Keeping imports inside the composition function prevents concrete database
    drivers from leaking into the backend-neutral contract modules.
    """

    try:
        from .vector_store_impl import register_pgvector_backend
    except ModuleNotFoundError as exc:
        missing_root = (exc.name or "").split(".", 1)[0]
        if missing_root not in {"psycopg", "psycopg_pool"}:
            raise
        raise SkillCapabilityUnavailableError(
            "pgvector capability is unavailable because its PostgreSQL driver is not installed. "
            "Install it with `pip install 'mindmemos-skill[pgvector]'`."
        ) from exc

    register_pgvector_backend(registry)


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
