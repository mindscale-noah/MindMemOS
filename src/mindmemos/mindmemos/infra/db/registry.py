"""Lazy low-level database client registry."""

from __future__ import annotations

import asyncio
import weakref
from dataclasses import dataclass
from typing import Any

from mindmemos_skill.infra.database import (
    DatabaseConfig as StructuredDatabaseConfig,
)
from mindmemos_skill.infra.database import (
    DatabaseRequirements,
    ScopedDatabase,
    create_database,
)

from ...config import get_config
from .collections import SkillVersionRepository
from .neo4j import Neo4jStore
from .qdrant import QdrantStore
from .skill_relational import SkillRelationalRepository, build_cloud_skill_tables


@dataclass(slots=True)
class _LoopDatabaseClients:
    """Event-loop-scoped async database clients used by pipelines."""

    qdrant: QdrantStore
    neo4j: Neo4jStore
    skill: SkillRelationalRepository
    skill_database: ScopedDatabase
    legacy_skill: SkillVersionRepository

    async def close(self) -> None:
        """Close all underlying database clients.

        The relational Skill repository and the memory stores own independent
        connections and are closed independently.
        """

        await self.skill_database.close()
        await self.qdrant.close()
        await self.neo4j.close()


class DatabaseClients:
    """Synchronous provider for the current event loop's async database clients.

    The provider itself is safe to construct and pass around from synchronous
    code. Accessing ``qdrant``, ``neo4j`` or ``skill`` resolves the real clients
    for the currently running event loop.
    """

    @property
    def qdrant(self) -> QdrantStore:
        """Return Qdrant clients for the currently running event loop."""

        return _get_loop_database_clients().qdrant

    @property
    def neo4j(self) -> Neo4jStore:
        """Return Neo4j clients for the currently running event loop."""

        return _get_loop_database_clients().neo4j

    @property
    def skill(self) -> SkillRelationalRepository:
        """Return skill repository clients for the currently running event loop."""

        return _get_loop_database_clients().skill

    @property
    def legacy_skill(self) -> SkillVersionRepository:
        """Temporary Qdrant reader used only by migration/legacy flows."""

        return _get_loop_database_clients().legacy_skill

    async def close(self) -> None:
        """Close database clients for the currently running event loop."""

        await close_database_clients()


_database_clients = DatabaseClients()
_database_clients_by_loop: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, _LoopDatabaseClients] = (
    weakref.WeakKeyDictionary()
)


def get_database_clients() -> DatabaseClients:
    """Get the process-wide database client provider.

    The returned object can be stored from synchronous code. Its attributes
    resolve to real async clients for the running event loop when used.
    """

    return _database_clients


def _get_loop_database_clients() -> _LoopDatabaseClients:
    """Get low-level database clients for the current event loop."""

    loop = asyncio.get_running_loop()
    clients = _database_clients_by_loop.get(loop)
    if clients is None:
        clients = _create_database_clients()
        _database_clients_by_loop[loop] = clients
    return clients


def _create_database_clients() -> _LoopDatabaseClients:
    cfg = get_config().database
    qdrant = QdrantStore(cfg.qdrant)
    skill_options: dict[str, Any]
    if cfg.skill.provider == "sqlite":
        skill_options = {"path": cfg.skill.path}
    elif cfg.skill.provider in {"postgres", "postgresql"}:
        if not cfg.skill.dsn:
            raise ValueError("database.skill.dsn is required for the postgres provider")
        skill_options = {"dsn": cfg.skill.dsn, "pool_size": cfg.skill.pool_size}
    else:
        skill_options = {}
    skill_database = create_database(
        StructuredDatabaseConfig(
            provider=cfg.skill.provider,
            options=skill_options,
            required=DatabaseRequirements(
                metadata_filtering=True,
                batch_record_io=True,
                atomic_batch_write=True,
                transactions=True,
                compare_and_swap=True,
            ),
        ),
        build_cloud_skill_tables(),
    )
    return _LoopDatabaseClients(
        qdrant=qdrant,
        neo4j=Neo4jStore(cfg.neo4j),
        skill=SkillRelationalRepository(skill_database),
        skill_database=skill_database,
        legacy_skill=SkillVersionRepository(cfg.qdrant, engine=qdrant.engine),
    )


def resolve_database_clients(clients: Any | None = None) -> Any:
    """Return provided database clients or the current event-loop clients."""

    return clients if clients is not None else get_database_clients()


async def close_database_clients() -> None:
    """Close and forget database clients for the current event loop."""

    loop = asyncio.get_running_loop()
    clients = _database_clients_by_loop.pop(loop, None)
    if clients is not None:
        await clients.close()


def reset_database_clients() -> None:
    """Forget database client registry entries, mainly for tests and config refreshes.

    Call ``close_database_clients`` first when live clients may hold network
    resources.
    """

    _database_clients_by_loop.clear()
