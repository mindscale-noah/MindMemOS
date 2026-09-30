"""Project-scoped graph reads over the repository's shared persistence."""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any
from uuid import UUID

from ..infra.database import Scope
from ..persistence import SkillVersionRecord
from ..persistence.db_services.skill_management.service import GraphQueryService
from ..persistence.store import SkillPersistence


class GraphQueryPipeline:
    """Provide read-only graph operations without exposing a database to callers.

    Args:
        persistence: Borrowed repository data services.
        project_id: Project selected by the caller's service.
        authorize: Callback that rechecks access before each graph query.
    """

    def __init__(self, persistence: SkillPersistence, project_id: str, authorize: Callable[[], Awaitable[Scope]]):
        self._persistence = persistence
        self._project_id = project_id
        self._authorize = authorize

    async def _execute(self, operation: Callable[[GraphQueryService], Any]):
        """Create a fresh scoped handle for one graph query."""
        scope = await self._authorize()
        if scope.require("project_id") != self._project_id:
            raise PermissionError("Authorized graph project changed")
        with self._persistence.session(scope) as store:
            queries = GraphQueryService(store.database, self._project_id)
            return await asyncio.to_thread(operation, queries)

    async def list_graphs(self) -> list[dict]:
        """List graphs in the bound project."""
        return await self._execute(lambda queries: queries.list_graphs())

    async def snapshots(self, graph_id: str) -> list[dict]:
        """List graph snapshots and publication revisions.

        Args:
            graph_id: Project-local graph identity.
        """
        return await self._execute(lambda queries: queries.snapshots(graph_id))

    async def snapshot(self, graph_id: str, snapshot_id: UUID) -> dict:
        """Read one graph's nodes and edges.

        Args:
            graph_id: Project-local graph identity.
            snapshot_id: Immutable graph content identity.
        """
        return await self._execute(lambda queries: queries.snapshot(graph_id, snapshot_id))

    async def version(self, version_id: str) -> SkillVersionRecord:
        """Read an exact Skill version within the bound project.

        Args:
            version_id: Immutable Skill version identity.
        """
        return await self._execute(lambda queries: queries.version(version_id))

    async def versions(self, skill_id: str) -> list[SkillVersionRecord]:
        """Read a Skill's version history within the bound project.

        Args:
            skill_id: Exact Skill identity.
        """
        return await self._execute(lambda queries: queries.versions(skill_id))
