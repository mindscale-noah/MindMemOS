"""Project-scoped graph reads over the repository's shared persistence."""

import asyncio
from uuid import UUID

from ..persistence import SkillVersionRecord
from ..persistence.db_services.skill_management.service import GraphQueryService
from ..persistence.store import SkillPersistence


class GraphQueryPipeline:
    """Provide read-only graph operations without exposing a database to callers.

    Args:
        persistence: Borrowed repository data services.
        project_id: Scope already resolved by the caller's service.
    """

    def __init__(self, persistence: SkillPersistence, project_id: str):
        self._queries = GraphQueryService(persistence.database, project_id)

    async def list_graphs(self) -> list[dict]:
        """List graphs in the bound project."""
        return await asyncio.to_thread(self._queries.list_graphs)

    async def snapshots(self, graph_id: str) -> list[dict]:
        """List graph snapshots and publication revisions.

        Args:
            graph_id: Project-local graph identity.
        """
        return await asyncio.to_thread(self._queries.snapshots, graph_id)

    async def snapshot(self, graph_id: str, snapshot_id: UUID) -> dict:
        """Read one graph's nodes and edges.

        Args:
            graph_id: Project-local graph identity.
            snapshot_id: Immutable graph content identity.
        """
        return await asyncio.to_thread(self._queries.snapshot, graph_id, snapshot_id)

    async def version(self, version_id: str) -> SkillVersionRecord:
        """Read an exact Skill version within the bound project.

        Args:
            version_id: Immutable Skill version identity.
        """
        return await asyncio.to_thread(self._queries.version, version_id)

    async def versions(self, skill_id: str) -> list[SkillVersionRecord]:
        """Read a Skill's version history within the bound project.

        Args:
            skill_id: Exact Skill identity.
        """
        return await asyncio.to_thread(self._queries.versions, skill_id)
