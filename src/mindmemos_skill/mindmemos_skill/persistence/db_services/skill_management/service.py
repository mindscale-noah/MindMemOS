"""Project-scoped read access to persisted Skill organization graphs."""

from uuid import UUID

from ....infra.database import Database
from ...constants import (
    SKILL_GRAPH_EDGES_TABLE,
    SKILL_GRAPH_NODES_TABLE,
    SKILL_GRAPH_REVISIONS_TABLE,
    SKILL_GRAPH_VERSIONS_TABLE,
    SKILL_GRAPHS_TABLE,
    SKILL_VERSIONS_TABLE,
)
from ..skills.typing import SkillVersionRecord


class GraphQueryService:
    """Read graph snapshots without creating schemas or changing publication state.

    Args:
        database: Borrowed database with registered persistence tables.
        project_id: Explicit project isolation boundary for all queries.
    """

    def __init__(self, database: Database, project_id: str):
        if not project_id.strip():
            raise ValueError("project_id is required")
        self.database = database
        self.project_id = project_id

    def _rows(self, table, **filters):
        rows = []
        while True:
            page = self.database.select(
                table, filters={"project_id": self.project_id, **filters}, limit=100, offset=len(rows)
            )
            rows.extend(page)
            if len(page) < 100:
                return rows

    def list_graphs(self) -> list[dict]:
        """Return graphs belonging to the selected project."""
        return self._rows(SKILL_GRAPHS_TABLE)

    def snapshots(self, graph_id: str) -> list[dict]:
        """Return a graph's snapshots and their published revisions.

        Args:
            graph_id: Project-local graph identity.
        """
        revisions = self._rows(SKILL_GRAPH_REVISIONS_TABLE, graph_id=graph_id)
        return [
            {**row, "revisions": [r["revision"] for r in revisions if r["graph_version_id"] == row["graph_version_id"]]}
            for row in self._rows(SKILL_GRAPH_VERSIONS_TABLE, graph_id=graph_id)
        ]

    def snapshot(self, graph_id: str, snapshot_id: UUID) -> dict:
        """Resolve Skill-version nodes and their organization edges.

        Args:
            graph_id: Project-local graph identity.
            snapshot_id: Immutable graph content identity.

        Raises:
            ValueError: The requested snapshot is absent in this project.
        """
        scope = {"graph_id": graph_id, "graph_version_id": snapshot_id}
        if not self._rows(SKILL_GRAPH_VERSIONS_TABLE, **scope):
            raise ValueError("Graph snapshot not found")
        nodes = self._rows(SKILL_GRAPH_NODES_TABLE, **scope, node_type="skill")
        result = []
        for node in nodes:
            skill = self.version(node["skill_version_id"])
            result.append(
                {
                    "node_id": node["node_id"],
                    "skill_version_id": node["skill_version_id"],
                    "skill_id": skill.skill_id,
                    "name": skill.name,
                    "version_label": skill.version_label,
                    "status": skill.status,
                }
            )
        ids = {node["node_id"] for node in result}
        edges = [
            e
            for e in self._rows(SKILL_GRAPH_EDGES_TABLE, **scope)
            if e["source_node_id"] in ids and e["target_node_id"] in ids
        ]
        return {"nodes": result, "edges": edges, "graph_version_id": str(snapshot_id)}

    def version(self, version_id: str) -> SkillVersionRecord:
        """Read a version without crossing the selected project boundary.

        Args:
            version_id: Local immutable Skill version identity.
        """
        rows = self._rows(SKILL_VERSIONS_TABLE, version_id=version_id)
        if not rows:
            raise ValueError("Skill version not found")
        return SkillVersionRecord.model_validate(rows[0])

    def versions(self, skill_id: str) -> list[SkillVersionRecord]:
        """Return a Skill's linear history in creation order.

        Args:
            skill_id: Local Skill family identity.
        """
        rows = self._rows(SKILL_VERSIONS_TABLE, skill_id=skill_id)
        return sorted(
            (SkillVersionRecord.model_validate(row) for row in rows), key=lambda v: (v.created_at, v.version_id)
        )
