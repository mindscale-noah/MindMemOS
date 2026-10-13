"""Project-scoped read access to persisted Skill organization graphs."""

from uuid import UUID

from ....common.time_utils import utcnow
from ....errors import GraphConflictError, GraphNotFoundError
from ....infra.database import Database
from ...constants import (
    SKILL_GRAPH_EDGES_TABLE,
    SKILL_GRAPH_NODES_TABLE,
    SKILL_GRAPH_VERSIONS_TABLE,
    SKILL_GRAPHS_TABLE,
    SKILL_VERSIONS_TABLE,
    SKILLS_TABLE,
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

    async def _rows(self, table, **filters):
        rows = []
        while True:
            page = await self.database.select(
                table, filters={"project_id": self.project_id, **filters}, limit=100, offset=len(rows)
            )
            rows.extend(page)
            if len(page) < 100:
                return rows

    async def list_graphs(self) -> list[dict]:
        """Return graphs belonging to the selected project."""
        graphs = await self._rows(SKILL_GRAPHS_TABLE, is_deleted=False)
        return graphs

    async def delete(self, reference: str) -> None:
        """Soft-delete one graph by its public identity or alias.

        Args:
            reference: Active graph identity or alias.
        """
        graph = await self.graph(reference)
        now = utcnow()
        (
            await self.database.update(
                SKILL_GRAPHS_TABLE,
                {"is_deleted": True, "deleted_at": now, "updated_at": now},
                filters={"project_id": self.project_id, "graph_id": graph["graph_id"], "is_deleted": False},
            )
        )

    async def graph(self, reference: str) -> dict:
        """Find one stored graph by alias or public identity.

        Args:
            reference: Exact alias, storage ID, or public graph ID.

        Returns:
            Project-owned stored graph row.
        """
        graphs = await self.list_graphs()
        matches = [item for item in graphs if reference in (item["graph_id"], item.get("alias"))]
        if matches and reference.startswith("skg-"):
            prefix_matches = [item for item in graphs if (item["graph_id"]).startswith(reference)]
            if any(item["graph_id"] != matches[0]["graph_id"] for item in prefix_matches):
                raise GraphConflictError("Reference matches a graph alias and another ID prefix")
        if not matches and reference.startswith("skg-") and len(reference) > 4:
            matches = [item for item in graphs if (item["graph_id"]).startswith(reference)]
        if len(matches) != 1:
            if not matches:
                raise GraphNotFoundError("Graph not found")
            raise GraphConflictError("Ambiguous graph reference")
        return matches[0]

    async def resolve(self, reference: str) -> dict:
        """Resolve a graph alias, family ID, or unique version ID prefix.

        Args:
            reference: Project-local alias or prefixed graph identity.

        Returns:
            Public graph identity and selected immutable snapshot.

        Raises:
            GraphNotFoundError: The reference is absent from this project.
            GraphConflictError: The ID prefix or alias is ambiguous.
            ValueError: A version prefix has no identifier characters.
        """
        graphs = await self.list_graphs()
        if reference.startswith("skgv-"):
            aliases = [item for item in graphs if item.get("alias") == reference]
            versions = await self._rows(SKILL_GRAPH_VERSIONS_TABLE)
            active_ids = {item["graph_id"] for item in graphs}
            versions = [item for item in versions if item["graph_id"] in active_ids]
            matches = [
                version
                for version in versions
                if (version.get("public_version_id") or f"skgv-{version['graph_version_id'].hex}").startswith(reference)
            ]
            if aliases:
                if matches:
                    raise GraphConflictError("Reference matches a graph alias and version prefix")
                graph = aliases[0]
                current = graph["current_version_id"]
                selected = await self._rows(SKILL_GRAPH_VERSIONS_TABLE, graph_id=graph["graph_id"])
                version = next((item for item in selected if item["graph_version_id"] == current), None)
            else:
                if len(reference) == 5:
                    raise ValueError("Graph version prefix is incomplete")
                if len(matches) != 1:
                    if not matches:
                        raise GraphNotFoundError("Graph version not found")
                    raise GraphConflictError("Ambiguous graph version prefix")
                version = matches[0]
                graph = next(item for item in graphs if item["graph_id"] == version["graph_id"])
        else:
            graph = await self.graph(reference)
            current = graph["current_version_id"]
            version = None
            if current is not None:
                selected = await self._rows(SKILL_GRAPH_VERSIONS_TABLE, graph_id=graph["graph_id"])
                version = next((item for item in selected if item["graph_version_id"] == current), None)
        graph_id = graph["graph_id"]
        public_version_id = None
        if version is not None:
            public_version_id = version.get("public_version_id") or f"skgv-{version['graph_version_id'].hex}"
        return {
            "graph_id": graph_id,
            "graph_version_id": public_version_id,
            "alias": graph.get("alias") or graph["graph_id"],
            "is_current": version is not None and version["graph_version_id"] == graph["current_version_id"],
            "status": version["status"] if version is not None else None,
        }

    async def snapshots(self, graph_id: str) -> list[dict]:
        """Return a graph's immutable snapshots.

        Args:
            graph_id: Project-local graph identity.
        """
        (await self._active_graph(graph_id))
        return await self._rows(SKILL_GRAPH_VERSIONS_TABLE, graph_id=graph_id)

    async def snapshot(self, graph_id: str, snapshot_id: UUID) -> dict:
        """Resolve Skill-version nodes and their organization edges.

        Args:
            graph_id: Project-local graph identity.
            snapshot_id: Immutable graph content identity.

        Raises:
            ValueError: The requested snapshot is absent in this project.
        """
        (await self._active_graph(graph_id))
        scope = {"graph_id": graph_id, "graph_version_id": snapshot_id}
        if not (await self._rows(SKILL_GRAPH_VERSIONS_TABLE, **scope)):
            raise ValueError("Graph snapshot not found")
        nodes = await self._rows(SKILL_GRAPH_NODES_TABLE, **scope)
        by_id = {node["node_id"]: node for node in nodes}
        result = []
        for node in nodes:
            owner = node if node["node_type"] == "skill" else by_id.get(node["owner_node_id"])
            if owner is None or owner["node_type"] != "skill":
                raise ValueError("Scene owner is missing from the graph snapshot")
            skill = await self.version(owner["skill_version_id"])
            result.append(
                {
                    "node_id": node["node_id"],
                    "node_type": node["node_type"],
                    "owner_node_id": node["owner_node_id"],
                    "skill_version_id": owner["skill_version_id"],
                    "skill_id": skill.skill_id,
                    "skill_name": skill.name,
                    "name": skill.name if node["node_type"] == "skill" else node["name"],
                    "version_label": skill.version_label,
                    "status": skill.status,
                    "description": node["description"],
                    "one_line_capability": node["one_line_capability"],
                    "inputs": node["inputs"],
                    "outputs": node["outputs"],
                    "domain_tags": node["domain_tags"],
                    "tooling": node["tooling"],
                    "scene_metadata": {
                        key: node["metadata"][key]
                        for key in ("scene_id", "scene_type", "entry_conditions", "exit_conditions")
                        if key in node["metadata"]
                    },
                }
            )
        ids = {node["node_id"] for node in result}
        edges = []
        for edge in await self._rows(SKILL_GRAPH_EDGES_TABLE, **scope):
            if edge["source_node_id"] not in ids or edge["target_node_id"] not in ids:
                raise ValueError("Graph edge references a node outside its snapshot")
            item = {key: edge[key] for key in ("edge_id", "source_node_id", "target_node_id", "edge_type", "weight")}
            for key in ("condition", "evidence"):
                if key in edge["metadata"]:
                    item[key] = edge["metadata"][key]
            edges.append(item)
        return {"nodes": result, "edges": edges, "graph_version_id": str(snapshot_id)}

    async def version(self, version_id: str) -> SkillVersionRecord:
        """Read a version without crossing the selected project boundary.

        Args:
            version_id: Local immutable Skill version identity.
        """
        rows = await self._rows(SKILL_VERSIONS_TABLE, version_id=version_id)
        if not rows:
            raise ValueError("Skill version not found")
        family = await self.database.get(SKILLS_TABLE, {"project_id": self.project_id, "skill_id": rows[0]["skill_id"]})
        if family is None or family["is_deleted"]:
            raise ValueError("Skill version not found")
        return SkillVersionRecord.model_validate(rows[0])

    async def versions(self, skill_id: str) -> list[SkillVersionRecord]:
        """Return a Skill's linear history in creation order.

        Args:
            skill_id: Local Skill family identity.
        """
        family = await self.database.get(SKILLS_TABLE, {"project_id": self.project_id, "skill_id": skill_id})
        if family is None or family["is_deleted"]:
            return []
        rows = await self._rows(SKILL_VERSIONS_TABLE, skill_id=skill_id)
        return sorted(
            (SkillVersionRecord.model_validate(row) for row in rows), key=lambda v: (v.created_at, v.version_id)
        )

    async def _active_graph(self, graph_id: str) -> dict:
        """Return an active graph or reject its hidden history."""
        graph = await self.database.get(SKILL_GRAPHS_TABLE, {"project_id": self.project_id, "graph_id": graph_id})
        if graph is None or graph["is_deleted"]:
            raise GraphNotFoundError("Graph not found")
        return graph
