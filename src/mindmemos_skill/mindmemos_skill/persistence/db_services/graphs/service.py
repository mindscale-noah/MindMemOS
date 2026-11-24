"""Project-scoped read access to persisted Skill organization graphs."""

from pydantic import JsonValue

from ....common.time_utils import utcnow
from ....errors import GraphConflictError, GraphNotFoundError
from ....infra.database import Filter
from ...constants import (
    SKILL_GRAPH_EDGES_TABLE,
    SKILL_GRAPH_NODES_TABLE,
    SKILL_GRAPH_VERSIONS_TABLE,
    SKILL_GRAPHS_TABLE,
)
from ..skills.service import SkillDataService
from .typing import (
    GraphNodeType,
    GraphPreview,
    GraphResolution,
    GraphWriteRequest,
    SkillGraphEdgeRecord,
    SkillGraphEdgeWrite,
    SkillGraphNodeRecord,
    SkillGraphRecord,
    SkillGraphVersionRecord,
)
from .writer import SkillGraphStore, load_snapshot


class GraphDataService:
    """Query graph records and assemble operation-bound publication adapters.

    Args:
        database: Borrowed database with registered persistence tables.
        skills: Session-owned Skill reader for version visibility checks.
    """

    def __init__(self, database, *, skills: SkillDataService):
        """Borrow graph storage and the session's visible Skill reader.

        Args:
            database: Active database restricted to graph tables and authorized rows.
            skills: Skill repository sharing the same authorization scope.
        """
        self.database = database
        self.skills = skills

    async def _rows(self, table, filters=None, *, any_filters=None):
        """Read matching rows using backend filtering and deterministic primary-key order."""
        return await self.database.select(table, filters=filters, any_filters=any_filters, limit=None)

    async def list_graphs(self, *, limit: int | None = None, offset: int = 0) -> list[SkillGraphRecord]:
        """Read a stable page of graphs belonging to the active scope.

        Args:
            limit: Maximum records; None returns all matching graphs.
            offset: Number of ordered graphs to skip.

        Returns:
            Validated graph identities in primary-key order.
        """
        graphs = await self.database.select(
            SKILL_GRAPHS_TABLE, filters={"is_deleted": False}, limit=limit, offset=offset
        )
        return [SkillGraphRecord.model_validate(row) for row in graphs]

    async def delete(self, reference: str) -> None:
        """Soft-delete one graph by its public identity or alias.

        Args:
            reference: Active graph identity or alias.
        """
        graph = await self.get_graph(reference)
        now = utcnow()
        changed = await self.database.update(
            SKILL_GRAPHS_TABLE,
            {"is_deleted": True, "deleted_at": now, "updated_at": now},
            filters={"graph_id": graph.graph_id, "is_deleted": False},
        )
        if changed != 1:
            raise GraphNotFoundError("Graph was removed during deletion")

    async def get_graph(self, reference: str) -> SkillGraphRecord:
        """Find one stored graph by alias or public identity.

        Args:
            reference: Exact alias, storage ID, or public graph ID.

        Returns:
            Project-owned stored graph row.
        """
        rows = await self._rows(
            SKILL_GRAPHS_TABLE,
            {"is_deleted": False},
            any_filters=[Filter(field="graph_id", value=reference), Filter(field="alias", value=reference)],
        )
        prefixes = []
        if reference.startswith("skg-") and len(reference) > 4:
            prefixes = await self.database.select(
                SKILL_GRAPHS_TABLE,
                filters=[
                    Filter(field="is_deleted", value=False),
                    Filter(field="graph_id", op="starts_with", value=reference),
                ],
                limit=2,
            )
        if rows and any(row["graph_id"] != rows[0]["graph_id"] for row in prefixes):
            raise GraphConflictError("Reference matches a graph alias and another ID prefix")
        matches = rows or prefixes
        if len(matches) != 1:
            if not matches:
                raise GraphNotFoundError("Graph not found")
            raise GraphConflictError("Ambiguous graph reference")
        return SkillGraphRecord.model_validate(matches[0])

    async def resolve(self, reference: str) -> GraphResolution:
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
        if reference.startswith("skgv-"):
            aliases = await self._rows(SKILL_GRAPHS_TABLE, {"alias": reference, "is_deleted": False})
            candidates = await self._rows(
                SKILL_GRAPH_VERSIONS_TABLE,
                [Filter(field="graph_version_id", op="starts_with", value=reference)],
            )
            matches = []
            owners = {}
            for candidate in candidates:
                try:
                    owner = await self._active_graph(candidate["graph_id"])
                except GraphNotFoundError:
                    continue
                matches.append(SkillGraphVersionRecord.model_validate(candidate))
                owners[candidate["graph_id"]] = owner
            if aliases:
                if matches:
                    raise GraphConflictError("Reference matches a graph alias and version prefix")
                graph = SkillGraphRecord.model_validate(aliases[0])
                version = await self._current_version(graph)
            else:
                if len(reference) == 5:
                    raise ValueError("Graph version prefix is incomplete")
                if len(matches) != 1:
                    if not matches:
                        raise GraphNotFoundError("Graph version not found")
                    raise GraphConflictError("Ambiguous graph version prefix")
                version = matches[0]
                graph = owners[version.graph_id]
        else:
            graph = await self.get_graph(reference)
            version = await self._current_version(graph)
        graph_id = graph.graph_id
        return {
            "graph_id": graph_id,
            "graph_version_id": version.graph_version_id if version is not None else None,
            "alias": graph.alias or graph.graph_id,
            "is_current": version is not None and version.graph_version_id == graph.current_version_id,
            "status": version.status if version is not None else None,
        }

    async def snapshots(
        self, graph_id: str, *, limit: int | None = None, offset: int = 0
    ) -> list[SkillGraphVersionRecord]:
        """Read a stable page of an active graph's immutable snapshots.

        Args:
            graph_id: Project-local graph identity.
            limit: Maximum records; None returns the complete history.
            offset: Number of ordered snapshots to skip.

        Returns:
            Validated snapshot records in creation and identity order.
        """
        (await self._active_graph(graph_id))
        rows = await self.database.select(
            SKILL_GRAPH_VERSIONS_TABLE,
            filters={"graph_id": graph_id},
            order_by=["created_at", "graph_version_id"],
            limit=limit,
            offset=offset,
        )
        return [SkillGraphVersionRecord.model_validate(row) for row in rows]

    async def snapshot(self, graph_id: str, snapshot_id: str) -> GraphPreview:
        """Resolve Skill-version nodes and their organization edges.

        Args:
            graph_id: Project-local graph identity.
            snapshot_id: Immutable graph content identity.

        Raises:
            ValueError: The requested snapshot is absent in this project.
        """
        (await self._active_graph(graph_id))
        scope = {"graph_id": graph_id, "graph_version_id": snapshot_id}
        if not (await self._rows(SKILL_GRAPH_VERSIONS_TABLE, scope)):
            raise ValueError("Graph snapshot not found")
        nodes = [SkillGraphNodeRecord.model_validate(row) for row in await self._rows(SKILL_GRAPH_NODES_TABLE, scope)]
        by_id = {node.node_id: node for node in nodes}
        result = []
        versions = {}
        for node in nodes:
            owner = node if node.node_type == GraphNodeType.SKILL else by_id.get(node.owner_node_id)
            if owner is None or owner.node_type != GraphNodeType.SKILL:
                raise ValueError("Scene owner is missing from the graph snapshot")
            version_id = owner.skill_version_id
            if version_id not in versions:
                versions[version_id] = await self.skills.get(version_id)
            skill = versions[version_id]
            result.append(
                {
                    "node_id": node.node_id,
                    "node_type": node.node_type,
                    "owner_node_id": node.owner_node_id,
                    "skill_version_id": owner.skill_version_id,
                    "skill_id": skill.skill_id,
                    "skill_name": skill.name,
                    "name": skill.name if node.node_type == GraphNodeType.SKILL else node.name,
                    "version_label": skill.version_label,
                    "status": skill.status,
                    "description": node.description,
                    "one_line_capability": node.one_line_capability,
                    "inputs": node.inputs,
                    "outputs": node.outputs,
                    "domain_tags": node.domain_tags,
                    "tooling": node.tooling,
                    "scene_metadata": {
                        key: node.metadata[key]
                        for key in ("scene_id", "scene_type", "entry_conditions", "exit_conditions")
                        if key in node.metadata
                    },
                }
            )
        ids = {node["node_id"] for node in result}
        edges = []
        for row in await self._rows(SKILL_GRAPH_EDGES_TABLE, scope):
            edge = SkillGraphEdgeRecord.model_validate(row)
            if edge.source_node_id not in ids or edge.target_node_id not in ids:
                raise ValueError("Graph edge references a node outside its snapshot")
            item = edge.model_dump(include={"edge_id", "source_node_id", "target_node_id", "edge_type", "weight"})
            for key in ("condition", "evidence"):
                if key in edge.metadata:
                    item[key] = edge.metadata[key]
            edges.append(item)
        return {"nodes": result, "edges": edges, "graph_version_id": str(snapshot_id)}

    async def _active_graph(self, graph_id: str) -> SkillGraphRecord:
        """Read an exact active graph within the authorized database scope."""
        rows = await self.database.select(
            SKILL_GRAPHS_TABLE, filters={"graph_id": graph_id, "is_deleted": False}, limit=2
        )
        if not rows:
            raise GraphNotFoundError("Graph not found")
        if len(rows) != 1:
            raise GraphConflictError("Graph identity requires a project scope")
        return SkillGraphRecord.model_validate(rows[0])

    async def _current_version(self, graph: SkillGraphRecord) -> SkillGraphVersionRecord | None:
        """Read the pinned head without scanning the graph's history."""
        current = graph.current_version_id
        if current is None:
            return None
        rows = await self.database.select(
            SKILL_GRAPH_VERSIONS_TABLE,
            filters={"graph_id": graph.graph_id, "graph_version_id": current},
            limit=1,
        )
        if not rows:
            raise ValueError("Graph publication pointer has no snapshot")
        return SkillGraphVersionRecord.model_validate(rows[0])

    async def load(
        self, graph_id: str, snapshot_id: str
    ) -> tuple[list[SkillGraphNodeRecord], list[SkillGraphEdgeRecord], dict[str, JsonValue]]:
        """Read and verify a snapshot without constructing a publication adapter.

        Args:
            graph_id: Exact graph family identity.
            snapshot_id: Exact immutable snapshot identity.

        Returns:
            Validated nodes, edges, and build metadata.
        """
        if self.database.scope is None:
            raise RuntimeError("Graph reads require a project session")
        return await load_snapshot(
            self.database,
            graph_id,
            snapshot_id,
        )

    async def publish(
        self,
        request: GraphWriteRequest,
        snapshot_id: str,
        *,
        request_hash: str,
        nodes: list[SkillGraphNodeRecord],
        edges: list[SkillGraphEdgeWrite],
        metadata: dict[str, JsonValue],
    ) -> str:
        """Publish prepared graph content with exact member and head checks.

        Args:
            request: Target graph, exact Skill bindings, actor, and expected head.
            snapshot_id: Proposed immutable content identity.
            request_hash: Build identity used for exact-input reuse.
            nodes: Complete prepared nodes with source bindings and embeddings.
            edges: Complete prepared relations with in-snapshot endpoints.
            metadata: Validated build settings and evidence.

        Returns:
            Published snapshot identity, including reuse of an existing identical build.
        """
        return await self.writer(request).publish(
            snapshot_id, request_hash=request_hash, nodes=nodes, edges=edges, metadata=metadata
        )

    def writer(self, request: GraphWriteRequest) -> SkillGraphStore:
        """Bind one graph operation without changing this repository's scope.

        Args:
            request: Graph publication scope, exact member versions, actor and head condition.

        Returns:
            Operation-owned algorithm storage adapter using this session's dependencies.
        """
        if self.database.scope is None:
            raise RuntimeError("Graph publication requires a project session")
        return SkillGraphStore(
            self.database,
            graph_id=request.graph_id,
            actor_id=request.actor_id,
            version_ids=request.version_ids,
            alias=request.alias,
            message=request.message,
            expected_snapshot=request.expected_snapshot,
            check_head=True,
            skills=self.skills,
        )
