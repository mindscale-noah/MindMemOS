"""Atomic graph publication on existing project-scoped tables."""

import hashlib
import json
from collections.abc import Mapping
from uuid import UUID

from ....common.time_utils import utcnow
from ....errors import GraphConflictError, GraphNotFoundError
from ....infra.database import Database
from ....skill import Skill
from ...constants import (
    SKILL_GRAPH_EDGES_TABLE,
    SKILL_GRAPH_NODES_TABLE,
    SKILL_GRAPH_VERSIONS_TABLE,
    SKILL_GRAPHS_TABLE,
)
from ..skills.service import SkillDataService
from .typing import (
    SkillGraphEdgeRecord,
    SkillGraphNodeRecord,
    SkillGraphRecord,
    SkillGraphVersionRecord,
)


def graph_digest(value: object) -> str:
    """Hash JSON-safe graph content independently of dictionary insertion order.

    Args:
        value: Canonical input or graph record payload.
    """
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


class SkillGraphStore:
    """Bind caller-selected versions and publish immutable graph snapshots.

    Version mappings are supplied by the outer scheduler. This service never
    chooses Skill membership or registers new Skill versions.
    """

    def __init__(
        self,
        database: Database,
        *,
        project_id: str,
        graph_id: str,
        actor_id: str,
        version_ids: Mapping[str, str],
        alias: str | None = None,
        message: str | None = None,
        expected_snapshot: UUID | None = None,
        check_head: bool = False,
    ):
        """Borrow storage and freeze the scheduler's version bindings.

        Args:
            database: Initialized database containing the existing graph tables.
            project_id: Project boundary applied to every read and write.
            graph_id: Graph identity within that project.
            alias: Project-unique shortcut for the graph.
            actor_id: Caller identity recorded for publication provenance.
            version_ids: Collection member ID to exact persisted version ID.
            message: Optional explanation recorded with a successful publication.
            expected_snapshot: Captured graph head before model work.
            check_head: Require the captured head at atomic publication.
        """
        if any(not value.strip() for value in (project_id, graph_id, actor_id)):
            raise ValueError("Project, graph, and actor IDs must be nonempty")
        if len(set(version_ids.values())) != len(version_ids):
            raise ValueError("Each collection member must reference a distinct stored version")
        self.database = database
        self.project_id = project_id
        self.graph_id = graph_id
        self.alias = alias or graph_id
        self.actor_id = actor_id
        self.message = message
        self.expected_snapshot = expected_snapshot
        self.check_head = check_head
        self._version_ids = dict(version_ids)

    @property
    def version_ids(self) -> dict[str, str]:
        """Return a detached copy of the exact caller-bound versions."""
        return dict(self._version_ids)

    @property
    def scope(self) -> dict[str, str]:
        """Return the project and graph isolation keys."""
        return {"project_id": self.project_id, "graph_id": self.graph_id}

    async def validate_skills(self, skills: Mapping[str, Skill]) -> dict[str, str]:
        """Verify that supplied objects exactly match scheduler-bound versions.

        Args:
            skills: Complete caller-selected candidate scope.

        Returns:
            A copy of the verified storage bindings for request hashing.
        """
        if set(skills) - set(self._version_ids):
            raise ValueError("Skill candidates require scheduler-provided version bindings")
        service = SkillDataService(self.database)
        for skill_id, skill in skills.items():
            record = await service.get(self._version_ids[skill_id])
            stored = await service.get_skill(record.version_id)
            if stored is None or stored.snapshot() != skill.snapshot():
                raise ValueError(f"Skill content differs from the bound stored version: {skill_id!r}")
        return {key: self._version_ids[key] for key in skills}

    async def resolve_families(self, member_ids: list[str]) -> dict[str, str]:
        """Resolve canonical families only for the caller-bound exact versions.

        Args:
            member_ids: Explicit candidate keys.

        Returns:
            Candidate key to canonical family identity.
        """
        service = SkillDataService(self.database)
        return {
            key: (await service.get(self._version_ids[key])).skill_id for key in member_ids
        }

    async def select_snapshot(self, snapshot_id: UUID) -> UUID:
        """Atomically select a ready reusable snapshot through the same head check.

        Args:
            snapshot_id: Ready immutable snapshot in this graph.

        Returns:
            Selected snapshot identity.
        """
        nodes, edges, metadata = await self.load(snapshot_id)
        version = await self.database.get(SKILL_GRAPH_VERSIONS_TABLE, {**self.scope, "graph_version_id": snapshot_id})
        return await self.publish(
            snapshot_id, request_hash=version["request_hash"], nodes=nodes, edges=edges, metadata=metadata
        )

    def bind_node(self, snapshot_id: UUID, skill_id: str, values: dict) -> SkillGraphNodeRecord:
        """Attach storage identity to algorithm-produced node features.

        Args:
            snapshot_id: New immutable graph snapshot identity.
            skill_id: Collection member owning this Skill or Scene node.
            values: Node features validated by the existing graph record schema.
        """
        if skill_id not in self._version_ids:
            raise ValueError("Node owner is outside the caller-selected scope")
        data = dict(values)
        data["metadata"] = {**data.get("metadata", {}), "collection_skill_id": skill_id}
        data["skill_version_id"] = self._version_ids[skill_id] if data.get("node_type", "skill") == "skill" else None
        return SkillGraphNodeRecord(**self.scope, graph_version_id=snapshot_id, **data)

    async def find(self, request_hash: str) -> UUID | None:
        """Find a committed build that is already the graph's current snapshot.

        Args:
            request_hash: Hash of the complete explicit source scope and settings.

        Returns:
            Current matching snapshot, or None to use transactional publication.
        """
        graph = await self.database.get(SKILL_GRAPHS_TABLE, self.scope)
        if graph is not None and graph["is_deleted"]:
            raise GraphNotFoundError("Graph not found")
        rows = await self.database.select(
            SKILL_GRAPH_VERSIONS_TABLE, filters={**self.scope, "request_hash": request_hash, "status": "ready"}, limit=1
        )
        if rows and graph is not None:
            return rows[0]["graph_version_id"]
        return None

    async def publish(
        self,
        snapshot_id: UUID,
        *,
        request_hash: str,
        nodes: list[SkillGraphNodeRecord],
        edges: list[SkillGraphEdgeRecord],
        metadata: dict,
    ) -> UUID:
        """Commit a complete snapshot and advance its head in one transaction.

        Args:
            snapshot_id: Identity shared by all supplied nodes and edges.
            request_hash: Idempotency identity for this exact build request.
            nodes: Complete validated Skill and Scene nodes, including vectors.
            edges: Complete validated edges with endpoints in this snapshot.
            metadata: Non-secret algorithm configuration and build evidence.

        Returns:
            Committed snapshot ID, including the original ID on an exact retry.
        """
        node_ids = {node.node_id for node in nodes}
        if len(node_ids) != len(nodes):
            raise ValueError("Graph node IDs must be unique")
        for record in [*nodes, *edges]:
            if (
                record.project_id != self.project_id
                or record.graph_id != self.graph_id
                or record.graph_version_id != snapshot_id
            ):
                raise ValueError("Graph record scope mismatch")
        by_id = {node.node_id: node for node in nodes}
        members = {}
        for node in nodes:
            skill_id = node.metadata.get("collection_skill_id")
            if node.node_type == "skill":
                if skill_id not in self._version_ids or node.skill_version_id != self._version_ids[skill_id]:
                    raise ValueError("Graph node differs from the scheduler's version binding")
                if skill_id in members:
                    raise ValueError("Duplicate collection member in graph")
                members[skill_id] = node
            else:
                owner = by_id.get(node.owner_node_id)
                if owner is None or owner.node_type != "skill" or owner.metadata.get("collection_skill_id") != skill_id:
                    raise ValueError("Scene owner is absent or belongs to another Skill")
        if any(edge.source_node_id not in node_ids or edge.target_node_id not in node_ids for edge in edges):
            raise ValueError("Graph edge endpoint is absent")
        digest = graph_digest(
            {
                "nodes": sorted((node.model_dump(mode="json") for node in nodes), key=lambda row: row["node_id"]),
                "edges": sorted((edge.model_dump(mode="json") for edge in edges), key=lambda row: row["edge_id"]),
                "metadata": metadata,
            }
        )
        async with self.database.transaction() as transaction:
            (
                await transaction.upsert(
                    SKILL_GRAPHS_TABLE,
                    [
                        SkillGraphRecord(
                            **self.scope,
                            alias=self.alias,
                            name=self.alias,
                        ).model_dump(),
                    ],
                )
            )
            # Start with a write to avoid upgrading a stale SQLite read snapshot.
            existing = await transaction.get(SKILL_GRAPHS_TABLE, self.scope)
            if existing is not None and existing["is_deleted"]:
                raise GraphNotFoundError("Graph not found")
            # The conditional write locks the graph before reading reusable snapshots.
            filters = {**self.scope, "is_deleted": False}
            if self.check_head:
                filters["current_version_id"] = self.expected_snapshot
            changed = await transaction.update(SKILL_GRAPHS_TABLE, {"updated_at": utcnow()}, filters=filters)
            if changed != 1:
                raise GraphConflictError("Graph head changed during construction")
            previous = await transaction.select(
                SKILL_GRAPH_VERSIONS_TABLE,
                filters={**self.scope, "request_hash": request_hash, "status": "ready"},
                limit=1,
            )
            if previous:
                (
                    await transaction.update(
                        SKILL_GRAPHS_TABLE,
                        {"current_version_id": previous[0]["graph_version_id"], "updated_at": utcnow()},
                        filters=self.scope,
                    )
                )
                return previous[0]["graph_version_id"]
            ready_at = utcnow()
            version = SkillGraphVersionRecord(
                **self.scope,
                graph_version_id=snapshot_id,
                public_version_id=f"skgv-{snapshot_id.hex}",
                status="ready",
                node_count=len(nodes),
                edge_count=len(edges),
                vector_count=sum(node.embedding is not None for node in nodes),
                snapshot_hash=digest,
                request_hash=request_hash,
                actor_id=self.actor_id,
                message=self.message,
                metadata=metadata,
                created_at=ready_at,
                ready_at=ready_at,
            )
            (await transaction.insert(SKILL_GRAPH_VERSIONS_TABLE, [version.model_dump()]))
            skill_nodes = [node.model_dump() for node in nodes if node.node_type == "skill"]
            scene_nodes = [node.model_dump() for node in nodes if node.node_type == "scene"]
            if skill_nodes:
                (await transaction.insert(SKILL_GRAPH_NODES_TABLE, skill_nodes))
            if scene_nodes:
                (await transaction.insert(SKILL_GRAPH_NODES_TABLE, scene_nodes))
            if edges:
                (await transaction.insert(SKILL_GRAPH_EDGES_TABLE, [edge.model_dump() for edge in edges]))
            (
                await transaction.update(
                    SKILL_GRAPHS_TABLE, {"current_version_id": snapshot_id, "updated_at": utcnow()}, filters=self.scope
                )
            )
        return snapshot_id

    async def _rows(self, table: str, snapshot_id: UUID) -> list[dict]:
        """Load all rows in a pinned snapshot without truncating large collections."""
        rows = []
        while True:
            page = await self.database.select(
                table, filters={**self.scope, "graph_version_id": snapshot_id}, limit=1000, offset=len(rows)
            )
            rows.extend(page)
            if len(page) < 1000:
                return rows

    async def load(self, snapshot_id: UUID) -> tuple[list[SkillGraphNodeRecord], list[SkillGraphEdgeRecord], dict]:
        """Load and verify a pinned immutable snapshot in this project and graph.

        Args:
            snapshot_id: Exact committed graph identity, never the mutable head.
        """
        graph = await self.database.get(SKILL_GRAPHS_TABLE, self.scope)
        if graph is None or graph["is_deleted"]:
            raise GraphNotFoundError("Graph not found")
        versions = await self._rows(SKILL_GRAPH_VERSIONS_TABLE, snapshot_id)
        if not versions or versions[0]["status"] != "ready":
            raise ValueError("Ready graph snapshot not found")
        version = SkillGraphVersionRecord.model_validate(versions[0])
        nodes = [
            SkillGraphNodeRecord.model_validate(row) for row in (await self._rows(SKILL_GRAPH_NODES_TABLE, snapshot_id))
        ]
        edges = [
            SkillGraphEdgeRecord.model_validate(row) for row in (await self._rows(SKILL_GRAPH_EDGES_TABLE, snapshot_id))
        ]
        digest = graph_digest(
            {
                "nodes": sorted((node.model_dump(mode="json") for node in nodes), key=lambda row: row["node_id"]),
                "edges": sorted((edge.model_dump(mode="json") for edge in edges), key=lambda row: row["edge_id"]),
                "metadata": version.metadata,
            }
        )
        if (len(nodes), len(edges), sum(node.embedding is not None for node in nodes), digest) != (
            version.node_count,
            version.edge_count,
            version.vector_count,
            version.snapshot_hash,
        ):
            raise ValueError("Stored graph snapshot failed integrity validation")
        return nodes, edges, version.metadata

    async def search(
        self, snapshot_id: UUID, vector: list[float], *, top_k: int, node_type: str | None = None, offset: int = 0
    ):
        """Search vectors only inside a pinned project-scoped graph snapshot.

        Args:
            snapshot_id: Immutable graph identity used by the current selection.
            vector: Query embedding validated by the database vector codec.
            top_k: Maximum number of seed nodes.
            node_type: Optional Skill or Scene restriction.
            offset: Sorted neighbors to skip for owner deduplication.
        """
        filters = {**self.scope, "graph_version_id": snapshot_id}
        if node_type is not None:
            filters["node_type"] = node_type
        hits = await self.database.search(
            SKILL_GRAPH_NODES_TABLE,
            vector_field="embedding",
            query_vector=vector,
            filters=filters,
            top_k=top_k + offset,
        )

        return hits[offset:]
