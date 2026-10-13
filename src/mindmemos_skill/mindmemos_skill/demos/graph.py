"""Publish deterministic demo graphs using the real graph storage contract."""

from time import sleep
from uuid import UUID

from mindmemos_skill.common.ids import generate_id

from ..api.typing import GraphRecommendationScore, OrganizeRequest, OrganizeResult, RecommendRequest, RecommendResult
from ..logging import get_logger
from ..persistence.db_services.graphs.typing import GraphNodeType, SkillGraphEdgeRecord
from ..persistence.db_services.graphs.writer import SkillGraphStore, graph_digest
from ..pipelines import Pipeline
from ..pipelines.builtin_pipelines.manage.graph_manage import GraphManagePipeline
from .config import DemoGraphConfig


@Pipeline.register("demo_graph")
class DemoGraphPipeline(GraphManagePipeline):
    """Build a chain of Skill nodes with no embedding or extraction calls."""

    config_type = DemoGraphConfig
    """Graph identity settings and local publication delay."""

    async def organize(self, request: OrganizeRequest) -> OrganizeResult:
        """Publish a synthetic graph for the supplied version collection.

        Args:
            request: Authorized graph identity and explicit Skill references.

        Returns:
            A real immutable graph snapshot identity.
        """
        get_logger(__name__).info(
            "algorithm.progress", extra={"event_body": {"stage": "organize", "message": "Building offline demo graph"}}
        )
        result = await self._organize(OrganizeRequest.model_validate(request))
        get_logger(__name__).info(
            "algorithm.progress",
            extra={"event_body": {"stage": "organized", "message": "Demo graph snapshot published"}},
        )
        return result

    async def recommend(self, request: RecommendRequest) -> RecommendResult:
        """Rank pinned graph members by simple word overlap.

        Args:
            request: Authorized graph snapshot, public query, and result limit.

        Returns:
            Exact member versions ranked without embedding or model calls.
        """
        if request.strategy != "graph":
            raise NotImplementedError("Demo graphs do not support vector retrieval")
        return await self._recommend(RecommendRequest.model_validate(request))

    async def _recommend(self, request: RecommendRequest) -> RecommendResult:
        """Restore a pinned snapshot and return deterministically ranked members."""
        graph = await self.graphs.get_graph(request.graph_id)
        store = SkillGraphStore(
            self.persistence.database,
            graph_id=graph.graph_id,
            actor_id="demo-recommend",
            version_ids={},
        )
        snapshot_id = request.graph_version_id or graph.current_version_id
        nodes, _, _ = await store.load(snapshot_id)
        words = set(request.task.lower().split())
        members = sorted(
            (node for node in nodes if node.node_type == GraphNodeType.SKILL),
            key=lambda node: (-len(words & set((node.name + " " + node.description).lower().split())), node.node_id),
        )
        selected = members[: request.top_k]
        versions = [await self.persistence.skills.get(node.skill_version_id) for node in selected]
        return RecommendResult(
            skills=versions,
            graph_version_id=snapshot_id,
            strategy="graph",
            score_type="graph_components",
            scores=[
                GraphRecommendationScore(
                    skill_version_id=version.version_id,
                    ppr_score=0,
                    scene_boost=0,
                    graph_score=0,
                    rerank_score=len(words & set((node.name + " " + node.description).lower().split())),
                )
                for version, node in zip(versions, selected, strict=True)
            ],
        )

    async def _organize(self, request: OrganizeRequest) -> OrganizeResult:
        """Resolve exact versions and atomically publish a simple dependency chain."""
        if request.graph_id is not None:
            if request.alias is not None:
                raise ValueError("An existing graph cannot receive a new alias")
            graph = await self.graphs.get_graph(request.graph_id)
            graph_id, alias = graph.graph_id, graph.alias
        else:
            graph_id, alias = generate_id("skg-"), request.alias
        versions, skills = {}, {}
        for reference in request.skill_refs:
            owner, version = await self.persistence.skills.resolve(reference)
            if owner.skill_id in versions:
                raise ValueError("Only one version per Skill family may be organized")
            versions[owner.skill_id] = version.version_id
            skills[owner.skill_id] = await self.persistence.skills.get_skill(version.version_id)
            sleep(self.config.delay_seconds)
        store = SkillGraphStore(
            self.persistence.database,
            graph_id=graph_id,
            alias=alias,
            actor_id=request.agent_id or request.user_id or "demo",
            version_ids=versions,
            message=request.commit_message,
        )
        await store.validate_skills(skills)
        snapshot_id = generate_id("skgv-")
        nodes = [
            store.bind_node(
                snapshot_id,
                skill_id,
                {
                    "node_id": skill_id,
                    "name": skill.source.name,
                    "description": skill.source.description,
                    "rendered_snippet": skill.content,
                    "metadata": {"demo": True},
                },
            )
            for skill_id, skill in skills.items()
        ]
        edges = [
            SkillGraphEdgeRecord(
                project_id=request.project_id,
                graph_id=graph_id,
                graph_version_id=snapshot_id,
                edge_id=UUID(generate_id()),
                source_node_id=left.node_id,
                target_node_id=right.node_id,
                edge_type="flow",
            )
            for left, right in zip(nodes, nodes[1:])
        ]
        snapshot_id = await store.publish(
            snapshot_id,
            request_hash=graph_digest({"demo": True, "versions": versions}),
            nodes=nodes,
            edges=edges,
            metadata={"demo": True, "llm_calls": 0},
        )
        return OrganizeResult(
            graph_id=graph_id, graph_version_id=snapshot_id, skill_version_ids=list(versions.values())
        )
