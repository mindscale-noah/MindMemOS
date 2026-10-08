"""Publish deterministic demo graphs using the real graph storage contract."""

from time import sleep
from uuid import uuid4

from ..api.typing import OrganizeRequest, OrganizeResult, RecommendRequest, RecommendResult
from ..common.asyncio import run_sync
from ..persistence.db_services.skill_management.typing import SkillGraphEdgeRecord
from ..persistence.db_services.skill_management.writer import SkillGraphStore, graph_digest
from ..pipelines import Pipeline
from ..pipelines.builtin_pipelines.manage.graph_manage import GraphManagePipeline
from .config import DemoGraphConfig


@Pipeline.register("demo_graph")
class DemoGraphPipeline(GraphManagePipeline):
    """Build a chain of Skill nodes with no embedding or extraction calls."""

    config_type = DemoGraphConfig
    """Graph identity settings and local publication delay."""

    async def organize(self, request: OrganizeRequest, *, reporter=None) -> OrganizeResult:
        """Publish a synthetic graph for the supplied version collection.

        Args:
            request: Authorized graph identity and explicit Skill references.
            reporter: Optional event sink for direct service calls.

        Returns:
            A real immutable graph snapshot identity.
        """
        if reporter is not None:
            await reporter.report(stage="organize", message="Building offline demo graph")
        result = await run_sync(self._organize, OrganizeRequest.model_validate(request))
        if reporter is not None:
            await reporter.report(stage="organized", message="Demo graph snapshot published")
        return result

    async def recommend(self, request: RecommendRequest) -> RecommendResult:
        """Rank pinned graph members by simple word overlap.

        Args:
            request: Authorized graph snapshot, public query, and result limit.

        Returns:
            Exact member versions ranked without embedding or model calls.
        """
        return await run_sync(self._recommend, RecommendRequest.model_validate(request))

    def _recommend(self, request: RecommendRequest) -> RecommendResult:
        """Restore a pinned snapshot and return deterministically ranked members."""
        graph = self._queries(request.project_id).graph(request.graph_id)
        store = SkillGraphStore(
            self.persistence.database,
            project_id=request.project_id,
            graph_id=graph["graph_id"],
            actor_id="demo-recommend",
            version_ids={},
        )
        snapshot_id = request.graph_version_id or graph["current_version_id"]
        nodes, _, _ = store.load(snapshot_id)
        words = set(request.task.lower().split())
        members = sorted(
            (node for node in nodes if node.node_type == "skill"),
            key=lambda node: (-len(words & set((node.name + " " + node.description).lower().split())), node.node_id),
        )
        return RecommendResult(
            skills=[
                self.persistence.skills.version_by_id(request.project_id, node.skill_version_id)
                for node in members[: request.top_k]
            ]
        )

    def _organize(self, request: OrganizeRequest) -> OrganizeResult:
        """Resolve exact versions and atomically publish a simple dependency chain."""
        if request.graph_id is not None:
            if request.alias is not None:
                raise ValueError("An existing graph cannot receive a new alias")
            graph = self._queries(request.project_id).graph(request.graph_id)
            graph_id, alias = graph["graph_id"], graph["alias"]
        else:
            graph_id, alias = f"skg-{uuid4().hex}", request.alias
        versions, skills = {}, {}
        for reference in request.skill_refs:
            owner, version = self.persistence.skills.resolve(request.project_id, reference)
            if owner.skill_id in versions:
                raise ValueError("Only one version per Skill family may be organized")
            versions[owner.skill_id] = version.version_id
            skills[owner.skill_id] = self.persistence.skills.get_skill(version.version_id)
            sleep(self.config.delay_seconds)
        store = SkillGraphStore(
            self.persistence.database,
            project_id=request.project_id,
            graph_id=graph_id,
            alias=alias,
            actor_id=request.agent_id or request.user_id or "demo",
            version_ids=versions,
            message=request.commit_message,
        )
        store.validate_skills(skills)
        snapshot_id = uuid4()
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
                edge_id=uuid4(),
                source_node_id=left.node_id,
                target_node_id=right.node_id,
                edge_type="flow",
            )
            for left, right in zip(nodes, nodes[1:])
        ]
        snapshot_id = store.publish(
            snapshot_id,
            request_hash=graph_digest({"demo": True, "versions": versions}),
            nodes=nodes,
            edges=edges,
            metadata={"demo": True, "llm_calls": 0},
        )
        return OrganizeResult(
            graph_id=graph_id, graph_version_id=snapshot_id, skill_version_ids=list(versions.values())
        )
