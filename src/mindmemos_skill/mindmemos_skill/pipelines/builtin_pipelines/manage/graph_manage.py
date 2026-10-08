"""Organize, recommend, and query project-scoped Skill Sets."""

from uuid import UUID, uuid4

from ....algos.manage.skill_manage import SkillManageAlgorithm
from ....api.typing import OrganizeRequest, OrganizeResult, RecommendRequest, RecommendResult
from ....common.asyncio import run_sync
from ....envs.typing import Task
from ....errors import GraphNotFoundError
from ....infra.task_manager.progress import ProgressReporter
from ....persistence.db_services.skill_management.service import GraphQueryService
from ....persistence.db_services.skill_management.writer import SkillGraphStore
from ....persistence.store import SkillPersistence
from ....skill import Skill
from ...base import Pipeline
from ...manage import ManagePipeline
from .config import GraphManagePipelineConfig


@Pipeline.register("graph_manage")
class GraphManagePipeline(ManagePipeline[GraphManagePipelineConfig]):
    """Publish graph snapshots, recommend versions, and query Skill Sets."""

    config_type = GraphManagePipelineConfig
    """Optional graph algorithm settings used by management operations."""

    async def resolve_graph(self, project_id: str, reference: str) -> dict:
        """Resolve one graph alias, family, or version in the authorized project.

        Args:
            project_id: Authorized project identity.
            reference: Alias or unique prefixed graph identity.

        Returns:
            Public identity and selected graph snapshot.
        """
        return await run_sync(self._queries(project_id).resolve, reference)

    async def delete_graph(self, project_id: str, reference: str) -> None:
        """Soft-delete a graph in the authorized project.

        Args:
            project_id: Authorized project identity.
            reference: Active graph identity or alias.
        """
        await run_sync(self._queries(project_id).delete, reference)

    def __init__(self, config: GraphManagePipelineConfig, *, persistence: SkillPersistence):
        """Borrow the shared store without performing graph or model I/O.

        Args:
            config: Complete algorithm options.
            persistence: Shared project-scoped Skill persistence.
        """
        if not isinstance(persistence, SkillPersistence):
            raise TypeError("Graph management requires SkillPersistence")
        super().__init__(config, persistence=persistence)

    async def organize(self, request: OrganizeRequest, *, reporter: ProgressReporter | None = None) -> OrganizeResult:
        """Publish or reuse the graph for exactly the supplied Skill versions.

        Args:
            request: Authorized project, graph, actor, versions, and scenes.
            reporter: Optional progress sink for graph publication.

        Returns:
            Immutable snapshot identity after successful publication.
        """
        if self.config.algorithm is None:
            raise NotImplementedError("Graph organization requires algorithm configuration")
        validated = OrganizeRequest.model_validate(request)
        if reporter is not None:
            await reporter.report(stage="organize", message="Organizing Skill Set")
        result = await run_sync(self._organize, validated)
        if reporter is not None:
            await reporter.report(stage="organized", message="Skill Set snapshot published")
        return result

    def _organize(self, request: OrganizeRequest) -> OrganizeResult:
        """Load versions and call the synchronous graph algorithm."""
        graph = None
        if request.graph_id is not None:
            if request.alias is not None:
                raise ValueError("An existing graph cannot receive a new alias")
            if request.graph_id.startswith("skgv-"):
                raise ValueError("A graph version cannot be used as a graph identity")
            graph = self._queries(request.project_id).graph(request.graph_id)
        graph_id = (
            graph["graph_id"]
            if graph is not None
            else (f"skg-{uuid4().hex}" if self.config.prefixed_graph_ids else (request.alias or uuid4().hex))
        )
        alias = graph.get("alias") if graph is not None else request.alias
        versions: dict[str, str] = {}
        skills: dict[str, Skill] = {}
        for reference in request.skill_refs:
            _, version = self.persistence.skills.resolve(request.project_id, reference)
            version_id = version.version_id
            if version.skill_id in versions:
                raise ValueError("Only one version per Skill family may be organized")
            source = self.persistence.skills.get_skill(version_id)
            if source is None:
                raise ValueError("Stored Skill version has no source")
            versions[version.skill_id] = version_id
            skills[version.skill_id] = source
        store = SkillGraphStore(
            self.persistence.database,
            project_id=request.project_id,
            graph_id=graph_id,
            alias=alias,
            actor_id=request.agent_id or request.user_id or "organize",
            version_ids=versions,
            message=request.commit_message,
        )
        algorithm = SkillManageAlgorithm(self.config.algorithm, store=store, scene_records={})
        algorithm.organize(skills=skills)
        snapshot_id = algorithm.snapshot_id
        if snapshot_id is None:
            raise RuntimeError("Organization completed without a graph snapshot")
        public_id = self._queries(request.project_id).graph(graph_id)["public_id"]
        return OrganizeResult(
            graph_id=public_id,
            graph_version_id=snapshot_id,
            skill_version_ids=list(versions.values()),
        )

    async def recommend(self, request: RecommendRequest) -> RecommendResult:
        """Recommend only versions in the specified immutable graph snapshot.

        Args:
            request: Authorized project, pinned graph, public task, and limit.

        Returns:
            Ordered exact version IDs belonging to the graph's project.
        """
        if self.config.algorithm is None:
            raise NotImplementedError("Graph recommendation requires algorithm configuration")
        return await run_sync(self._recommend, RecommendRequest.model_validate(request))

    def _recommend(self, request: RecommendRequest) -> RecommendResult:
        """Restore graph members and call the synchronous recommendation algorithm."""
        graph = self._queries(request.project_id).graph(request.graph_id)
        graph_id = graph["graph_id"]
        snapshot_id = request.graph_version_id
        if snapshot_id is None:
            snapshot_id = graph["current_version_id"]
            if snapshot_id is None:
                raise ValueError("Graph has no published organization")
        store = SkillGraphStore(
            self.persistence.database,
            project_id=request.project_id,
            graph_id=graph_id,
            actor_id="recommend",
            version_ids={},
        )
        nodes, _, _ = store.load(snapshot_id)
        versions: dict[str, str] = {}
        skills: dict[str, Skill] = {}
        for node in nodes:
            if node.node_type != "skill":
                continue
            skill_id = node.metadata.get("collection_skill_id")
            if not isinstance(skill_id, str) or not node.skill_version_id or skill_id in versions:
                raise ValueError("Graph has invalid Skill version bindings")
            version = self.persistence.skills.version_by_id(request.project_id, node.skill_version_id)
            if version.skill_id != skill_id:
                raise ValueError("Graph member differs from its stored Skill family")
            source = self.persistence.skills.get_skill(version.version_id)
            if source is None:
                raise ValueError("Stored Skill version has no source")
            versions[skill_id] = version.version_id
            skills[skill_id] = source
        store = SkillGraphStore(
            self.persistence.database,
            project_id=request.project_id,
            graph_id=graph_id,
            actor_id="recommend",
            version_ids=versions,
        )
        algorithm = SkillManageAlgorithm(self.config.algorithm, store=store, scene_records={})
        algorithm.restore(snapshot_id, skills=skills)
        selected = algorithm.recommend(task=Task(id="recommend", instruction=request.task), limit=request.top_k)
        return RecommendResult(
            skills=[
                self.persistence.skills.version_by_id(request.project_id, versions[skill_id]) for skill_id in selected
            ]
        )

    def _queries(self, project_id: str) -> GraphQueryService:
        """Borrow the active scoped database for the current graph representation."""
        return GraphQueryService(self.persistence.database, project_id)

    async def list_skill_sets(self, project_id: str) -> list[dict]:
        """List the project-owned Skill Sets and their publication pointers.

        Args:
            project_id: Authorized project identity.

        Returns:
            Skill Sets with stable public identities and representation modes.
        """
        rows = await run_sync(self._queries(project_id).list_graphs)
        result = []
        for row in rows:
            item = dict(row)
            internal_id = item.pop("graph_id")
            item["skill_set_id"] = item.pop("public_id")
            if item["current_version_id"] is not None:
                item["current_version_id"] = f"skgv-{item['current_version_id'].hex}"
            item["mode"] = "graph"
            result.append(item)
        return result

    async def list_skill_set_snapshots(self, project_id: str, skill_set_id: str) -> list[dict]:
        """List immutable snapshots and their publication revisions.

        Args:
            project_id: Authorized project identity.
            skill_set_id: Exact project-local Skill Set identity.

        Returns:
            Snapshots with stable public identities and revision numbers.
        """
        try:
            graph = await run_sync(self._queries(project_id).graph, skill_set_id)
        except GraphNotFoundError:
            return []
        public_id = graph["public_id"]
        rows = await run_sync(self._queries(project_id).snapshots, graph["graph_id"])
        result = []
        for row in rows:
            item = dict(row)
            item.pop("graph_id")
            item["skill_set_id"] = public_id
            item.pop("graph_version_id")
            item["snapshot_id"] = item.pop("public_version_id") or f"skgv-{row['graph_version_id'].hex}"
            item["mode"] = "graph"
            result.append(item)
        return result

    async def get_skill_set_snapshot(self, project_id: str, skill_set_id: str, snapshot_id: UUID) -> dict:
        """Read the selected snapshot's Skill members and relations.

        Args:
            project_id: Authorized project identity.
            skill_set_id: Exact project-local Skill Set identity.
            snapshot_id: Immutable snapshot identity.

        Returns:
            The current graph representation of the selected Skill Set.
        """
        graph = await run_sync(self._queries(project_id).graph, skill_set_id)
        public_id = graph["public_id"]
        row = await run_sync(self._queries(project_id).snapshot, graph["graph_id"], snapshot_id)
        item = dict(row)
        item["skill_set_id"] = public_id
        item.pop("graph_version_id")
        item["snapshot_id"] = f"skgv-{snapshot_id.hex}"
        item["mode"] = "graph"
        return item
