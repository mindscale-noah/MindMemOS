"""Organize, recommend, and query project-scoped graphs."""

import asyncio
from time import perf_counter
from uuid import UUID, uuid4

from ....agents import Agent
from ....algos.manage.skill_manage.algorithm import SkillManageAlgorithm
from ....algos.manage.skill_manage.extraction import SceneExtractor
from ....algos.manage.skill_manage.parsing import parse_skill_document
from ....api.typing import (
    AddGraphMembersRequest,
    GraphRecommendationScore,
    OrganizeRequest,
    OrganizeResult,
    RecommendRequest,
    RecommendResult,
    RemoveGraphMembersRequest,
    VectorRecommendationScore,
)
from ....envs.typing import Task
from ....errors import GraphNotFoundError
from ....logging import get_logger
from ....persistence.db_services.skill_management.service import GraphQueryService
from ....persistence.db_services.skill_management.writer import SkillGraphStore
from ....persistence.db_services.skills.typing import SceneData
from ....persistence.store import SkillPersistence
from ...base import Pipeline
from ...manage import ManagePipeline
from .config import GraphManagePipelineConfig


@Pipeline.register("graph_manage")
class GraphManagePipeline(ManagePipeline[GraphManagePipelineConfig]):
    """Publish graph snapshots, recommend versions, and query graphs."""

    config_type = GraphManagePipelineConfig
    """Validated construction, retrieval and query settings."""

    async def resolve_graph(self, project_id: str, reference: str) -> dict:
        """Resolve one graph alias, family, or version in the authorized project.

        Args:
            project_id: Authorized project identity.
            reference: Alias or unique prefixed graph identity.

        Returns:
            Public identity and selected graph snapshot.
        """
        return await self._queries(project_id).resolve(reference)

    async def delete_graph(self, project_id: str, reference: str) -> None:
        """Soft-delete a graph in the authorized project.

        Args:
            project_id: Authorized project identity.
            reference: Active graph identity or alias.
        """
        await self._queries(project_id).delete(reference)

    def __init__(self, config: GraphManagePipelineConfig, *, persistence: SkillPersistence):
        """Borrow the shared store without performing graph or model I/O.

        Args:
            config: Validated graph query options.
            persistence: Shared project-scoped Skill persistence.
        """
        if not isinstance(persistence, SkillPersistence):
            raise TypeError("Graph management requires SkillPersistence")
        super().__init__(config, persistence=persistence)

    def _algorithm(self, store, records=None):
        """Construct one request-owned algorithm with captured Scene records."""
        if self.config.algorithm is None:
            raise ValueError("Graph algorithm model settings are required")
        return SkillManageAlgorithm(self.config.algorithm, store=store, provided=records)

    async def _resolve_skill_versions(self, project_id, ref_skills):
        """Resolve every input before any extraction or publication."""
        result = {}
        for ref in ref_skills:
            family, version = await self.persistence.skills.resolve(project_id, ref)
            if family.skill_id in result:
                raise ValueError(
                    "Cannot construct graph: Multiple versions of skills with same skill id are not allowed."
                )
            result[family.skill_id] = version
        return result

    async def _get_skill_by_versions(self, skill_versions):
        """Load only the exact captured source versions."""
        result = {}
        for key, version in skill_versions.items():
            source = await self.persistence.skills.get_skill(version.version_id)
            if source is None:
                raise ValueError("Skill version not found")
            result[key] = source
        return result

    async def _get_or_extract_scene_records(self, project_id, versions, sources, force):
        """Read valid version caches or extract and merge validated replacements."""
        result = {}
        if self.config.algorithm is not None and not self.config.algorithm.build_scene and not force:
            return result
        if not versions:
            return result
        if self.config.algorithm is None:
            raise ValueError("Graph algorithm model settings are required")
        logger = get_logger(__name__)
        started = perf_counter()
        total = len(versions)
        completed = 0
        semaphore = asyncio.Semaphore(self.config.scene_concurrency)
        logger.info("Graph Scene processing started | total=%d concurrency=%d", total, self.config.scene_concurrency)

        async def process(key):
            """Validate, read or extract, and persist one exact version within the concurrency limit."""
            nonlocal completed
            async with semaphore:
                version = versions[key]
                item_started = perf_counter()
                logger.info(
                    "Graph Scene processing Skill | progress=%d/%d skill=%s version_id=%s elapsed_s=%.2f",
                    completed,
                    total,
                    version.name,
                    version.version_id,
                    item_started - started,
                )
                try:
                    # Keep the original document validation gate before extraction I/O.
                    parse_skill_document(
                        sources[key].content, skill_id=key, max_chars=self.config.algorithm.max_input_chars
                    )
                    data = None if force else await self.persistence.skills.get_scene_data(version.version_id)
                    cache_hit = data is not None
                    if data is None:
                        extractor = SceneExtractor(Agent.from_config(self.config.algorithm.chat))
                        record = await extractor.extract(sources[key].content)
                        data = SceneData(record=record)
                        await self.persistence.skills.set_scene_data(project_id, version.version_id, data)
                    completed += 1
                    logger.info(
                        "Graph Scene Skill completed | progress=%d/%d skill=%s version_id=%s "
                        "cache_hit=%s duration_s=%.2f elapsed_s=%.2f",
                        completed,
                        total,
                        version.name,
                        version.version_id,
                        cache_hit,
                        perf_counter() - item_started,
                        perf_counter() - started,
                    )
                    return key, data.record
                except Exception:
                    logger.exception(
                        "Graph Scene Skill failed | progress=%d/%d skill=%s version_id=%s elapsed_s=%.2f",
                        completed,
                        total,
                        version.name,
                        version.version_id,
                        perf_counter() - started,
                    )
                    raise

        tasks = [asyncio.create_task(process(key)) for key in sorted(versions)]
        try:
            return dict(await asyncio.gather(*tasks))
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        finally:
            logger.info(
                "Graph Scene processing finished | progress=%d/%d elapsed_s=%.2f",
                completed,
                total,
                perf_counter() - started,
            )

    async def _resolve_graph(self, project_id, reference):
        """Resolve a modifiable graph family and capture its current head once."""
        graph = await self._queries(project_id).graph(reference)
        return graph

    async def _members(self, project_id, graph):
        """Read exact members and semantic nodes from one immutable head."""
        if graph["current_version_id"] is None:
            return {}, None
        store = SkillGraphStore(
            self.persistence.database,
            project_id=project_id,
            graph_id=graph["graph_id"],
            actor_id=project_id,
            version_ids={},
        )
        nodes, edges, metadata = await store.load(graph["current_version_id"])
        versions = {}
        for node in nodes:
            if node.node_type == "skill":
                version = await self.persistence.skills.get(node.skill_version_id)
                versions[version.skill_id] = version
        return versions, (nodes, edges, metadata)

    async def _publish(self, request, graph_id, versions, sources, records, graph=None, prior=None):
        """Perform model work outside storage transactions and conditionally publish."""
        store = SkillGraphStore(
            self.persistence.database,
            project_id=request.project_id,
            graph_id=graph_id,
            actor_id=request.agent_id or request.user_id or request.project_id,
            alias=graph["alias"] if graph else request.alias,
            message=request.commit_message,
            version_ids={key: value.version_id for key, value in versions.items()},
            expected_snapshot=graph["current_version_id"] if graph else None,
            check_head=True,
        )
        algorithm = self._algorithm(store, records)
        if prior is not None:
            nodes, edges, metadata = prior
            algorithm.reuse_snapshot(graph["current_version_id"], nodes=nodes, edges=edges, metadata=metadata)
        get_logger(__name__).info("Graph linking and embedding started")
        await algorithm.organize(skills=sources)
        get_logger(__name__).info("Graph publication completed")
        return OrganizeResult(
            graph_id=graph_id,
            graph_version_id=algorithm.snapshot_id,
            skill_version_ids=[versions[key].version_id for key in sorted(versions)],
        )

    async def organize(self, request: OrganizeRequest) -> OrganizeResult:
        """Build the complete selected scope and atomically publish its snapshot.

        Args:
            request: Authorized references and optional Scene refresh.

        Returns:
            Actual published or reused snapshot and its exact members.
        """
        if self.config.algorithm is None:
            raise ValueError("Run skill organize requires algorithm settings")
        if self.config.algorithm.chat is None:
            raise ValueError("Run skill organize requires chat model settings")
        if self.config.algorithm.embedding is None:
            raise ValueError("Run skill organize requires embedding model settings")

        graph = await self._resolve_graph(request.project_id, request.graph_id) if request.graph_id else None
        if graph is not None and request.alias is not None:
            raise ValueError("Existing graph aliases cannot be changed by organize")
        versions = await self._resolve_skill_versions(request.project_id, request.skill_refs)
        sources = await self._get_skill_by_versions(versions)
        records = await self._get_or_extract_scene_records(
            request.project_id, versions, sources, request.force_extract_scene
        )
        prior = (await self._members(request.project_id, graph))[1] if graph else None
        return await self._publish(
            request, graph["graph_id"] if graph else "skg-" + uuid4().hex, versions, sources, records, graph, prior
        )

    async def add_graph_members(self, request: AddGraphMembersRequest) -> OrganizeResult:
        """Validate the entire batch, then add new families without replacing versions.

        Args:
            request: Graph family and new member references.

        Returns:
            Snapshot containing all prior members and the new members.
        """
        graph = await self._resolve_graph(request.project_id, request.graph_ref)
        current, prior = await self._members(request.project_id, graph)
        added = await self._resolve_skill_versions(request.project_id, request.skill_refs)
        if set(current) & set(added):
            raise ValueError("Skill family already belongs to the graph")
        new_sources = await self._get_skill_by_versions(added)
        records = await self._get_or_extract_scene_records(
            request.project_id, added, new_sources, request.force_extract_scene
        )
        versions = {**current, **added}
        sources = await self._get_skill_by_versions(versions)
        if self.config.algorithm is not None and self.config.algorithm.build_scene and prior is not None:
            scene_members = {node.metadata["collection_skill_id"] for node in prior[0] if node.node_type == "scene"}
            missing = {key: value for key, value in current.items() if key not in scene_members}
            records.update(await self._get_or_extract_scene_records(request.project_id, missing, sources, False))
        return await self._publish(request, graph["graph_id"], versions, sources, records, graph, prior)

    async def remove_graph_members(self, request: RemoveGraphMembersRequest) -> OrganizeResult:
        """Remove a complete validated family batch, preserving Skill source and history.

        Args:
            request: Graph family and member references.

        Returns:
            Published remaining scope, including an empty graph.
        """
        graph = await self._resolve_graph(request.project_id, request.graph_ref)
        current, prior = await self._members(request.project_id, graph)
        removed = await self._resolve_skill_versions(request.project_id, request.skill_refs)
        if not set(removed) <= set(current):
            raise ValueError("Skill family is not a graph member")
        versions = {key: value for key, value in current.items() if key not in removed}
        sources = await self._get_skill_by_versions(versions)
        records = {}
        return await self._publish(request, graph["graph_id"], versions, sources, records, graph, prior)

    async def recommend(self, request: RecommendRequest) -> RecommendResult:
        """Rank exact versions from one pinned snapshot using graph or cosine scores.

        Args:
            request: Public instruction and strategy within the authorized graph.

        Returns:
            Selected versions and original score components in the same order.
        """
        graph = await self._queries(request.project_id).graph(request.graph_id)
        snapshot = request.graph_version_id or graph["current_version_id"]
        if snapshot is None:
            raise ValueError("Graph has no published snapshot")
        store = SkillGraphStore(
            self.persistence.database,
            project_id=request.project_id,
            graph_id=graph["graph_id"],
            actor_id=request.project_id,
            version_ids={},
        )
        nodes, edges, metadata = await store.load(snapshot)
        versions = {
            node.metadata["collection_skill_id"]: await self.persistence.skills.get(node.skill_version_id)
            for node in nodes
            if node.node_type == "skill"
        }
        store = SkillGraphStore(
            self.persistence.database,
            project_id=request.project_id,
            graph_id=graph["graph_id"],
            actor_id=request.project_id,
            version_ids={key: value.version_id for key, value in versions.items()},
        )
        algorithm = self._algorithm(store)
        task = Task(id="recommend", instruction=request.task)
        if request.strategy == "vector":
            algorithm.validate_vector_metadata(metadata)
            algorithm.bind_vector_snapshot(
                snapshot,
                skill_ids=set(versions),
                skill_versions={key: value.version_id for key, value in versions.items()},
            )
            result = await algorithm.retrieve_vector(
                task=task, limit=request.top_k, include_scenes=request.include_scenes
            )
            scores = [
                VectorRecommendationScore(
                    skill_version_id=versions[score.member_id].version_id,
                    similarity=score.similarity,
                    node_id=score.node_id,
                    node_type=score.node_type,
                )
                for score in result.scores[: request.top_k]
            ]
        else:
            await algorithm.load_snapshot(
                snapshot,
                skills=await self._get_skill_by_versions(versions),
                nodes=nodes,
                edges=edges,
                metadata=metadata,
            )
            result = await algorithm.retrieve(task=task, limit=request.top_k)
            scores = [
                GraphRecommendationScore(
                    skill_version_id=versions[score.member_id].version_id,
                    ppr_score=score.ppr_score,
                    scene_boost=score.scene_boost,
                    graph_score=score.graph_score,
                    rerank_score=score.rerank_score,
                )
                for score in result.scores[: request.top_k]
            ]
        return RecommendResult(
            skills=[versions[score.member_id] for score in result.scores[: request.top_k]],
            graph_version_id=snapshot,
            strategy=request.strategy,
            score_type=result.score_type,
            scores=scores,
            include_scenes=request.include_scenes,
        )

    async def preview_graph(self, project_id: str, graph_ref: str) -> dict:
        """Read the published snapshot selected by an authorized reference.

        Args:
            project_id: Authorized project identity.
            graph_ref: Graph family, alias, or immutable snapshot reference.

        Returns:
            Complete published topology without model calls or writes.
        """
        selected = await self._queries(project_id).resolve(graph_ref)
        version = selected["graph_version_id"]
        if version is None:
            raise ValueError("Graph has no published snapshot")
        snapshot_id = UUID(version.removeprefix("skgv-"))
        return await self.get_graph_snapshot(project_id, selected["graph_id"], snapshot_id)

    def _queries(self, project_id: str) -> GraphQueryService:
        """Borrow the active scoped database for the current graph representation."""
        return GraphQueryService(self.persistence.database, project_id)

    async def list_graph(self, project_id: str) -> list[dict]:
        """List the project-owned graphs and their publication pointers.

        Args:
            project_id: Authorized project identity.

        Returns:
            Graphs with stable public identities and publication pointers.
        """
        rows = await self._queries(project_id).list_graphs()
        result = []
        for row in rows:
            item = dict(row)
            if item["current_version_id"] is not None:
                item["current_version_id"] = f"skgv-{item['current_version_id'].hex}"
            result.append(item)
        return result

    async def list_graph_snapshots(self, project_id: str, graph_id: str) -> list[dict]:
        """List immutable snapshots and their creation metadata.

        Args:
            project_id: Authorized project identity.
            graph_id: Exact project-local graph identity.

        Returns:
            Snapshots with stable public identities and creation metadata.
        """
        try:
            graph = await self._queries(project_id).graph(graph_id)
        except GraphNotFoundError:
            return []
        rows = await self._queries(project_id).snapshots(graph["graph_id"])
        result = []
        for row in rows:
            item = dict(row)
            item["graph_version_id"] = item.pop("public_version_id") or f"skgv-{row['graph_version_id'].hex}"
            result.append(item)
        return result

    async def get_graph_snapshot(self, project_id: str, graph_id: str, snapshot_id: UUID) -> dict:
        """Read the selected snapshot's Skill members and relations.

        Args:
            project_id: Authorized project identity.
            graph_id: Exact project-local graph identity.
            snapshot_id: Immutable snapshot identity.

        Returns:
            The current graph representation of the selected graph.
        """
        graph = await self._queries(project_id).graph(graph_id)
        row = await self._queries(project_id).snapshot(graph["graph_id"], snapshot_id)
        item = dict(row)
        item["graph_id"] = graph["graph_id"]
        item["graph_version_id"] = f"skgv-{snapshot_id.hex}"
        return item
