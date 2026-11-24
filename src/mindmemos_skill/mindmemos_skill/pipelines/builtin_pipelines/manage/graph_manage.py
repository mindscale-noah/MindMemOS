"""Organize, recommend, and query project-scoped graphs."""

import asyncio
from collections import Counter
from collections.abc import Mapping, Sequence
from time import perf_counter
from typing import cast

from ....agents import Agent, OpenAIChatAgent
from ....algos.manage.skill_manage.algorithm import SkillManageAlgorithm
from ....algos.manage.skill_manage.config import SkillManageConfig
from ....algos.manage.skill_manage.constant import (
    METADATA_COLLECTION_SKILL_ID,
)
from ....algos.manage.skill_manage.errors import SceneGraphError, SceneGraphFailure
from ....algos.manage.skill_manage.extraction import SceneExtractor
from ....algos.manage.skill_manage.parsing import parse_skill_document
from ....algos.manage.typing import GeneratedGraph, GraphSnapshot
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
from ....common.ids import generate_id
from ....envs.typing import Task
from ....errors import GraphNotFoundError
from ....infra.model_access.openai import resolve_embedding_space
from ....logging import get_logger
from ....persistence.db_services.graphs.service import GraphDataService
from ....persistence.db_services.graphs.typing import GraphNodeType, GraphResolution, SkillGraphRecord
from ....persistence.db_services.graphs.writer import SkillGraphStore
from ....persistence.db_services.skills.typing import SceneData, SkillVersionRecord
from ....persistence.store import SkillPersistence
from ....scene.typing import ProvidedSceneRecord
from ....skill import Skill
from ...base import Pipeline
from ...manage import ManagePipeline
from ...typing import GraphListItem, GraphSnapshotListItem, GraphTopology
from .build_config import pin_embedding_space, restore_build_config
from .config import GraphManagePipelineConfig


@Pipeline.register("graph_manage")
class GraphManagePipeline(ManagePipeline[GraphManagePipelineConfig]):
    """Publish graph snapshots, recommend versions, and query graphs."""

    config_type = GraphManagePipelineConfig
    """Validated construction, retrieval and query settings."""

    persistence: SkillPersistence
    """Required project-scoped storage borrowed from the active service session."""

    def __init__(self, config: GraphManagePipelineConfig, *, persistence: SkillPersistence) -> None:
        """Borrow the shared store without performing graph or model I/O.

        Args:
            config: Validated graph query options.
            persistence: Shared project-scoped Skill persistence.
        """
        if not isinstance(persistence, SkillPersistence):
            raise TypeError("Graph management requires SkillPersistence")
        super().__init__(config, persistence=persistence)

    @property
    def graphs(self) -> GraphDataService:
        """Return the graph service bound to the active persistence session."""
        return self.persistence.graphs

    def _algorithm(
        self,
        store: SkillGraphStore,
        records: Mapping[str, ProvidedSceneRecord | None] | None = None,
        config: SkillManageConfig | None = None,
    ) -> SkillManageAlgorithm:
        """Construct one request-owned algorithm with captured Scene records."""
        if self.config.algorithm is None:
            raise ValueError("Graph algorithm model settings are required")
        return SkillManageAlgorithm(config or self.config.algorithm, store=store, provided=records)

    def _build_config(self, prior: GraphSnapshot | None) -> SkillManageConfig:
        """Capture defaults for a new graph or restore the exact prior build settings."""
        config = self.config.algorithm
        if config is None:
            raise ValueError("Graph algorithm model settings are required")
        return restore_build_config(config, prior.metadata) if prior is not None else config.model_copy(deep=True)

    async def _resolve_skill_versions(self, references: Sequence[str]) -> dict[str, SkillVersionRecord]:
        """Resolve every input before any extraction or publication."""
        result: dict[str, SkillVersionRecord] = {}
        for ref in references:
            family, version = await self.persistence.skills.resolve(ref)
            if family.skill_id in result:
                raise ValueError(
                    "Cannot construct graph: Multiple versions of skills with same skill id are not allowed."
                )
            result[family.skill_id] = version
        return result

    def _skills_from_versions(self, versions: Mapping[str, SkillVersionRecord]) -> dict[str, Skill]:
        """Restore Skill objects directly from the versions captured by this Pipeline."""
        return {key: Skill.from_snapshot(version.snapshot()) for key, version in versions.items()}

    async def _get_or_extract_scene_records(
        self,
        versions: Mapping[str, SkillVersionRecord],
        sources: Mapping[str, Skill],
        force: bool,
        config: SkillManageConfig | None = None,
    ) -> dict[str, ProvidedSceneRecord | None]:
        """Read valid version caches or extract and merge validated replacements."""
        if not versions:
            return {}
        config = config or self._build_config(None)
        if not config.build_scene and not force:
            return {}
        logger = get_logger(__name__)
        started = perf_counter()
        total = len(versions)
        completed = 0
        cache_hits = 0
        without_scenes = 0
        semaphore = asyncio.Semaphore(self.config.scene_concurrency)
        logger.info("Graph Scene processing started | total=%d concurrency=%d", total, self.config.scene_concurrency)

        async def process(key: str) -> tuple[str, ProvidedSceneRecord | None]:
            """Validate, read or extract, and persist one exact version within the concurrency limit."""
            nonlocal completed, cache_hits, without_scenes
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
                    parse_skill_document(sources[key].content, skill_id=key, max_chars=config.max_input_chars)
                    data = None if force else await self.persistence.skills.get_scene_data(version.version_id)
                    cache_hit = data is not None
                    cache_hits += int(cache_hit)
                    if data is None:
                        if config.chat is None:
                            raise ValueError("Scene extraction requires chat model settings")
                        client = cast(OpenAIChatAgent, Agent.from_config(config.chat))
                        extractor = SceneExtractor(client)
                        try:
                            record = await extractor.extract(sources[key].content)
                        except SceneGraphError as error:
                            if error.code != SceneGraphFailure.INSUFFICIENT_SOURCE:
                                raise
                            record = None
                        data = SceneData(record=record)
                        await self.persistence.skills.set_scene_data(version.version_id, data)
                    if data.record is None:
                        without_scenes += 1
                        logger.info(
                            "Graph Skill retained without Scenes | skill=%s version_id=%s reason=insufficient_source cache_hit=%s",
                            version.name,
                            version.version_id,
                            cache_hit,
                        )
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
                "Graph Scene processing finished | progress=%d/%d elapsed_s=%.2f cache_hits=%d extracted=%d without_scenes=%d",
                completed,
                total,
                perf_counter() - started,
                cache_hits,
                completed - cache_hits,
                without_scenes,
            )

    async def _resolve_graph(self, reference: str) -> SkillGraphRecord:
        """Resolve a modifiable graph family and capture its current head once."""
        return await self.graphs.get_graph(reference)

    async def _read_prior_graph(self, graph: SkillGraphRecord | None) -> GraphSnapshot | None:
        """Read the captured immutable head without resolving its member versions."""
        if graph is None or graph.current_version_id is None:
            return None
        snapshot_id = graph.current_version_id
        nodes, edges, metadata = await self.graphs.load(graph.graph_id, snapshot_id)
        return GraphSnapshot(snapshot_id=snapshot_id, nodes=nodes, edges=edges, metadata=metadata)

    async def _get_member_versions(self, prior: GraphSnapshot | None) -> dict[str, SkillVersionRecord]:
        """Resolve the exact Skill versions referenced by a captured graph."""
        versions: dict[str, SkillVersionRecord] = {}
        if prior is not None:
            for node in prior.nodes:
                if node.node_type == GraphNodeType.SKILL:
                    if node.skill_version_id is None:
                        raise ValueError("Skill node requires a stored version")
                    version = await self.persistence.skills.get(node.skill_version_id)
                    versions[version.skill_id] = version
        return versions

    def _graph_store(
        self,
        request: OrganizeRequest | AddGraphMembersRequest | RemoveGraphMembersRequest,
        graph_id: str,
        versions: Mapping[str, SkillVersionRecord],
        graph: SkillGraphRecord | None,
    ) -> SkillGraphStore:
        """Capture version bindings and the expected graph head for publication."""
        alias = graph.alias if graph is not None else None
        if graph is None and isinstance(request, OrganizeRequest):
            alias = request.alias
        return SkillGraphStore(
            self.persistence.database,
            graph_id=graph_id,
            actor_id=request.agent_id or request.user_id or request.project_id,
            alias=alias,
            message=request.commit_message,
            version_ids={key: value.version_id for key, value in versions.items()},
            expected_snapshot=graph.current_version_id if graph else None,
            check_head=True,
        )

    async def _generate_graph(
        self,
        store: SkillGraphStore,
        sources: Mapping[str, Skill],
        records: Mapping[str, ProvidedSceneRecord | None],
        prior: GraphSnapshot | None,
        config: SkillManageConfig | None = None,
    ) -> GeneratedGraph:
        """Generate complete graph data, reusing the prior snapshot where applicable."""
        algorithm = self._algorithm(store, records, config or self._build_config(prior))
        if prior is not None:
            algorithm.reuse_snapshot(prior.snapshot_id, nodes=prior.nodes, edges=prior.edges, metadata=prior.metadata)
        get_logger(__name__).info("Graph generation started")
        generated = await algorithm.organize(skills=sources)
        get_logger(__name__).info(
            "Graph generation completed | graph_id=%s graph_version_id=%s skills=%d scenes=%d nodes=%d edges=%d edge_types=%s",
            store.graph_id,
            generated.snapshot_id,
            sum(node.node_type == GraphNodeType.SKILL for node in generated.nodes),
            sum(node.node_type == GraphNodeType.SCENE for node in generated.nodes),
            len(generated.nodes),
            len(generated.edges),
            dict(sorted(Counter(edge.edge_type for edge in generated.edges).items())),
        )
        return generated

    async def _publish_graph(self, store: SkillGraphStore, generated: GeneratedGraph) -> OrganizeResult:
        """Atomically publish generated graph data with the captured head check."""
        get_logger(__name__).info("Graph atomic publication started")
        snapshot_id = await store.publish(
            generated.snapshot_id,
            request_hash=generated.request_hash,
            nodes=generated.nodes,
            edges=generated.edges,
            metadata=generated.metadata,
        )
        get_logger(__name__).info("Graph publication completed")
        versions = store.version_ids
        return OrganizeResult(
            graph_id=store.graph_id,
            graph_version_id=snapshot_id,
            skill_version_ids=[versions[key] for key in sorted(versions)],
        )

    async def resolve_graph(self, reference: str) -> GraphResolution:
        """Resolve one graph alias, family, or version in the authorized project.

        Args:
            reference: Alias or unique prefixed graph identity.

        Returns:
            Public identity and selected graph snapshot.
        """
        return await self.graphs.resolve(reference)

    async def delete_graph(self, reference: str) -> None:
        """Soft-delete a graph in the authorized project.

        Args:
            reference: Active graph identity or alias.
        """
        await self.graphs.delete(reference)

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

        graph = await self._resolve_graph(request.graph_id) if request.graph_id else None
        if graph is not None and request.alias is not None:
            raise ValueError("Existing graph aliases cannot be changed by organize")
        versions = await self._resolve_skill_versions(request.skill_refs)
        get_logger(__name__).info("Graph organize inputs resolved | skills=%d", len(versions))
        sources = self._skills_from_versions(versions)
        prior = await self._read_prior_graph(graph)
        config = self._build_config(prior)
        embedding = self.config.algorithm.embedding
        space = await resolve_embedding_space(embedding)
        if space is not None:
            pinned = pin_embedding_space(config, embedding, space)
            if pinned.embedding_profile != config.embedding_profile or config.embedding.space_id != space.space_id:
                prior = None
            config = pinned
        captured = (
            {
                node.metadata[METADATA_COLLECTION_SKILL_ID]: node.skill_version_id
                for node in prior.nodes
                if node.node_type == GraphNodeType.SKILL
            }
            if prior is not None
            else {}
        )
        pending = {
            key: version
            for key, version in versions.items()
            if request.force_extract_scene or captured.get(key) != version.version_id
        }
        records = await self._get_or_extract_scene_records(pending, sources, request.force_extract_scene, config)
        graph_id = graph.graph_id if graph else generate_id("skg-")
        store = self._graph_store(request, graph_id, versions, graph)
        generated = await self._generate_graph(store, sources, records, prior, config)
        return await self._publish_graph(store, generated)

    async def add_graph_members(self, request: AddGraphMembersRequest) -> OrganizeResult:
        """Validate the entire batch, then add new families without replacing versions.

        Args:
            request: Graph family and new member references.

        Returns:
            Snapshot containing all prior members and the new members.
        """
        graph = await self._resolve_graph(request.graph_ref)
        prior = await self._read_prior_graph(graph)
        current = await self._get_member_versions(prior)
        added = await self._resolve_skill_versions(request.skill_refs)
        if set(current) & set(added):
            raise ValueError("Skill family already belongs to the graph")
        config = self._build_config(prior)
        new_sources = self._skills_from_versions(added)
        records = await self._get_or_extract_scene_records(added, new_sources, request.force_extract_scene, config)
        versions = {**current, **added}
        get_logger(__name__).info(
            "Graph add inputs resolved | prior_skills=%d added_skills=%d total_skills=%d",
            len(current),
            len(added),
            len(versions),
        )
        sources = {**self._skills_from_versions(current), **new_sources}
        store = self._graph_store(request, graph.graph_id, versions, graph)
        generated = await self._generate_graph(store, sources, records, prior, config)
        return await self._publish_graph(store, generated)

    async def remove_graph_members(self, request: RemoveGraphMembersRequest) -> OrganizeResult:
        """Remove a complete validated family batch, preserving Skill source and history.

        Args:
            request: Graph family and member references.

        Returns:
            Published remaining scope, including an empty graph.
        """
        graph = await self._resolve_graph(request.graph_ref)
        prior = await self._read_prior_graph(graph)
        current = await self._get_member_versions(prior)
        removed = await self._resolve_skill_versions(request.skill_refs)
        if not set(removed) <= set(current):
            raise ValueError("Skill family is not a graph member")
        versions = {key: value for key, value in current.items() if key not in removed}
        get_logger(__name__).info(
            "Graph remove inputs resolved | prior_skills=%d removed_skills=%d remaining_skills=%d",
            len(current),
            len(removed),
            len(versions),
        )
        sources = self._skills_from_versions(versions)
        records: dict[str, ProvidedSceneRecord | None] = {}
        config = self._build_config(prior)
        store = self._graph_store(request, graph.graph_id, versions, graph)
        generated = await self._generate_graph(store, sources, records, prior, config)
        return await self._publish_graph(store, generated)

    async def recommend(self, request: RecommendRequest) -> RecommendResult:
        """Rank exact versions from one pinned snapshot using graph or cosine scores.

        Args:
            request: Public instruction and strategy within the authorized graph.

        Returns:
            Selected versions and original score components in the same order.
        """
        graph = await self._resolve_graph(request.graph_id)
        snapshot = request.graph_version_id or graph.current_version_id
        if snapshot is None:
            raise ValueError("Graph has no published snapshot")
        nodes, edges, metadata = await self.graphs.load(graph.graph_id, snapshot)
        version_ids: dict[str, str] = {}
        for node in nodes:
            if node.node_type != GraphNodeType.SKILL:
                continue
            member_id = node.metadata[METADATA_COLLECTION_SKILL_ID]
            if not isinstance(member_id, str) or node.skill_version_id is None:
                raise ValueError("Skill node requires a member identity and a stored version")
            version_ids[member_id] = node.skill_version_id
        await self.persistence.skills.validate_version_visibility(list(version_ids.values()))
        store = SkillGraphStore(
            self.persistence.database,
            graph_id=graph.graph_id,
            actor_id=request.project_id,
            version_ids=version_ids,
        )
        config = restore_build_config(self._build_config(None), metadata)
        algorithm = self._algorithm(store, config=config)
        task = Task(id="recommend", instruction=request.task)
        scores: list[VectorRecommendationScore] | list[GraphRecommendationScore]
        if request.strategy == "vector":
            algorithm.validate_vector_metadata(metadata)
            algorithm.bind_vector_snapshot(
                snapshot,
                skill_ids=set(version_ids),
                skill_versions=version_ids,
            )
            vector_result = await algorithm.retrieve_vector(
                task=task, limit=request.top_k, include_scenes=request.include_scenes
            )
            scores = [
                VectorRecommendationScore(
                    skill_version_id=version_ids[score.member_id],
                    similarity=score.similarity,
                    node_id=score.node_id,
                    node_type=score.node_type,
                )
                for score in vector_result.scores[: request.top_k]
            ]
            score_type = vector_result.score_type
        else:
            await algorithm.load_snapshot(
                snapshot,
                nodes=nodes,
                edges=edges,
                metadata=metadata,
            )
            graph_result = await algorithm.retrieve(task=task, limit=request.top_k)
            scores = [
                GraphRecommendationScore(
                    skill_version_id=version_ids[score.member_id],
                    ppr_score=score.ppr_score,
                    scene_boost=score.scene_boost,
                    graph_score=score.graph_score,
                    rerank_score=score.rerank_score,
                )
                for score in graph_result.scores[: request.top_k]
            ]
            score_type = graph_result.score_type
        selected_versions = [await self.persistence.skills.get(score.skill_version_id) for score in scores]
        return RecommendResult(
            skills=selected_versions,
            graph_version_id=snapshot,
            strategy=request.strategy,
            score_type=score_type,
            scores=scores,
            include_scenes=request.include_scenes,
        )

    async def preview_graph(self, graph_ref: str) -> GraphTopology:
        """Read the published snapshot selected by an authorized reference.

        Args:
            graph_ref: Graph family, alias, or immutable snapshot reference.

        Returns:
            Graph identity, selected snapshot identity, and complete published topology.
        """
        selected = await self.graphs.resolve(graph_ref)
        snapshot_id = selected["graph_version_id"]
        if snapshot_id is None:
            raise ValueError("Graph has no published snapshot")
        return await self.get_graph_snapshot(selected["graph_id"], snapshot_id)

    async def list_graph(self) -> list[GraphListItem]:
        """List the project-owned graphs and their publication pointers.

        Returns:
            Graph records with string identities, timestamps, and publication pointers.
        """
        graphs = await self.graphs.list_graphs()
        return [cast(GraphListItem, graph.model_dump()) for graph in graphs]

    async def list_graph_snapshots(self, graph_id: str) -> list[GraphSnapshotListItem]:
        """List immutable snapshots and their creation metadata.

        Args:
            graph_id: Project-local graph identity or alias.

        Returns:
            Snapshot records with string identities, build metadata, and timestamps.
        """
        try:
            graph = await self._resolve_graph(graph_id)
        except GraphNotFoundError:
            return []
        snapshots = await self.graphs.snapshots(graph.graph_id)
        return [cast(GraphSnapshotListItem, snapshot.model_dump()) for snapshot in snapshots]

    async def get_graph_snapshot(self, graph_id: str, snapshot_id: str) -> GraphTopology:
        """Read the selected snapshot's Skill members and relations.

        Args:
            graph_id: Project-local graph identity or alias.
            snapshot_id: Exact prefixed immutable snapshot identity.

        Returns:
            Graph identity, snapshot identity, and published nodes and edges.
        """
        graph = await self._resolve_graph(graph_id)
        preview = await self.graphs.snapshot(graph.graph_id, snapshot_id)
        return {**preview, "graph_id": graph.graph_id}
