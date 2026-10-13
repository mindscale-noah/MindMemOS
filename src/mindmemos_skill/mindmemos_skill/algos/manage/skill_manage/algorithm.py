"""Native asynchronous Scene graph construction and retrieval."""

import hashlib
import json
from collections.abc import Iterator, Mapping
from itertools import islice
from math import hypot
from time import perf_counter
from uuid import UUID

from pydantic import JsonValue

from mindmemos_skill.common.ids import generate_id

from ....agents import Agent
from ....common.asyncio import map_bounded
from ....common.json import parse_model
from ....envs.typing import Task
from ....infra.database import Vector
from ....infra.embedding import EmbeddingModel
from ....logging import get_logger, traced
from ....persistence.db_services.graphs.typing import GraphNodeType, SkillGraphEdgeRecord, SkillGraphNodeRecord
from ....persistence.db_services.graphs.writer import SkillGraphStore, graph_digest
from ....scene.typing import ProvidedSceneRecord
from ....skill import Skill
from ...base import Algorithm
from ..base import SkillManager
from ..typing import GeneratedGraph
from .config import EmbeddingProfile, RetrievalStrategyConfig, SkillManageConfig
from .constant import (
    JOINT_LINKING_VERSION,
    METADATA_BASE_URL,
    METADATA_BUILD_COVERAGE,
    METADATA_BUILD_SETTINGS,
    METADATA_BUILD_WARNINGS,
    METADATA_CANDIDATE_PAIRS,
    METADATA_CHAT,
    METADATA_COLLECTION_SKILL_ID,
    METADATA_CONFIG,
    METADATA_CONTENT_HASH,
    METADATA_EDGES,
    METADATA_EMBEDDING,
    METADATA_EMBEDDING_PROFILE,
    METADATA_ENDPOINT_HASH,
    METADATA_ENRICHMENT,
    METADATA_ENRICHMENT_USAGE,
    METADATA_EXAMPLE_TASKS,
    METADATA_EXTRACTION_USAGE,
    METADATA_FAMILY_ID,
    METADATA_KEY,
    METADATA_LINKING_VERSION,
    METADATA_REPRESENTATION,
    METADATA_REUSED_VECTORS,
    METADATA_SCENES,
    METADATA_SKILLS,
    METADATA_SOURCE_HASH,
    METADATA_TOOLING,
    METADATA_VECTORS,
    SCENE_GRAPH_REPRESENTATION_VERSION,
)
from .enrichment import SkillEnricher, enrichment_identity
from .errors import SceneGraphError, SceneGraphFailure
from .linking import (
    RELATION_TYPE_WEIGHTS,
    joint_candidate_pairs,
    joint_dependency_edges,
    joint_model_candidates,
    joint_relation_payload,
)
from .parsing import parse_skill_document
from .prompts import JOINT_RELATION_SYSTEM, JOINT_RELATION_WEIGHT_SYSTEM, QUERY_SYSTEM
from .query import normalize_query, public_task_text, query_text
from .representation import (
    document_features,
    joint_embedding_fields,
    linking_query_text,
    merge_skill_semantics,
    scene_features,
    scene_flow_edges,
)
from .retrieval import rank_joint
from .seeding import balance_seeds, select_seeds
from .typing import (
    BuildSettings,
    DerivedEdgeMetadata,
    ExtractionSummary,
    InferredEdge,
    JointEdgeDraft,
    JointLinkResult,
    JointRelations,
    JointRepresentation,
    OrganizedGraph,
    QuerySchema,
    SceneGraphContent,
    SceneGraphRetrieval,
    SceneGraphVectorRetrieval,
    SceneGraphVectorScore,
)


def node_id(skill_id: str, scene_id: str | None = None) -> str:
    """Encode collision-free graph IDs without using display names.

    Args:
        skill_id: Caller-selected collection member ID.
        scene_id: Scene identity within that member, or None for the Skill node.
    """
    values = [GraphNodeType.SKILL, skill_id] if scene_id is None else [GraphNodeType.SCENE, skill_id, scene_id]
    return json.dumps(values, ensure_ascii=False, separators=(",", ":"))


def embedding_text(node: SkillGraphNodeRecord) -> str:
    """Render source-compatible tagged fields for a persisted node.

    Args:
        node: Skill or Scene record with parsed public retrieval features.
    """
    parts = [f"[{'Skill' if node.node_type == GraphNodeType.SKILL else 'Scene'}] {node.name}"]
    fields = (
        ("DESCRIPTION", node.description),
        ("CAPABILITY", node.one_line_capability),
        ("INPUTS", "\n".join(node.inputs)),
        ("OUTPUTS", "\n".join(node.outputs)),
        ("DOMAIN_TAGS", "\n".join(node.domain_tags)),
        ("TOOLING", "\n".join(node.tooling)),
        ("EXAMPLE_TASKS", "\n".join(node.example_tasks)),
        ("SNIPPET", node.rendered_snippet),
    )
    parts.extend(f"[{tag}] {value}" for tag, value in fields if value)
    if METADATA_FAMILY_ID in node.metadata:
        parts.extend(f"[{tag}] {value}" for tag, value in joint_embedding_fields(node) if value)
    return "\n".join(parts)


@Algorithm.register("skill_manage")
class SkillManageAlgorithm(SkillManager[SkillManageConfig]):
    """Build validated Scene records and rank a pinned explicit Skill scope."""

    config_type = SkillManageConfig
    """Configuration used by the unified algorithm registry."""

    def __init__(
        self,
        config: SkillManageConfig,
        *,
        store: SkillGraphStore | None = None,
        provided: Mapping[str, ProvidedSceneRecord | None] | None = None,
    ) -> None:
        """Retain explicit inputs without I/O.

        Args:
            config: Original construction and retrieval settings.
            store: Pipeline-owned project and version bindings.
            provided: Validated records by member; explicit None means confirmed insufficient source.
        """
        super().__init__(config.model_copy(deep=True))
        self.store = store
        self._provided = {
            key: ProvidedSceneRecord.model_validate(value.model_dump()) if value is not None else None
            for key, value in (provided or {}).items()
        }
        self.embedding_model = EmbeddingModel.from_config(self.config.embedding)
        self.chat_model = Agent.from_config(self.config.chat) if self.config.chat is not None else None
        self.extraction_summaries: dict[str, ExtractionSummary] = {}
        self.enrichment_summaries: dict[str, ExtractionSummary] = {}
        self.build_warnings: list[str] = []
        self._state: OrganizedGraph | None = None
        self._scene_digests: dict[str, str] = {}

    @property
    def snapshot_id(self) -> str:
        """Return the actual successfully bound immutable snapshot."""
        if self._state is None:
            raise RuntimeError("No graph snapshot is bound")
        return self._state.snapshot_id

    def reuse_snapshot(
        self,
        snapshot_id: str,
        *,
        nodes: list[SkillGraphNodeRecord],
        edges: list[SkillGraphEdgeRecord],
        metadata: dict[str, JsonValue],
    ) -> None:
        """Capture prior immutable nodes for member operations and vector reuse.

        Args:
            snapshot_id: Exact graph head captured by the Pipeline.
            nodes: Full validated nodes from that head.
            edges: Full validated relationships from that head.
            metadata: Immutable build settings from that head.
        """
        settings = metadata.get(METADATA_BUILD_SETTINGS, {})
        profile = EmbeddingProfile.model_validate(settings.get(METADATA_CONFIG, {}).get(METADATA_EMBEDDING_PROFILE, {}))
        content = SceneGraphContent(embedding=profile, nodes=nodes, edges=edges)
        if content.content_digest() != metadata.get(METADATA_CONTENT_HASH):
            raise ValueError("Captured graph logical content digest differs")
        members = frozenset(
            node.metadata[METADATA_COLLECTION_SKILL_ID] for node in nodes if node.node_type == GraphNodeType.SKILL
        )
        warnings = tuple(
            warning
            for warning in metadata.get(METADATA_BUILD_WARNINGS, [])
            if warning == "SCENE_GRAPH_ENRICHMENT_RULE_FALLBACK"
        )
        self._state = OrganizedGraph("prior", snapshot_id, nodes, edges, members, profile, warnings)
        self._scene_digests = dict(metadata.get(METADATA_BUILD_SETTINGS, {}).get(METADATA_SCENES, {}))

    async def organize(self, *, skills: Mapping[str, Skill]) -> GeneratedGraph:
        """Generate or read a reusable graph without publishing a version.

        Args:
            skills: Complete exact source snapshots keyed by canonical family identity.

        Returns:
            Complete graph data for a separate atomic publication step.
        """
        prior = self._state
        self._state = None
        if self.store is None:
            raise ValueError("Graph storage must be supplied by the Pipeline")
        captured = prior.skill_ids if prior is not None else frozenset()
        if set(self._provided) - set(skills) or (
            self.config.build_scene and not set(skills) <= (set(self._provided) | captured)
        ):
            raise SceneGraphError(SceneGraphFailure.INVALID_INPUT)
        self._scene_digests = (
            {key: value for key, value in self._scene_digests.items() if key in skills}
            if self.config.build_scene
            else {}
        )
        bindings = self.store.version_ids
        request_hash = self._source_hash(skills, bindings)
        snapshot_id = await self.store.find(request_hash)
        if snapshot_id is None:
            get_logger(__name__).info(
                "Graph snapshot cache missed | graph_id=%s skills=%d build_scene=%s enrich_skill=%s",
                self.store.scope["graph_id"],
                len(skills),
                self.config.build_scene,
                self.config.enrich_skill,
            )
            return await self._build_joint(skills, request_hash, prior)
        nodes, edges, metadata = await self.store.load(snapshot_id)
        get_logger(__name__).info(
            "Graph snapshot reused | graph_id=%s graph_version_id=%s skills=%d nodes=%d edges=%d",
            self.store.scope["graph_id"],
            snapshot_id,
            len(skills),
            len(nodes),
            len(edges),
        )
        return GeneratedGraph(
            snapshot_id=snapshot_id, request_hash=request_hash, nodes=nodes, edges=edges, metadata=metadata
        )

    async def recommend(self, *, task: Task, limit: int = 5) -> list[str]:
        """Return ranked member identities from the pinned scope.

        Args:
            task: Public task used for ranking.
            limit: Positive result cap.
        """
        return [score.member_id for score in (await self.retrieve(task=task, limit=limit)).scores]

    async def retrieve(self, *, task: Task, limit: int = 5) -> SceneGraphRetrieval:
        """Return original graph score components.

        Args:
            task: Complete public task.
            limit: Positive graph candidate cap before reranking.
        """
        return await self._retrieve(self._state, task, limit)

    def _build_settings(self) -> BuildSettings:
        """Identify construction inputs independently of query and transport options."""
        config = self.config.model_dump(
            mode="json", exclude={METADATA_EMBEDDING: {"api_key"}, METADATA_CHAT: {"api_key"}}
        )
        for key in (
            *RetrievalStrategyConfig.model_fields,
            "mode",
            "query_rewrite",
            "enrichment_concurrency",
            "skill_candidates",
            "scene_candidates",
            "skill_seeds",
            "scene_seeds",
            "alpha",
            "aggregation",
            "score_threshold",
            "damping",
            "max_iter",
            "tolerance",
        ):
            config.pop(key)
        config[METADATA_EMBEDDING_PROFILE] = self._embedding_profile().model_dump(mode="json")
        for connection in (config[METADATA_EMBEDDING], config[METADATA_CHAT]):
            if connection is not None:
                connection[METADATA_ENDPOINT_HASH] = graph_digest(connection.pop(METADATA_BASE_URL))
                for key in (
                    "batch_size",
                    "timeout_s",
                    "max_retries",
                    "sdk_max_retries",
                    "retry_initial_wait_s",
                    "retry_max_wait_s",
                    "retry_policy",
                    "retry_wait",
                    "retry_waits_s",
                ):
                    connection.pop(key, None)
        if not (
            self.config.build_scene
            or self.config.llm_linking
            or self.config.scene_llm_linking
            or self.config.enrich_skill
        ):
            config.pop(METADATA_CHAT)
        return {
            METADATA_REPRESENTATION: SCENE_GRAPH_REPRESENTATION_VERSION,
            METADATA_CONFIG: config,
            METADATA_LINKING_VERSION: JOINT_LINKING_VERSION,
            METADATA_ENRICHMENT: enrichment_identity() if self.config.enrich_skill else None,
            METADATA_SCENES: {
                **self._scene_digests,
                **{
                    key: graph_digest(value.model_dump(mode="json") if value is not None else None)
                    for key, value in sorted(self._provided.items())
                },
            }
            if self.config.build_scene
            else {},
        }

    def _source_hash(self, skills: Mapping[str, Skill], bindings: Mapping[str, str]) -> str:
        """Hash exact selected sources and construction settings without query settings."""
        return graph_digest(
            {
                METADATA_SKILLS: [[key, skills[key].snapshot().model_dump(mode="json")] for key in sorted(skills)],
                "bindings": bindings,
                "settings": self._build_settings(),
            }
        )

    async def load_snapshot(
        self,
        snapshot_id: str,
        *,
        nodes: list[SkillGraphNodeRecord] | None = None,
        edges: list[SkillGraphEdgeRecord] | None = None,
        metadata: dict[str, JsonValue] | None = None,
    ) -> None:
        """Pin compatible committed content without rebuilding or requiring the original SSL.

        Args:
            snapshot_id: Immutable snapshot in this manager's project and graph.
            nodes: Already loaded node records, avoiding a second full snapshot read.
            edges: Already loaded edge records matching the supplied nodes.
            metadata: Already loaded snapshot metadata matching the supplied nodes.

        Raises:
            SceneGraphError: The representation or full embedding profile is incompatible.
            ValueError: Content integrity or exact member/version bindings differ.
        """
        self._state = None
        bindings = self.store.version_ids
        if nodes is None or edges is None or metadata is None:
            nodes, edges, metadata = await self.store.load(snapshot_id)
        settings = metadata.get(METADATA_BUILD_SETTINGS, {})
        expected_profile = self._embedding_profile()
        profile = EmbeddingProfile.model_validate(settings.get(METADATA_CONFIG, {}).get(METADATA_EMBEDDING_PROFILE, {}))
        if (
            settings.get(METADATA_REPRESENTATION) != SCENE_GRAPH_REPRESENTATION_VERSION
            or profile.node_template not in ("scene_graph_nodes_v2", "scene_graph_nodes_v3")
            or profile.model_dump(exclude={"node_template"}) != expected_profile.model_dump(exclude={"node_template"})
        ):
            raise SceneGraphError(SceneGraphFailure.PROFILE_MISMATCH)
        content = SceneGraphContent(embedding=profile, nodes=nodes, edges=edges)
        if content.content_digest() != metadata.get(METADATA_CONTENT_HASH):
            raise ValueError("Joint graph logical content digest differs")
        actual = {
            node.metadata[METADATA_COLLECTION_SKILL_ID]: node.skill_version_id
            for node in nodes
            if node.node_type == GraphNodeType.SKILL
        }
        if actual != bindings:
            raise ValueError("Persisted graph versions differ from the candidate scope")
        self.build_warnings = [
            warning
            for warning in metadata.get(METADATA_BUILD_WARNINGS, [])
            if warning == "SCENE_GRAPH_ENRICHMENT_RULE_FALLBACK"
        ]
        self._state = OrganizedGraph(
            metadata[METADATA_SOURCE_HASH],
            snapshot_id,
            nodes,
            edges,
            frozenset(bindings),
            profile,
            tuple(self.build_warnings),
        )

    def _embedding_profile(self) -> EmbeddingProfile:
        """Resolve deterministic joint text and vector provenance without model I/O."""
        profile = self.config.embedding_profile or EmbeddingProfile(
            space_id=graph_digest(self.config.embedding.base_url),
            model=self.config.embedding.model,
            dimensions=self.config.embedding.dimensions,
            query_template="legacy_query_v1",
        )
        return profile.model_copy(update={"node_template": "scene_graph_nodes_v3"})

    @traced("skill_manage.build")
    async def _build_joint(
        self, skills: Mapping[str, Skill], request_hash: str, prior: OrganizedGraph | None
    ) -> GeneratedGraph:
        """Generate joint semantics, vectors, and links without publishing them."""
        snapshot_id = generate_id("skgv-")
        representation = self._build_joint_representation(snapshot_id, skills, prior)
        nodes = await self._enrich_joint_nodes(representation.nodes, skills, representation.reused_members, prior)
        nodes, reused_vectors = await self._embed_joint_nodes(nodes, prior)
        links = await self._build_joint_links(
            snapshot_id, nodes, representation.structural_edges, member_count=len(skills)
        )
        return self._finalize_joint_graph(
            snapshot_id=snapshot_id,
            request_hash=request_hash,
            nodes=nodes,
            links=links,
            member_count=len(skills),
            reused_vectors=reused_vectors,
        )

    @traced("skill_manage.build.representation")
    def _build_joint_representation(
        self, snapshot_id: str, skills: Mapping[str, Skill], prior: OrganizedGraph | None
    ) -> JointRepresentation:
        """Build or reuse each member's nodes and internal structural edges."""
        logger = get_logger(__name__)
        started = perf_counter()
        logger.info(
            "Graph representation stage started | graph_id=%s skills=%d build_scene=%s",
            self.store.scope["graph_id"],
            len(skills),
            self.config.build_scene,
        )
        nodes: list[SkillGraphNodeRecord] = []
        edge_data: list[JointEdgeDraft] = []
        reused_members = set()
        prior_versions = (
            {
                node.metadata[METADATA_COLLECTION_SKILL_ID]: node.skill_version_id
                for node in prior.nodes
                if node.node_type == GraphNodeType.SKILL
            }
            if prior is not None
            else {}
        )
        for key in sorted(skills):
            if (
                prior is not None
                and key not in self._provided
                and prior_versions.get(key) == self.store.version_ids[key]
            ):
                member_nodes, member_edges = self._reuse_member_representation(snapshot_id, key, prior)
                reused_members.add(key)
            else:
                member_nodes, member_edges = self._build_member_representation(snapshot_id, key, skills[key])
            nodes.extend(member_nodes)
            edge_data.extend(member_edges)

        logger.info(
            "Graph representation stage completed | graph_id=%s skills=%d scenes=%d nodes=%d structural_edges=%d duration_s=%.2f",
            self.store.scope["graph_id"],
            len(skills),
            sum(node.node_type == GraphNodeType.SCENE for node in nodes),
            len(nodes),
            len(edge_data),
            perf_counter() - started,
        )
        return JointRepresentation(nodes=nodes, structural_edges=edge_data, reused_members=reused_members)

    @traced("skill_manage.build.representation.reuse_member")
    def _reuse_member_representation(
        self, snapshot_id: str, member_id: str, prior: OrganizedGraph
    ) -> tuple[list[SkillGraphNodeRecord], list[JointEdgeDraft]]:
        """Copy captured member nodes and structural edges into the new snapshot."""
        captured = [node for node in prior.nodes if node.metadata[METADATA_COLLECTION_SKILL_ID] == member_id]
        if (
            self.config.build_scene
            and not any(node.node_type == GraphNodeType.SCENE for node in captured)
            and self._scene_digests.get(member_id) != graph_digest(None)
        ):
            raise ValueError("Captured Scene-enabled member has no Scene nodes")
        nodes = [node.model_copy(update={"graph_version_id": snapshot_id}) for node in captured]
        ids = {node.node_id for node in captured}
        edges: list[JointEdgeDraft] = [
            {
                "source": edge.source_node_id,
                "target": edge.target_node_id,
                "type": edge.edge_type,
                "weight": self.config.scene_flow_weight if edge.edge_type == "flow" else edge.weight,
                **edge.metadata,
            }
            for edge in prior.edges
            if edge.edge_type in ("flow", "containment") and edge.source_node_id in ids and edge.target_node_id in ids
        ]
        return nodes, edges

    @traced("skill_manage.build.representation.build_member")
    def _build_member_representation(
        self, snapshot_id: str, member_id: str, skill: Skill
    ) -> tuple[list[SkillGraphNodeRecord], list[JointEdgeDraft]]:
        """Parse one Skill and attach its optional Scene nodes and structural edges."""
        parsed = parse_skill_document(skill.content, skill_id=member_id, max_chars=self.config.max_input_chars)
        values = {
            name: parsed[name]
            for name in (
                "name",
                "description",
                "one_line_capability",
                "inputs",
                "outputs",
                "domain_tags",
                METADATA_TOOLING,
                METADATA_EXAMPLE_TASKS,
                "rendered_snippet",
            )
        }
        derived, values[METADATA_TOOLING] = document_features(skill, parsed, member_id=member_id, family_id=member_id)
        owner = self.store.bind_node(
            snapshot_id,
            member_id,
            {
                "node_id": node_id(member_id),
                **values,
                METADATA_KEY: derived.model_dump(mode="json"),
            },
        )
        if not self.config.build_scene:
            return [owner], []
        record = self._provided[member_id]
        if record is None:
            return [owner], []
        nodes = [merge_skill_semantics(owner, record)]
        edge_data: list[JointEdgeDraft] = []
        scene_node_ids = {scene.scene_id: node_id(member_id, scene.scene_id) for scene in record.scenes}
        for scene in record.scenes:
            scene_key = scene_node_ids[scene.scene_id]
            nodes.append(
                self.store.bind_node(
                    snapshot_id,
                    member_id,
                    {
                        "node_id": scene_key,
                        "node_type": GraphNodeType.SCENE,
                        "owner_node_id": owner.node_id,
                        **scene_features(scene, record, member_id=member_id, family_id=member_id),
                    },
                )
            )
            edge_data.append({"source": scene_key, "target": owner.node_id, "type": "containment", "weight": 1.0})
            edge_data.extend(scene_flow_edges(scene, scene_node_ids, weight=self.config.scene_flow_weight))
        return nodes, edge_data

    @traced("skill_manage.build.enrichment")
    async def _enrich_joint_nodes(
        self,
        nodes: list[SkillGraphNodeRecord],
        skills: Mapping[str, Skill],
        reused_members: set[str],
        prior: OrganizedGraph | None,
    ) -> list[SkillGraphNodeRecord]:
        """Enrich new Skill nodes concurrently while retaining order and reuse warnings."""
        if reused_members and prior is not None:
            self.build_warnings.extend(prior.warnings)
        if not self.config.enrich_skill:
            return nodes
        logger = get_logger(__name__)
        enrichment_started = perf_counter()
        logger.info("Graph enrichment stage started | graph_id=%s skills=%d", self.store.scope["graph_id"], len(skills))

        async def enrich_node(node: SkillGraphNodeRecord) -> SkillGraphNodeRecord:
            """Enrich new Skill nodes while preserving Scenes and reused members."""
            if (
                node.node_type == GraphNodeType.SKILL
                and node.metadata[METADATA_COLLECTION_SKILL_ID] not in reused_members
            ):
                return await self._enrich(node, skills)
            return node

        nodes = await map_bounded(enrich_node, nodes, concurrency=self.config.enrichment_concurrency)
        logger.info(
            "Graph enrichment stage completed | graph_id=%s skills=%d warnings=%d duration_s=%.2f",
            self.store.scope["graph_id"],
            len(skills),
            len(self.build_warnings),
            perf_counter() - enrichment_started,
        )
        return nodes

    @traced("skill_manage.build.linking")
    async def _build_joint_links(
        self,
        snapshot_id: str,
        nodes: list[SkillGraphNodeRecord],
        structural_edges: list[JointEdgeDraft],
        *,
        member_count: int,
    ) -> JointLinkResult:
        """Combine structural, deterministic dependency, and inferred relation edges."""
        logger = get_logger(__name__)
        edge_data = list(structural_edges)
        linking_started = perf_counter()
        logger.info(
            "Graph linking stage started | graph_id=%s nodes=%d skill_llm_linking=%s scene_llm_linking=%s",
            self.store.scope["graph_id"],
            len(nodes),
            self.config.llm_linking,
            self.config.scene_llm_linking,
        )
        dependencies = list(
            joint_dependency_edges(
                nodes, dependency_match_threshold=self.config.joint_linking.dependency_match_threshold
            )
        )
        edge_data.extend(dependencies)
        dependency_pairs = {frozenset((edge["source"], edge["target"])) for edge in dependencies}
        model_linking = self.config.llm_linking or self.config.scene_llm_linking
        query_vectors = []
        if model_linking and self.config.joint_linking.dense_candidates and member_count > 1:
            query_vectors = await self._embed_linking_queries(nodes)
        if model_linking and member_count > 1:
            candidates = joint_model_candidates(
                nodes,
                query_vectors,
                self.config.joint_linking,
                skill_linking=self.config.llm_linking,
                scene_linking=self.config.scene_llm_linking,
                dependency_pairs=dependency_pairs,
            )
        else:
            candidates = {}
        logger.info(
            "Graph linking candidates selected | graph_id=%s dependency_edges=%d candidate_nodes=%d directed_candidates=%d batch_size=%d",
            self.store.scope["graph_id"],
            len(dependencies),
            len(candidates),
            sum(len(values) for values in candidates.values()),
            self.config.llm_batch_size,
        )
        if model_linking:
            edge_data.extend((await self._infer_joint_relations(joint_candidate_pairs(nodes, candidates))))

        edges = self._make_joint_edge_records(snapshot_id, edge_data)
        logger.info(
            "Graph linking stage completed | graph_id=%s nodes=%d edges=%d duration_s=%.2f",
            self.store.scope["graph_id"],
            len(nodes),
            len(edges),
            perf_counter() - linking_started,
        )
        return JointLinkResult(
            edges=edges,
            candidate_pairs=sum(1 for _ in joint_candidate_pairs(nodes, candidates)),
        )

    @traced("skill_manage.build.linking.query_embedding")
    async def _embed_linking_queries(self, nodes: list[SkillGraphNodeRecord]) -> list[list[float]]:
        """Reuse matching node vectors and embed only distinct missing query texts."""
        by_text = {embedding_text(node): node.embedding for node in nodes}
        query_texts = [linking_query_text(node) for node in nodes]
        missing = list(dict.fromkeys(text for text in query_texts if text not in by_text))
        by_text.update(zip(missing, (await self._embed(missing)), strict=True))
        return [by_text[text] for text in query_texts]

    @traced("skill_manage.build.linking.edge_records")
    def _make_joint_edge_records(self, snapshot_id: str, edge_data: list[JointEdgeDraft]) -> list[SkillGraphEdgeRecord]:
        """Bind deduplicated edges to the snapshot, retaining the first evidence."""
        edges = {}
        for edge in edge_data:
            key = (edge["source"], edge["target"], edge["type"])
            if key in edges:
                continue  # Deterministic evidence takes precedence over optional inference.
            edges[key] = SkillGraphEdgeRecord(
                **self.store.scope,
                graph_version_id=snapshot_id,
                edge_id=UUID(generate_id()),
                source_node_id=edge["source"],
                target_node_id=edge["target"],
                edge_type=edge["type"],
                weight=edge["weight"],
                metadata=DerivedEdgeMetadata(
                    evidence=edge.get("description", "")[:1000],
                    condition=edge.get("condition", ""),
                    confidence=edge.get("confidence"),
                ).model_dump(mode="json"),
            )
        return list(edges.values())

    @traced("skill_manage.build.finalize")
    def _finalize_joint_graph(
        self,
        *,
        snapshot_id: str,
        request_hash: str,
        nodes: list[SkillGraphNodeRecord],
        links: JointLinkResult,
        member_count: int,
        reused_vectors: int,
    ) -> GeneratedGraph:
        """Validate graph content and collect provenance, usage, and coverage metadata."""
        content = SceneGraphContent(embedding=self._embedding_profile(), nodes=nodes, edges=links.edges)
        return GeneratedGraph(
            snapshot_id=snapshot_id,
            request_hash=request_hash,
            nodes=nodes,
            edges=links.edges,
            metadata={
                METADATA_BUILD_SETTINGS: self._build_settings(),
                METADATA_SOURCE_HASH: request_hash,
                METADATA_CONTENT_HASH: content.content_digest(),
                METADATA_EXTRACTION_USAGE: {
                    key: value.model_dump(mode="json") for key, value in self.extraction_summaries.items()
                },
                **(
                    {
                        METADATA_ENRICHMENT_USAGE: {
                            key: value.model_dump(mode="json") for key, value in self.enrichment_summaries.items()
                        },
                        METADATA_BUILD_WARNINGS: list(self.build_warnings),
                    }
                    if self.config.enrich_skill
                    else {}
                ),
                METADATA_BUILD_COVERAGE: {
                    METADATA_SKILLS: member_count,
                    METADATA_SCENES: len(nodes) - member_count,
                    METADATA_EDGES: len(links.edges),
                    METADATA_VECTORS: len(nodes),
                    METADATA_REUSED_VECTORS: reused_vectors,
                    METADATA_CANDIDATE_PAIRS: links.candidate_pairs,
                },
            },
        )

    @traced("skill_manage.build.enrichment.member")
    async def _enrich(self, node: SkillGraphNodeRecord, skills: Mapping[str, Skill]) -> SkillGraphNodeRecord:
        """Fill only missing fields, retaining safe summaries and explicit fallback warnings."""
        key = node.metadata[METADATA_COLLECTION_SKILL_ID]
        enricher = SkillEnricher(self.chat_model)
        try:
            return await enricher.enrich(node, skills[key].content)
        except SceneGraphError:
            if self.config.enrichment_failure == "error":
                raise
            self.build_warnings.append("SCENE_GRAPH_ENRICHMENT_RULE_FALLBACK")
            return node
        finally:
            self.enrichment_summaries[key] = enricher.summary

    @traced("skill_manage.build.embedding")
    async def _embed_joint_nodes(
        self, nodes: list[SkillGraphNodeRecord], prior: OrganizedGraph | None
    ) -> tuple[list[SkillGraphNodeRecord], int]:
        """Reuse only identical rendered text in the same manager's fixed vector profile."""
        compatible = prior is not None and prior.embedding_profile == self._embedding_profile()
        vectors = {embedding_text(node): node.embedding for node in prior.nodes} if compatible else {}
        texts = [embedding_text(node) for node in nodes]
        reused = sum(text in vectors for text in texts)
        missing = list(dict.fromkeys(text for text in texts if text not in vectors))
        logger = get_logger(__name__)
        started = perf_counter()
        logger.info(
            "Graph embedding stage started | graph_id=%s nodes=%d reused_nodes=%d new_texts=%d model=%s dimensions=%d",
            self.store.scope["graph_id"],
            len(nodes),
            reused,
            len(missing),
            self.config.embedding.model,
            self.config.embedding.dimensions,
        )
        vectors.update(zip(missing, (await self._embed(missing)), strict=True))
        logger.info(
            "Graph embedding stage completed | graph_id=%s nodes=%d reused_nodes=%d embedded_texts=%d duration_s=%.2f",
            self.store.scope["graph_id"],
            len(nodes),
            reused,
            len(missing),
            perf_counter() - started,
        )
        return [
            type(node).model_validate(
                {
                    **node.model_dump(),
                    METADATA_EMBEDDING: vectors[text],
                    "embedding_input_hash": hashlib.sha256(text.encode()).hexdigest(),
                }
            )
            for node, text in zip(nodes, texts, strict=True)
        ], reused

    @traced("skill_manage.build.linking.inference")
    async def _infer_joint_relations(
        self, pairs: Iterator[tuple[SkillGraphNodeRecord, SkillGraphNodeRecord]]
    ) -> list[InferredEdge]:
        """Infer all batches concurrently and merge validated proposals in input order."""
        weighted = self.config.joint_linking.relation_weighting == "model_confidence"
        relation_validation = self.config.joint_linking.relation_validation
        system = JOINT_RELATION_WEIGHT_SYSTEM if weighted else JOINT_RELATION_SYSTEM
        completed_pairs = 0
        batches = []
        while batch := list(islice(pairs, self.config.llm_batch_size)):
            batches.append(batch)
        if not batches:
            return []

        async def infer_batch(
            item: tuple[int, list[tuple[SkillGraphNodeRecord, SkillGraphNodeRecord]]],
        ) -> list[InferredEdge]:
            """Infer and validate one independent batch before reporting completion."""
            nonlocal completed_pairs
            batch_index, batch = item
            edges: list[InferredEdge] = []
            started = perf_counter()
            get_logger(__name__).info(
                "Graph relation batch started | graph_id=%s batch=%d pairs=%d completed_pairs=%d",
                self.store.scope["graph_id"],
                batch_index,
                len(batch),
                completed_pairs,
            )
            payload = [[joint_relation_payload(node) for node in pair] for pair in batch]
            try:
                relations = parse_model((await self._chat(system, {"pairs": payload})), JointRelations)
                allowed = {frozenset((left.node_id, right.node_id)) for left, right in batch}
                for relation in relations.relations:
                    if (
                        relation.source_id == relation.target_id
                        or frozenset((relation.source_id, relation.target_id)) not in allowed
                    ):
                        if relation_validation == "strict":
                            raise ValueError("Relation endpoints must belong to a supplied pair")
                        continue
                    if weighted and relation.weight is None:
                        raise ValueError("Model-confidence weighting requires an explicit weight")
                    weight = relation.weight if weighted else RELATION_TYPE_WEIGHTS[relation.edge_type]
                    edge = {
                        "source": relation.source_id,
                        "target": relation.target_id,
                        "type": relation.edge_type,
                        "weight": weight * relation.confidence,
                        "confidence": relation.confidence,
                        "description": relation.evidence,
                    }
                    edges.append(edge)
            except Exception as exc:
                raise SceneGraphError(SceneGraphFailure.MODEL_FAILED) from exc
            completed_pairs += len(batch)
            get_logger(__name__).info(
                "Graph relation batch completed | graph_id=%s batch=%d pairs=%d completed_pairs=%d "
                "validated_edges=%d duration_s=%.2f",
                self.store.scope["graph_id"],
                batch_index,
                len(batch),
                completed_pairs,
                len(edges),
                perf_counter() - started,
            )
            return edges

        batch_results = await map_bounded(infer_batch, enumerate(batches, start=1), concurrency=len(batches))
        result: dict[tuple[str, str, str], InferredEdge] = {}
        for edges in batch_results:
            for edge in edges:
                key = (edge["source"], edge["target"], edge["type"])
                previous = result.get(key)
                if previous is None or (edge["confidence"], edge["weight"]) > (
                    previous["confidence"],
                    previous["weight"],
                ):
                    result[key] = edge
        return list(result.values())

    async def _chat(self, system: str, payload: dict[str, JsonValue]) -> str:
        """Require structured JSON from the configured chat dependency."""
        if self.chat_model is None:
            raise RuntimeError("Chat model is not configured")
        result = await self.chat_model.run(
            system=system, user=json.dumps(payload, ensure_ascii=False), response_format={"type": "json_object"}
        )
        if not isinstance(result.output, str):
            raise TypeError("Graph chat model must return JSON text")
        return result.output

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        """Validate injected models as strictly as the built-in transport."""
        if not texts:
            return []
        vectors = await self.embedding_model.embed(texts)
        if len(vectors) != len(texts):
            raise ValueError("Embedding count differs from input count")
        validator = Vector(self.config.embedding.dimensions)
        validated = [validator.validate(vector) for vector in vectors]
        if self._embedding_profile().normalization == "l2":
            return [validator.validate([value / hypot(*vector) for value in vector]) for vector in validated]
        return validated

    async def _retrieve(self, state: OrganizedGraph | None, task: Task, limit: int) -> SceneGraphRetrieval:
        """Execute retrieval entirely against the state captured at the public entry."""
        if state is None:
            raise RuntimeError("Organize a valid scope before recommending")
        if not state.skill_ids:
            return SceneGraphRetrieval(snapshot_id=state.snapshot_id, warnings=state.warnings)
        inferred = None
        if self.config.query_rewrite:
            try:
                inferred = parse_model((await self._chat(QUERY_SYSTEM, task.model_dump(mode="json"))), QuerySchema)
            except Exception:
                raise SceneGraphError(SceneGraphFailure.MODEL_FAILED) from None
        schema = normalize_query(public_task_text(task), inferred)
        vector = (await self._embed([query_text(schema, state.embedding_profile.query_template)]))[0]
        groups = []
        budgets = [(GraphNodeType.SKILL, self.config.skill_candidates, self.config.skill_seeds)]
        if self.config.mode in ("unified_ppr", "separate") and self.config.scene_seeds:
            budgets.append((GraphNodeType.SCENE, self.config.scene_candidates, self.config.scene_seeds))
        for node_type, candidates, seed_count in budgets:
            nodes = [node for node in state.nodes if node.node_type == node_type]
            if not nodes:
                continue
            hits = await self.store.search(state.snapshot_id, vector, top_k=candidates, node_type=node_type)
            groups.append(
                select_seeds(
                    nodes,
                    hits,
                    schema,
                    candidates=candidates,
                    seeds=seed_count,
                    threshold=self.config.score_threshold,
                )
            )
        balanced = balance_seeds(*groups)
        scores = rank_joint(state.nodes, state.edges, balanced, schema, config=self.config, limit=limit)
        if any(score.member_id not in state.skill_ids for score in scores):
            raise RuntimeError("Recommendation escaped the organized scope")
        return SceneGraphRetrieval(snapshot_id=state.snapshot_id, scores=tuple(scores), warnings=state.warnings)

    def build_warning_values(self, metadata: dict[str, JsonValue]) -> tuple[str, ...]:
        """Return only allowlisted warnings recognized by retrieval responses.

        Args:
            metadata: Immutable snapshot build metadata.
        """
        return tuple(
            value
            for value in metadata.get(METADATA_BUILD_WARNINGS, [])
            if value == "SCENE_GRAPH_ENRICHMENT_RULE_FALLBACK"
        )

    def validate_vector_metadata(self, metadata: dict[str, JsonValue]) -> EmbeddingProfile:
        """Validate the narrow metadata needed to interpret pure vector hits.

        Args:
            metadata: Immutable snapshot build metadata.
        """
        settings = metadata.get(METADATA_BUILD_SETTINGS, {})
        expected = self._embedding_profile()
        profile = EmbeddingProfile.model_validate(settings.get(METADATA_CONFIG, {}).get(METADATA_EMBEDDING_PROFILE, {}))
        if (
            settings.get(METADATA_REPRESENTATION) != SCENE_GRAPH_REPRESENTATION_VERSION
            or profile.node_template not in ("scene_graph_nodes_v2", "scene_graph_nodes_v3")
            or profile.model_dump(exclude={"node_template"}) != expected.model_dump(exclude={"node_template"})
        ):
            raise SceneGraphError(SceneGraphFailure.PROFILE_MISMATCH)
        return profile

    def bind_vector_snapshot(
        self,
        snapshot_id: str,
        *,
        skill_ids: frozenset[str] | set[str],
        skill_versions: Mapping[str, str] | None = None,
        warnings: tuple[str, ...] = (),
    ) -> None:
        """Pin only the owner/version identities required by pure vector ranking.

        Args:
            snapshot_id: Immutable snapshot selected before the query.
            skill_ids: Exact collection members authorized by the selected snapshot.
            skill_versions: Member IDs mapped to the versions captured by the Pipeline.
            warnings: Allowlisted build warnings already read from snapshot metadata.
        """
        self._state = OrganizedGraph(
            "vector-snapshot",
            snapshot_id,
            [],
            [],
            frozenset(skill_ids),
            self._embedding_profile(),
            tuple(warnings),
            dict(skill_versions or {}),
        )

    async def retrieve_vector(
        self, *, task: Task, limit: int = 5, include_scenes: bool = False
    ) -> SceneGraphVectorRetrieval:
        """Rank raw instruction vectors independently of the graph retrieval recipe.

        Args:
            task: Public task; only its stripped instruction is embedded, without rewriting.
            limit: Positive maximum number of unique Skill owners to return.
            include_scenes: Include Scene vectors and use the best cosine similarity per owner,
                searching at most limit * 100 nodes in at most five database searches.

        Returns:
            Pinned owner hits with actual cosine similarities and winning node identities.
            Ties use stable node IDs. No lexical fusion, threshold, PPR or field rerank applies.
            Fewer than limit owners may be returned when the node search budget is exhausted.

        Raises:
            RuntimeError: No successfully loaded or organized scope is available.
        """
        query = task.instruction.strip()
        state = self._state
        if state is None:
            raise RuntimeError("Organize a valid scope before recommending")
        if not state.skill_ids:
            return SceneGraphVectorRetrieval(
                snapshot_id=state.snapshot_id, include_scenes=include_scenes, warnings=state.warnings
            )
        vector = (await self._embed([query]))[0]
        if not include_scenes:
            hits = await self.store.search(
                state.snapshot_id,
                vector,
                top_k=limit,
                node_type=GraphNodeType.SKILL,
            )
        else:
            page_size = max(32, limit * 20)
            max_search_nodes = limit * 100
            offset = 0
            hits = []
            seen = set()
            while offset < max_search_nodes:
                search_count = min(page_size, max_search_nodes - offset)
                page = await self.store.search(state.snapshot_id, vector, top_k=search_count, offset=offset)
                offset += len(page)
                for hit in page:
                    member_id = hit.record[METADATA_KEY][METADATA_COLLECTION_SKILL_ID]
                    if member_id not in state.skill_ids:
                        raise RuntimeError("Recommendation escaped the organized scope")
                    if member_id in seen:
                        continue
                    seen.add(member_id)
                    hits.append(hit)
                    if len(hits) >= limit:
                        break
                # Search orders by distance then node ID; the first hit is each owner's best match.
                if len(hits) >= limit or len(page) < search_count:
                    break
        scores = []
        for hit in hits:
            metadata = hit.record[METADATA_KEY]
            member_id = metadata[METADATA_COLLECTION_SKILL_ID]
            if member_id not in state.skill_ids:
                raise RuntimeError("Recommendation escaped the organized scope")
            scores.append(
                SceneGraphVectorScore(
                    member_id=member_id,
                    node_id=hit.record["node_id"],
                    node_type=hit.record["node_type"],
                    skill_version_id=state.skill_versions.get(member_id)
                    if state.skill_versions
                    else hit.record["skill_version_id"],
                    metadata=metadata,
                    similarity=1 - hit.distance,
                )
            )
        return SceneGraphVectorRetrieval(
            snapshot_id=state.snapshot_id, scores=tuple(scores), include_scenes=include_scenes, warnings=state.warnings
        )
