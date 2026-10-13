"""Native asynchronous Scene graph construction and retrieval."""

import hashlib
import json
from collections.abc import Mapping
from itertools import islice
from math import hypot
from uuid import UUID, uuid4

from ....agents import Agent
from ....common.json import parse_model
from ....envs.typing import Task
from ....infra.database import Vector
from ....infra.embedding import EmbeddingModel
from ....logging import get_logger
from ....persistence.db_services.skill_management.typing import SkillGraphEdgeRecord, SkillGraphNodeRecord
from ....persistence.db_services.skill_management.writer import SkillGraphStore, graph_digest
from ....scene.typing import ProvidedSceneRecord
from ....skill import Skill
from ...base import Algorithm
from ..base import SkillManager
from .config import EmbeddingProfile, RetrievalStrategyConfig, SkillManageConfig
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
)
from .retrieval import rank_joint
from .seeding import balance_seeds, select_seeds
from .typing import (
    DerivedEdgeMetadata,
    ExtractionSummary,
    JointRelations,
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
    values = ["skill", skill_id] if scene_id is None else ["scene", skill_id, scene_id]
    return json.dumps(values, ensure_ascii=False, separators=(",", ":"))


def embedding_text(node) -> str:
    """Render source-compatible tagged fields for a persisted node.

    Args:
        node: Skill or Scene record with parsed public retrieval features.
    """
    parts = [f"[{'Skill' if node.node_type == 'skill' else 'Scene'}] {node.name}"]
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
    if "family_id" in node.metadata:
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
        provided: Mapping[str, ProvidedSceneRecord] | None = None,
    ):
        """Retain explicit inputs without I/O.

        Args:
            config: Original construction and retrieval settings.
            store: Pipeline-owned project and version bindings.
            provided: Validated Scene records captured by the Pipeline.
        """
        super().__init__(config.model_copy(deep=True))
        self.store = store
        self._provided = {
            key: ProvidedSceneRecord.model_validate(value.model_dump()) for key, value in (provided or {}).items()
        }
        self.embedding_model = EmbeddingModel.from_config(self.config.embedding)
        self.chat_model = Agent.from_config(self.config.chat) if self.config.chat is not None else None
        self.extraction_summaries: dict[str, ExtractionSummary] = {}
        self.enrichment_summaries: dict[str, ExtractionSummary] = {}
        self.build_warnings: list[str] = []
        self._state: OrganizedGraph | None = None
        self._scene_digests: dict[str, str] = {}
        self._enrichment_inputs: dict[str, SkillGraphNodeRecord] = {}

    @property
    def snapshot_id(self) -> UUID:
        """Return the actual successfully bound immutable snapshot."""
        if self._state is None:
            raise RuntimeError("No graph snapshot is bound")
        return self._state.snapshot_id

    def reuse_snapshot(self, snapshot_id: UUID, *, nodes, edges, metadata) -> None:
        """Capture prior immutable nodes for member operations and vector reuse.

        Args:
            snapshot_id: Exact graph head captured by the Pipeline.
            nodes: Full validated nodes from that head.
            edges: Full validated relationships from that head.
            metadata: Immutable build settings from that head.
        """
        settings = metadata.get("build_settings", {})
        profile = EmbeddingProfile.model_validate(settings.get("config", {}).get("embedding_profile", {}))
        content = SceneGraphContent(embedding=profile, nodes=nodes, edges=edges)
        if content.content_digest() != metadata.get("content_hash"):
            raise ValueError("Captured graph logical content digest differs")
        members = frozenset(node.metadata["collection_skill_id"] for node in nodes if node.node_type == "skill")
        self._state = OrganizedGraph("prior", snapshot_id, nodes, edges, members, profile)
        self._scene_digests = dict(metadata.get("build_settings", {}).get("scenes", {}))
        self._enrichment_inputs = {
            key: SkillGraphNodeRecord.model_validate(value)
            for key, value in metadata.get("enrichment_inputs", {}).items()
        }

    async def organize(self, *, skills: Mapping[str, Skill]) -> None:
        """Build or reuse the complete explicit scope.

        Args:
            skills: Exact source snapshots keyed by canonical family identity.
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
        bindings = await self.store.validate_skills(skills)
        request_hash = self._source_hash(skills, bindings)
        snapshot_id = await self.store.find(request_hash)
        if snapshot_id is None:
            snapshot_id = await self._build_joint(skills, request_hash, prior)
        else:
            snapshot_id = await self.store.select_snapshot(snapshot_id)
        await self._bind_state(snapshot_id, request_hash, skills)

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

    def _build_settings(self):
        """Identify construction inputs independently of query and transport options."""
        config = self.config.model_dump(mode="json", exclude={"embedding": {"api_key"}, "chat": {"api_key"}})
        for key in (
            *RetrievalStrategyConfig.model_fields,
            "mode",
            "query_rewrite",
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
        config["embedding_profile"] = self._embedding_profile().model_dump(mode="json")
        for connection in (config["embedding"], config["chat"]):
            if connection is not None:
                connection["endpoint_hash"] = graph_digest(connection.pop("base_url"))
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
        if not (self.config.llm_linking or self.config.scene_llm_linking or self.config.enrich_skill):
            config.pop("chat")
        return {
            "representation": "scene_graph_v2",
            "config": config,
            "linking_version": "joint_linking_v3",
            "enrichment": enrichment_identity() if self.config.enrich_skill else None,
            "scenes": {
                **self._scene_digests,
                **{key: graph_digest(value.model_dump(mode="json")) for key, value in sorted(self._provided.items())},
            }
            if self.config.build_scene
            else {},
        }

    def _source_hash(self, skills, bindings):
        """Hash exact selected sources and construction settings without query settings."""
        return graph_digest(
            {
                "skills": [[key, skills[key].snapshot().model_dump(mode="json")] for key in sorted(skills)],
                "bindings": bindings,
                "settings": self._build_settings(),
            }
        )

    async def _bind_state(self, snapshot_id, request_hash, member_ids):
        """Validate and pin one committed snapshot for later recommendation."""
        nodes, edges, metadata = await self.store.load(snapshot_id)
        self.build_warnings = [
            value for value in metadata.get("build_warnings", []) if value == "SCENE_GRAPH_ENRICHMENT_RULE_FALLBACK"
        ]
        if metadata.get("build_settings") != self._build_settings():
            raise SceneGraphError(SceneGraphFailure.PROFILE_MISMATCH)
        content = SceneGraphContent(embedding=self._embedding_profile(), nodes=nodes, edges=edges)
        if content.content_digest() != metadata.get("content_hash"):
            raise ValueError("Joint graph logical content digest differs")
        expected = set(member_ids)
        actual = {str(node.metadata["collection_skill_id"]) for node in nodes if node.node_type == "skill"}
        if actual != expected:
            raise ValueError("Persisted graph membership differs from the candidate scope")
        self._state = OrganizedGraph(
            request_hash,
            snapshot_id,
            nodes,
            edges,
            frozenset(expected),
            self._embedding_profile(),
            tuple(self.build_warnings),
        )

    async def load_snapshot(
        self,
        snapshot_id: UUID,
        *,
        skills: Mapping[str, Skill],
        nodes: list[SkillGraphNodeRecord] | None = None,
        edges: list[SkillGraphEdgeRecord] | None = None,
        metadata: dict | None = None,
    ) -> None:
        """Pin compatible committed content without rebuilding or requiring the original SSL.

        Args:
            snapshot_id: Immutable snapshot in this manager's project and graph.
            skills: Complete exact source versions selected from that snapshot by the caller.
            nodes: Already loaded node records, avoiding a second full snapshot read.
            edges: Already loaded edge records matching the supplied nodes.
            metadata: Already loaded snapshot metadata matching the supplied nodes.

        Raises:
            SceneGraphError: The representation or full embedding profile is incompatible.
            ValueError: Content integrity or exact member/version bindings differ.
        """
        self._state = None
        bindings = await self.store.validate_skills(skills)
        if nodes is None or edges is None or metadata is None:
            nodes, edges, metadata = await self.store.load(snapshot_id)
        settings = metadata.get("build_settings", {})
        expected_profile = self._embedding_profile()
        profile = EmbeddingProfile.model_validate(settings.get("config", {}).get("embedding_profile", {}))
        if (
            settings.get("representation") != "scene_graph_v2"
            or profile.node_template not in ("scene_graph_nodes_v2", "scene_graph_nodes_v3")
            or profile.model_dump(exclude={"node_template"}) != expected_profile.model_dump(exclude={"node_template"})
        ):
            raise SceneGraphError(SceneGraphFailure.PROFILE_MISMATCH)
        content = SceneGraphContent(embedding=profile, nodes=nodes, edges=edges)
        if content.content_digest() != metadata.get("content_hash"):
            raise ValueError("Joint graph logical content digest differs")
        actual = {
            node.metadata["collection_skill_id"]: node.skill_version_id for node in nodes if node.node_type == "skill"
        }
        if actual != bindings:
            raise ValueError("Persisted graph versions differ from the candidate scope")
        self.build_warnings = [
            warning
            for warning in metadata.get("build_warnings", [])
            if warning == "SCENE_GRAPH_ENRICHMENT_RULE_FALLBACK"
        ]
        self._state = OrganizedGraph(
            metadata["source_hash"],
            snapshot_id,
            nodes,
            edges,
            frozenset(bindings),
            profile,
            tuple(self.build_warnings),
        )

    def _embedding_profile(self):
        """Resolve deterministic joint text and vector provenance without model I/O."""
        profile = self.config.embedding_profile or EmbeddingProfile(
            space_id=graph_digest(self.config.embedding.base_url),
            model=self.config.embedding.model,
            dimensions=self.config.embedding.dimensions,
            query_template="legacy_query_v1",
        )
        return profile.model_copy(update={"node_template": "scene_graph_nodes_v3"})

    async def _build_joint(self, skills, request_hash, prior):
        """Finalize joint semantics and vectors before candidate linking and atomic publication."""
        get_logger(__name__).info("Graph embedding stage started")
        snapshot_id = uuid4()
        families = await self.store.resolve_families(sorted(skills))
        nodes, edge_data = [], []
        for key in sorted(skills):
            if self.config.build_scene and key not in self._provided and prior is not None:
                captured = [node for node in prior.nodes if node.metadata["collection_skill_id"] == key]
                owner = next(node for node in captured if node.node_type == "skill")
                if owner.skill_version_id != self.store.version_ids[key]:
                    raise ValueError("Captured member version changed")
                base = self._enrichment_inputs.get(key)
                if base is not None:
                    if (base.node_id, base.skill_version_id, base.project_id, base.graph_id) != (
                        owner.node_id,
                        owner.skill_version_id,
                        owner.project_id,
                        owner.graph_id,
                    ):
                        raise ValueError("Captured enrichment input differs from its owner")
                    captured = [base if node.node_type == "skill" else node for node in captured]
                nodes.extend(node.model_copy(update={"graph_version_id": snapshot_id}) for node in captured)
                ids = {node.node_id for node in captured}
                edge_data.extend(
                    {
                        "source": edge.source_node_id,
                        "target": edge.target_node_id,
                        "type": edge.edge_type,
                        "weight": edge.weight,
                        **edge.metadata,
                    }
                    for edge in prior.edges
                    if edge.edge_type in ("flow", "containment")
                    and edge.source_node_id in ids
                    and edge.target_node_id in ids
                )
                continue
            parsed = parse_skill_document(skills[key].content, skill_id=key, max_chars=self.config.max_input_chars)
            values = {
                name: parsed[name]
                for name in (
                    "name",
                    "description",
                    "one_line_capability",
                    "inputs",
                    "outputs",
                    "domain_tags",
                    "tooling",
                    "example_tasks",
                    "rendered_snippet",
                )
            }
            derived, values["tooling"] = document_features(skills[key], parsed, member_id=key, family_id=families[key])
            owner = self.store.bind_node(
                snapshot_id,
                key,
                {
                    "node_id": node_id(key),
                    **values,
                    "metadata": derived.model_dump(mode="json"),
                },
            )
            if not self.config.build_scene:
                nodes.append(owner)
                continue
            record = self._provided[key]
            nodes.append(merge_skill_semantics(owner, record))
            for scene in record.scenes:
                scene_key = node_id(key, scene.scene_id)
                nodes.append(
                    self.store.bind_node(
                        snapshot_id,
                        key,
                        {
                            "node_id": scene_key,
                            "node_type": "scene",
                            "owner_node_id": owner.node_id,
                            **scene_features(scene, record, member_id=key, family_id=families[key]),
                        },
                    )
                )
                edge_data.append({"source": scene_key, "target": owner.node_id, "type": "containment", "weight": 1.0})
                targets = {}
                for rule in scene.next_scene_rules:
                    if not rule.target.startswith("END_") and rule.target != scene.scene_id:
                        targets.setdefault(rule.target, []).append(rule.condition)
                for target, conditions in targets.items():
                    condition = (
                        "" if "" in conditions else " OR ".join(f"({value})" for value in dict.fromkeys(conditions))
                    )
                    if len(conditions) == 1:
                        condition = conditions[0]
                    edge_data.append(
                        {
                            "source": scene_key,
                            "target": node_id(key, target),
                            "type": "flow",
                            "weight": 0.3,
                            "condition": condition,
                        }
                    )

        if self.config.enrich_skill:
            self._enrichment_inputs = {
                node.metadata["collection_skill_id"]: node.model_copy(
                    update={"embedding": None, "embedding_input_hash": None}
                )
                for node in nodes
                if node.node_type == "skill"
            }
            nodes = [await self._enrich(node, skills) if node.node_type == "skill" else node for node in nodes]
        nodes, reused = await self._embed_joint_nodes(nodes, prior)
        get_logger(__name__).info("Graph linking stage started")
        dependencies = list(
            joint_dependency_edges(
                nodes, dependency_match_threshold=self.config.joint_linking.dependency_match_threshold
            )
        )
        edge_data.extend(dependencies)
        dependency_pairs = {frozenset((edge["source"], edge["target"])) for edge in dependencies}
        model_linking = self.config.llm_linking or self.config.scene_llm_linking
        query_vectors = []
        if model_linking and self.config.joint_linking.dense_candidates and len(set(families.values())) > 1:
            by_text = {embedding_text(node): node.embedding for node in nodes}
            query_texts = [linking_query_text(node) for node in nodes]
            missing = list(dict.fromkeys(text for text in query_texts if text not in by_text))
            by_text.update(zip(missing, (await self._embed(missing)), strict=True))
            query_vectors = [by_text[text] for text in query_texts]
        if model_linking and len(set(families.values())) > 1:
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
        if model_linking:
            edge_data.extend((await self._infer_joint_relations(joint_candidate_pairs(nodes, candidates))))

        edges = {}
        for edge in edge_data:
            key = (edge["source"], edge["target"], edge["type"])
            if key in edges:
                continue  # Deterministic evidence takes precedence over optional inference.
            edges[key] = SkillGraphEdgeRecord(
                **self.store.scope,
                graph_version_id=snapshot_id,
                edge_id=uuid4(),
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
        content = SceneGraphContent(embedding=self._embedding_profile(), nodes=nodes, edges=list(edges.values()))
        get_logger(__name__).info("Graph atomic publication started")
        return await self.store.publish(
            snapshot_id,
            request_hash=request_hash,
            nodes=nodes,
            edges=list(edges.values()),
            metadata={
                "build_settings": self._build_settings(),
                "source_hash": request_hash,
                "content_hash": content.content_digest(),
                "extraction_usage": {
                    key: value.model_dump(mode="json") for key, value in self.extraction_summaries.items()
                },
                **(
                    {
                        "enrichment_inputs": {
                            key: value.model_dump(mode="json") for key, value in self._enrichment_inputs.items()
                        },
                        "enrichment_usage": {
                            key: value.model_dump(mode="json") for key, value in self.enrichment_summaries.items()
                        },
                        "build_warnings": list(self.build_warnings),
                    }
                    if self.config.enrich_skill
                    else {}
                ),
                "build_coverage": {
                    "skills": len(skills),
                    "scenes": len(nodes) - len(skills),
                    "edges": len(edges),
                    "vectors": len(nodes),
                    "reused_vectors": reused,
                    "candidate_pairs": sum(1 for _ in joint_candidate_pairs(nodes, candidates)),
                },
            },
        )

    async def _enrich(self, node, skills):
        """Fill only missing fields, retaining safe summaries and explicit fallback warnings."""
        key = node.metadata["collection_skill_id"]
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

    async def _embed_joint_nodes(self, nodes, prior):
        """Reuse only identical rendered text in the same manager's fixed vector profile."""
        compatible = prior is not None and prior.embedding_profile == self._embedding_profile()
        vectors = {embedding_text(node): node.embedding for node in prior.nodes} if compatible else {}
        texts = [embedding_text(node) for node in nodes]
        reused = sum(text in vectors for text in texts)
        missing = list(dict.fromkeys(text for text in texts if text not in vectors))
        vectors.update(zip(missing, (await self._embed(missing)), strict=True))
        return [
            type(node).model_validate(
                {
                    **node.model_dump(),
                    "embedding": vectors[text],
                    "embedding_input_hash": hashlib.sha256(text.encode()).hexdigest(),
                }
            )
            for node, text in zip(nodes, texts, strict=True)
        ], reused

    async def _infer_joint_relations(self, pairs):
        """Validate bounded model proposals without exposing response content on failure."""
        result = {}
        weighted = self.config.joint_linking.relation_weighting == "model_confidence"
        relation_validation = self.config.joint_linking.relation_validation
        system = JOINT_RELATION_WEIGHT_SYSTEM if weighted else JOINT_RELATION_SYSTEM
        while batch := list(islice(pairs, self.config.llm_batch_size)):
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
                    key = (relation.source_id, relation.target_id, relation.edge_type)
                    edge = {
                        "source": relation.source_id,
                        "target": relation.target_id,
                        "type": relation.edge_type,
                        "weight": weight * relation.confidence,
                        "confidence": relation.confidence,
                        "description": relation.evidence,
                    }
                    previous = result.get(key)
                    if previous is None or (edge["confidence"], edge["weight"]) > (
                        previous["confidence"],
                        previous["weight"],
                    ):
                        result[key] = edge
            except Exception as exc:
                raise SceneGraphError(SceneGraphFailure.MODEL_FAILED) from exc
        return list(result.values())

    async def _chat(self, system, payload):
        """Require structured JSON from the configured chat dependency."""
        if self.chat_model is None:
            raise RuntimeError("Chat model is not configured")
        result = await self.chat_model.run(
            system=system, user=json.dumps(payload, ensure_ascii=False), response_format={"type": "json_object"}
        )
        if not isinstance(result.output, str):
            raise TypeError("Graph chat model must return JSON text")
        return result.output

    async def _embed(self, texts):
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

    async def _retrieve(self, state, task, limit):
        """Execute retrieval entirely against the state captured at the public entry."""
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer")
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
        budgets = [("skill", self.config.skill_candidates, self.config.skill_seeds)]
        if self.config.mode in ("unified_ppr", "separate") and self.config.scene_seeds:
            budgets.append(("scene", self.config.scene_candidates, self.config.scene_seeds))
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

    def build_warning_values(self, metadata: dict) -> tuple[str, ...]:
        """Return only allowlisted warnings recognized by retrieval responses.

        Args:
            metadata: Immutable snapshot build metadata.
        """
        return tuple(
            value for value in metadata.get("build_warnings", []) if value == "SCENE_GRAPH_ENRICHMENT_RULE_FALLBACK"
        )

    def validate_vector_metadata(self, metadata: dict) -> EmbeddingProfile:
        """Validate the narrow metadata needed to interpret pure vector hits.

        Args:
            metadata: Immutable snapshot build metadata.
        """
        settings = metadata.get("build_settings", {})
        expected = self._embedding_profile()
        profile = EmbeddingProfile.model_validate(settings.get("config", {}).get("embedding_profile", {}))
        if (
            settings.get("representation") != "scene_graph_v2"
            or profile.node_template not in ("scene_graph_nodes_v2", "scene_graph_nodes_v3")
            or profile.model_dump(exclude={"node_template"}) != expected.model_dump(exclude={"node_template"})
        ):
            raise SceneGraphError(SceneGraphFailure.PROFILE_MISMATCH)
        return profile

    def bind_vector_snapshot(
        self,
        snapshot_id: UUID,
        *,
        skill_ids: frozenset[str] | set[str],
        skill_versions: Mapping[str, str] | None = None,
        warnings: tuple[str, ...] = (),
    ) -> None:
        """Pin only the owner/version identities required by pure vector ranking.

        Args:
            snapshot_id: Immutable snapshot selected before the query.
            skill_ids: Exact collection members authorized by the selected snapshot.
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
            include_scenes: Include Scene vectors and use the best cosine similarity per owner.

        Returns:
            Pinned owner hits with actual cosine similarities and winning node identities.
            Ties use stable node IDs. No lexical fusion, threshold, PPR or field rerank applies.

        Raises:
            ValueError: The limit or instruction is invalid.
            RuntimeError: No successfully loaded or organized scope is available.
        """
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer")
        if not isinstance(include_scenes, bool):
            raise ValueError("include_scenes must be a boolean")
        query = task.instruction.strip()
        if not query:
            raise ValueError("Vector retrieval requires a nonempty instruction")
        state = self._state
        if state is None:
            raise RuntimeError("Organize a valid scope before recommending")
        if not state.skill_ids:
            return SceneGraphVectorRetrieval(
                snapshot_id=state.snapshot_id, include_scenes=include_scenes, warnings=state.warnings
            )
        vector = (await self._embed([query]))[0]
        page_size = max(32, limit * 8)
        offset = 0
        hits = []
        scores = []
        seen = set()
        while True:
            page = await self.store.search(
                state.snapshot_id,
                vector,
                top_k=page_size,
                node_type=None if include_scenes else "skill",
                offset=offset,
            )
            hits.extend(page)
            offset += len(page)
            for hit in page:
                metadata = hit.record["metadata"]
                member_id = metadata["collection_skill_id"]
                if member_id not in state.skill_ids:
                    raise RuntimeError("Recommendation escaped the organized scope")
                if member_id in seen:
                    continue
                seen.add(member_id)
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
                if len(scores) == limit:
                    break
            exhausted = len(page) < page_size
            boundary_safe = (
                len(scores) >= limit and offset < len(hits) and hits[offset].distance > scores[limit - 1].similarity
            )
            if exhausted or boundary_safe:
                break
        return SceneGraphVectorRetrieval(
            snapshot_id=state.snapshot_id, scores=tuple(scores), include_scenes=include_scenes, warnings=state.warnings
        )
