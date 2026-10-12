"""Persisted Skill/Scene graphs with model-assisted task-time recommendation."""

import hashlib
import json
from asyncio import Lock
from collections.abc import Mapping
from itertools import combinations, islice
from uuid import UUID, uuid4

from ....agents import OpenAIChatAgent
from ....envs.typing import Task
from ....infra.database import Vector
from ....infra.embedding import EmbeddingModel, OpenAIEmbeddingClient
from ....persistence.db_services.skill_management.typing import SkillGraphEdgeRecord
from ....persistence.db_services.skill_management.writer import SkillGraphStore, graph_digest
from ....skill import Skill
from ...base import Algorithm
from ..base import SkillManager
from .config import SkillManageConfig
from .linking import build_deterministic_dependency_edges
from .parsing import parse_skill_document
from .prompts import QUERY_SYSTEM, RELATION_SYSTEM
from .retrieval import fallback_query_schema, query_schema_values, rank_skills
from .typing import ChatModel, OrganizedGraph, QuerySchema, Relations, SceneRecord


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
    return "\n".join(parts)


@Algorithm.register("skill_manage")
class SkillManageAlgorithm(SkillManager[SkillManageConfig]):
    """Build only caller-selected Skills and recommend within a pinned graph."""

    config_type = SkillManageConfig
    """Validated graph and model settings used by Algorithm.build."""

    def __init__(
        self,
        config: SkillManageConfig,
        *,
        store: SkillGraphStore,
        scene_records: Mapping[str, SceneRecord] | None = None,
        embedding_model: EmbeddingModel | None = None,
        chat_model: ChatModel | None = None,
    ):
        """Retain borrowed dependencies without database or model I/O.

        Args:
            config: Explicit retrieval and model settings.
            store: Scheduler-bound graph storage; owns project/version validation.
            scene_records: Explicit scene records keyed by collection Skill ID.
            embedding_model: Optional model matching the configured embedding settings.
            chat_model: Optional chat model matching configured non-secret settings.
        """
        super().__init__(config.model_copy(deep=True))
        if config.build_scene and scene_records is None:
            raise ValueError("build_scene requires explicit scene_records; use {} for no scenes")
        if embedding_model is not None and embedding_model.config != config.embedding:
            raise ValueError("Injected embedding model must match configured settings")
        if chat_model is not None and chat_model.config != config.chat:
            raise ValueError("Injected chat model must match configured settings")
        self.store = store
        self.scene_records = {key: record.model_copy(deep=True) for key, record in (scene_records or {}).items()}
        self.embedding_model = embedding_model or OpenAIEmbeddingClient(self.config.embedding)
        self.chat_model = chat_model or (OpenAIChatAgent(self.config.chat) if self.config.chat is not None else None)
        self._lock = Lock()
        self._state: OrganizedGraph | None = None

    def config_snapshot(self) -> dict:
        """Return stable explicit configuration without triggering graph construction."""
        return {
            "algorithm": "skill_manage",
            "schema_version": 1,
            "config": self.config.model_dump(mode="json"),
            "scene_records": {key: value.model_dump(mode="json") for key, value in self.scene_records.items()},
            "storage": self.store.scope,
        }

    async def organize(self, *, skills: Mapping[str, Skill]) -> None:
        """Validate the explicit scope and reuse or atomically publish its graph.

        Args:
            skills: Complete candidate mapping, replacing any previous scope.

        Raises:
            Exception: A source, model, or storage failure invalidates this manager
                until a later organize call succeeds; committed history is retained.
        """
        async with self._lock:
            prior = self._state
            self._state = None
            if set(self.scene_records) - set(skills):
                raise ValueError("Scene records refer to Skills outside the candidate scope")
            bindings = await self.store.validate_skills(skills)
            settings = self.config_snapshot()
            if self.store.message is not None:
                settings["message"] = self.store.message
            request_hash = graph_digest(
                {
                    "skills": [[key, skill.snapshot().model_dump(mode="json")] for key, skill in skills.items()],
                    "bindings": bindings,
                    "settings": settings,
                }
            )
            if prior is not None and prior.request_hash == request_hash:
                self._state = prior
                return
            snapshot_id = await self.store.find(request_hash)
            if snapshot_id is None:
                snapshot_id = await self._build(skills, request_hash)
            nodes, edges, _ = await self.store.load(snapshot_id)
            expected = set(skills)
            actual = {str(node.metadata["collection_skill_id"]) for node in nodes if node.node_type == "skill"}
            if actual != expected:
                raise ValueError("Persisted graph membership differs from the candidate scope")
            self._state = OrganizedGraph(request_hash, snapshot_id, nodes, edges, frozenset(expected))

    @property
    def snapshot_id(self) -> UUID | None:
        """Return the pinned snapshot after a successful organization or restore."""
        return self._state.snapshot_id if self._state is not None else None

    async def restore(self, snapshot_id: UUID, *, skills: Mapping[str, Skill]) -> None:
        """Pin a persisted graph for recommendation without rebuilding it.

        Args:
            snapshot_id: Exact immutable graph version selected by the caller.
            skills: Complete project-owned version bindings for that snapshot.

        Raises:
            ValueError: Stored membership or algorithm settings differ from the request.
        """
        async with self._lock:
            self._state = None
            await self.store.validate_skills(skills)
            nodes, edges, metadata = await self.store.load(snapshot_id)
            expected_config = self.config.model_dump(mode="json")
            if metadata.get("algorithm") != "skill_manage" or metadata.get("config") != expected_config:
                raise ValueError("Graph algorithm settings differ from the recommendation pipeline")
            members = {node.metadata.get("collection_skill_id"): node for node in nodes if node.node_type == "skill"}
            if set(members) != set(skills) or len(members) != sum(node.node_type == "skill" for node in nodes):
                raise ValueError("Stored graph membership differs from the supplied Skill versions")
            for skill_id, node in members.items():
                if node.node_id != node_id(skill_id):
                    raise ValueError("Stored graph member identity is invalid")
            self._state = OrganizedGraph("", snapshot_id, nodes, edges, frozenset(skills))

    async def _build(self, skills, request_hash):
        """Build graph features and vectors before opening the publication transaction."""
        snapshot_id = uuid4()
        nodes = []
        for key, skill in skills.items():
            parsed = parse_skill_document(skill.content, source_path=key, max_chars=self.config.max_input_chars)
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
            nodes.append(self.store.bind_node(snapshot_id, key, {"node_id": node_id(key), **values}))
        edge_data = build_deterministic_dependency_edges(
            [node.model_dump() for node in nodes], dependency_match_threshold=self.config.dependency_match_threshold
        )
        if self.config.llm_linking:
            edge_data.extend((await self._infer_relations(nodes)))
        if self.config.build_scene:
            for key, record in self.scene_records.items():
                for scene in record.scenes:
                    scene_key = node_id(key, scene.scene_id)
                    nodes.append(
                        self.store.bind_node(
                            snapshot_id,
                            key,
                            {
                                "node_id": scene_key,
                                "node_type": "scene",
                                "owner_node_id": node_id(key),
                                "name": scene.scene_name,
                                "description": scene.scene_goal,
                                "one_line_capability": scene.scene_name,
                                "metadata": {"scene_id": scene.scene_id, "scene_type": scene.scene_type},
                            },
                        )
                    )
                    edge_data.append(
                        {"source": node_id(key), "target": scene_key, "type": "containment", "weight": 1.0}
                    )
                    edge_data.extend(
                        {"source": scene_key, "target": node_id(key, target), "type": "flow", "weight": 0.3}
                        for target in scene.next_scene_rules
                    )
        texts = [embedding_text(node) for node in nodes]
        vectors = await self._embed(texts)
        nodes = [
            type(node).model_validate(
                {
                    **node.model_dump(),
                    "embedding": vector,
                    "embedding_input_hash": hashlib.sha256(text.encode()).hexdigest(),
                }
            )
            for node, vector, text in zip(nodes, vectors, texts, strict=True)
        ]
        unique_edges = {}
        for edge in edge_data:
            unique_edges[(edge["source"], edge["target"], edge["type"])] = SkillGraphEdgeRecord(
                **self.store.scope,
                graph_version_id=snapshot_id,
                edge_id=uuid4(),
                source_node_id=edge["source"],
                target_node_id=edge["target"],
                edge_type=edge["type"],
                weight=edge["weight"],
                metadata={"evidence": edge.get("description", "")},
            )
        return await self.store.publish(
            snapshot_id,
            request_hash=request_hash,
            nodes=nodes,
            edges=list(unique_edges.values()),
            metadata=self.config_snapshot(),
        )

    async def _infer_relations(self, nodes):
        """Infer additional relations in bounded batches without merging same-name nodes."""
        pairs = combinations(nodes, 2)
        result = []
        while batch := list(islice(pairs, self.config.llm_batch_size)):
            payload = [
                [
                    {
                        "node_id": node.node_id,
                        "name": node.name,
                        "description": node.description,
                        "inputs": node.inputs,
                        "outputs": node.outputs,
                    }
                    for node in pair
                ]
                for pair in batch
            ]
            response = await self._chat(RELATION_SYSTEM, {"pairs": payload})
            relations = Relations.model_validate_json(response)
            allowed = {frozenset((left.node_id, right.node_id)) for left, right in batch}
            for relation in relations.relations:
                if (
                    frozenset((relation.source_id, relation.target_id)) not in allowed
                    or relation.source_id == relation.target_id
                ):
                    raise ValueError("Model relation is outside the supplied candidate pairs")
                result.append(
                    {
                        "source": relation.source_id,
                        "target": relation.target_id,
                        "type": relation.edge_type,
                        "weight": relation.weight,
                        "description": relation.evidence,
                    }
                )
        return result

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
        return [validator.validate(vector) for vector in vectors]

    async def recommend(self, *, task: Task, limit: int = 5) -> list[str]:
        """Rewrite and embed the public task, then rank only this manager's scope.

        Args:
            task: Public task without private scoring fields.
            limit: Positive maximum number of unique candidate Skill IDs.
        """
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer")
        state = self._state
        if state is None:
            raise RuntimeError("Organize a valid scope before recommending")
        if not state.skill_ids:
            return []
        if self.config.query_rewrite:
            schema = QuerySchema.model_validate_json(
                (await self._chat(QUERY_SYSTEM, task.model_dump(mode="json")))
            ).model_dump()
            query = "\n".join(query_schema_values(schema))
        else:
            query = task.instruction
            schema = fallback_query_schema(query)
        vector = (await self._embed([query]))[0]
        hits = await self.store.search(
            state.snapshot_id,
            vector,
            top_k=self.config.seed_top_k,
            node_type="skill" if self.config.mode == "skill_only" else None,
        )
        seeds = [hit.record["node_id"] for hit in hits if 1 - hit.distance >= self.config.score_threshold]
        result = rank_skills(state.nodes, state.edges, seeds, schema, config=self.config, limit=limit)
        if any(key not in state.skill_ids for key in result):
            raise RuntimeError("Recommendation escaped the organized scope")
        return result
