"""Built-in ``skill_manage`` algorithm component.

The factory only validates configuration and stores the build context. It never
opens a database, vector store or model router: resources are constructed by
``compose_runtime`` before the algorithm is built, and the dependency validator
here runs without I/O so the compiler can fail before any resource opens.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

from ...llm import llm_run_context
from ...registry import ComponentRequirements, ComponentType, register
from ...typing.skill_manage import (
    SkillGraphEdge,
    SkillGraphMembership,
    SkillGraphNode,
    SkillManageBuildInput,
    SkillManageBuildPlan,
    SkillManageEmbedInput,
    SkillManageEmbedOutput,
    SkillManageHit,
    SkillManageMutationPlan,
    SkillManageRemoveInput,
    SkillManageRetrieveInput,
    SkillManageRetrieveOutput,
    SkillManageUpsertInput,
    SkillManageWarning,
)
from .config import SkillManageConfig
from .embedding import (
    EmbeddingModel,
    embed_texts,
    require_template,
    skill_embedding_texts,
    with_content_hash,
)
from .errors import configuration_error
from .linking import build_deterministic_dependency_edges
from .parsing import parse_skill_document


def skill_manage_dependency_validator(config: SkillManageConfig) -> dict[str, str]:
    """Resolve the model roles a validated config actually requires.

    ``embedding`` is always required. The ``chat`` role is required only when
    LLM linking is enabled, so deterministic baselines can deploy without a
    chat model.
    """

    roles: dict[str, str] = {"embedding": "embedding"}
    if config.llm_linking:
        roles["chat"] = "chat"
    return roles


def skill_memberships(nodes: Sequence[SkillGraphNode]) -> tuple[SkillGraphMembership, ...]:
    """One membership per skill node: the exact version that node was parsed from."""

    return tuple(
        SkillGraphMembership(skill_id=node.skill_id, version_id=node.version_id, graph_node_id=node.graph_node_id)
        for node in nodes
        if node.node_type == "skill" and node.skill_id and node.version_id
    )


@register(
    type=ComponentType.ALGO,
    name="skill_manage",
    config_model=SkillManageConfig,
    capabilities={"skill_manage"},
    requirements=ComponentRequirements(
        required_model_roles=frozenset({"embedding"}),
        required_model_kinds={"embedding": "embedding"},
        dependency_validator=skill_manage_dependency_validator,
    ),
)
class DefaultSkillManageAlgorithm:
    """Parser/linker/PPR/rerank core behind ``skill_manage``.

    Plan and retrieve methods operate on in-memory inputs and return
    plans/results; ``embed`` is the single I/O boundary and only talks to the
    model bound to the ``embedding`` role of the build context. Nothing here
    touches storage or the canonical Skill version DAG.
    """

    algorithm_name: ClassVar[str] = "skill_manage"
    algorithm_version: ClassVar[str] = "1"

    def __init__(self, *, config: SkillManageConfig, context: object) -> None:
        self._config = config
        self._context = context

    def _embedding_model(self) -> EmbeddingModel:
        models = getattr(self._context, "embedding_models", None) or {}
        model = models.get("embedding") if hasattr(models, "get") else None
        if model is None:
            raise configuration_error("no embedding model is bound to the 'embedding' role")
        return model

    async def plan_build(self, request: SkillManageBuildInput) -> SkillManageBuildPlan:
        require_template(request.embedding_profile)
        parsed = [
            parse_skill_document(
                document.content,
                source_path=document.source_path,
                version_id=document.version_id,
                max_chars=self._config.max_input_chars,
            )
            for document in request.documents
        ]
        nodes: list[SkillGraphNode] = []
        for document, skill in zip(request.documents, parsed):
            nodes.append(
                SkillGraphNode(
                    graph_node_id=f"skill:{document.skill_id}",
                    node_type="skill",
                    skill_id=document.skill_id,
                    version_id=document.version_id,
                    name=skill["name"],
                    description=skill["description"],
                    one_line_capability=skill["one_line_capability"],
                    inputs=tuple(skill["inputs"]),
                    outputs=tuple(skill["outputs"]),
                    domain_tags=tuple(skill["domain_tags"]),
                    tooling=tuple(skill["tooling"]),
                    example_tasks=tuple(skill["example_tasks"]),
                    rendered_snippet=skill["rendered_snippet"],
                )
            )
        node_index = {node.name: node for node in nodes}
        raw_edges = build_deterministic_dependency_edges(
            [{"node_type": "skill", **node.model_dump(mode="python")} for node in nodes],
            dependency_match_threshold=request.dependency_match_threshold,
        )
        edges: list[SkillGraphEdge] = []
        for index, edge in enumerate(raw_edges):
            edges.append(
                SkillGraphEdge(
                    edge_id=f"edge:{index}:{edge['source']}:{edge['target']}:{edge['type']}",
                    edge_type=edge["type"],
                    source_id=node_index[edge["source"]].graph_node_id,
                    target_id=node_index[edge["target"]].graph_node_id,
                    weight=float(edge["weight"]),
                )
            )
        warnings: list[SkillManageWarning] = []
        if request.llm_linking and self._context is None:
            warnings.append(SkillManageWarning(code="SKILL_MANAGE_LLM_LINKING_DEGRADED"))
        hashed_nodes = with_content_hash(nodes)
        return SkillManageBuildPlan(
            scope=request.scope,
            nodes=hashed_nodes,
            edges=tuple(edges),
            memberships=skill_memberships(hashed_nodes),
            embedding_texts=skill_embedding_texts(hashed_nodes),
            warnings=tuple(warnings),
        )

    async def plan_upsert(self, request: SkillManageUpsertInput) -> SkillManageMutationPlan:
        """Replace one Skill's node with the re-parsed version document.

        When ``request.document`` is supplied, the target Skill node is rebuilt
        from the new canonical content and deterministic dependency edges are
        recomputed across the whole skill node set. Scene nodes are preserved.
        Without replacement content the topology must stay identical, so the
        snapshot is returned unchanged and the mutation is a pure revision
        advance. Memberships are always derived from the resulting skill nodes,
        so the replaced node's membership points at the new ``version_id``.
        """

        require_template(request.embedding_profile)
        snapshot = request.snapshot
        document = request.document
        if document is None:
            return SkillManageMutationPlan(
                scope=request.scope,
                nodes=snapshot.nodes,
                edges=snapshot.edges,
                memberships=skill_memberships(snapshot.nodes),
                embedding_texts=skill_embedding_texts(snapshot.nodes),
            )

        parsed = parse_skill_document(
            document.content,
            source_path=document.source_path,
            version_id=document.version_id,
            max_chars=self._config.max_input_chars,
        )
        target_node_id = f"skill:{document.skill_id}"
        replacement = SkillGraphNode(
            graph_node_id=target_node_id,
            node_type="skill",
            skill_id=document.skill_id,
            version_id=document.version_id,
            name=parsed["name"],
            description=parsed["description"],
            one_line_capability=parsed["one_line_capability"],
            inputs=tuple(parsed["inputs"]),
            outputs=tuple(parsed["outputs"]),
            domain_tags=tuple(parsed["domain_tags"]),
            tooling=tuple(parsed["tooling"]),
            example_tasks=tuple(parsed["example_tasks"]),
            rendered_snippet=parsed["rendered_snippet"],
        )
        nodes = tuple(replacement if node.graph_node_id == target_node_id else node for node in snapshot.nodes)
        nodes = with_content_hash(nodes)

        skill_nodes = [node for node in nodes if node.node_type == "skill"]
        raw_edges = build_deterministic_dependency_edges(
            [{"node_type": "skill", **node.model_dump(mode="python")} for node in skill_nodes],
            dependency_match_threshold=request.dependency_match_threshold,
        )
        node_index = {node.name: node for node in skill_nodes}
        edges: list[SkillGraphEdge] = []
        for index, edge in enumerate(raw_edges):
            source = node_index[edge["source"]]
            target = node_index[edge["target"]]
            edges.append(
                SkillGraphEdge(
                    edge_id=f"edge:{index}:{edge['source']}:{edge['target']}:{edge['type']}",
                    edge_type=edge["type"],
                    source_id=source.graph_node_id,
                    target_id=target.graph_node_id,
                    weight=float(edge["weight"]),
                )
            )
        # Preserve any non-dependency edges (e.g. Scene containment/flow) still
        # touching surviving nodes; the deterministic linker only emits dependency
        # edges, and dropping the rest would silently lose topology.
        surviving_ids = {node.graph_node_id for node in nodes}
        edges.extend(
            edge
            for edge in snapshot.edges
            if edge.edge_type != "dependency" and edge.source_id in surviving_ids and edge.target_id in surviving_ids
        )
        return SkillManageMutationPlan(
            scope=request.scope,
            nodes=nodes,
            edges=tuple(edges),
            memberships=skill_memberships(nodes),
            embedding_texts=skill_embedding_texts(nodes),
        )

    async def plan_remove(self, request: SkillManageRemoveInput) -> SkillManageMutationPlan:
        """Graph-only remove: drop the target node and its incident edges.

        The canonical Skill family/version is never touched; only this graph
        generation's membership/nodes/edges are excluded.
        """

        snapshot = request.snapshot
        selector = request.selector
        if selector.kind == "version":
            removed = {
                node.graph_node_id
                for node in snapshot.nodes
                if node.node_type == "skill"
                and node.skill_id == selector.skill_id
                and node.version_id == selector.version_id
            }
        else:
            removed = {
                node.graph_node_id
                for node in snapshot.nodes
                if node.node_type == "skill" and node.skill_id == selector.skill_id
            }
        # Cascading removal: Scene nodes owned by a removed Skill are dropped
        # together with it (their incident edges go via the edge filter below).
        removed.update(
            node.graph_node_id
            for node in snapshot.nodes
            if node.node_type == "scene" and node.parent_skill_id == selector.skill_id
        )
        remaining_nodes = tuple(node for node in snapshot.nodes if node.graph_node_id not in removed)
        remaining_edges = tuple(
            edge for edge in snapshot.edges if edge.source_id not in removed and edge.target_id not in removed
        )
        return SkillManageMutationPlan(
            scope=request.scope,
            nodes=remaining_nodes,
            edges=remaining_edges,
            memberships=skill_memberships(remaining_nodes),
            embedding_texts=skill_embedding_texts(remaining_nodes),
        )

    async def embed(self, request: SkillManageEmbedInput) -> SkillManageEmbedOutput:
        model = self._embedding_model()
        if request.run_id is None:
            vectors = await embed_texts(model, request.embedding_profile, request.texts, task=request.task)
        else:
            with llm_run_context(request.run_id):
                vectors = await embed_texts(model, request.embedding_profile, request.texts, task=request.task)
        return SkillManageEmbedOutput(embedding_profile=request.embedding_profile, vectors=vectors)

    async def retrieve(self, request: SkillManageRetrieveInput) -> SkillManageRetrieveOutput:
        # Component discovery must not require the optional NumPy dependency.
        from .retrieval import retrieve_skills

        snapshot = request.snapshot
        skill_nodes = [node for node in snapshot.nodes if node.node_type == "skill"]
        # The PPR core indexes by node name and reads EdgeLike.source/.target, so
        # adapt the domain SkillGraphEdge (source_id/target_id/edge_type) to the
        # protocol (source/target/type) and translate graph_node_id -> name.
        name_by_node_id = {node.graph_node_id: node.name for node in skill_nodes}

        class _EdgeLike:
            __slots__ = ("source", "target", "type", "weight")

            def __init__(self, edge: SkillGraphEdge) -> None:
                self.source = name_by_node_id.get(edge.source_id, "")
                self.target = name_by_node_id.get(edge.target_id, "")
                self.type = edge.edge_type
                self.weight = edge.weight

        edges = [_EdgeLike(edge) for edge in snapshot.edges if edge.edge_type not in {"containment", "flow"}]
        has_scenes = any(node.node_type == "scene" for node in snapshot.nodes)
        hits, warnings = retrieve_skills(
            nodes=skill_nodes,
            edges=edges,
            vector_seed=request.vector_seed,
            vector_similarity=request.vector_similarity,
            lexical_seed=request.lexical_seed,
            query=request.query,
            mode=request.mode.value,
            has_scenes=has_scenes,
            top_k=request.top_k,
        )

        return SkillManageRetrieveOutput(
            generation_id=snapshot.generation_id,
            revision=snapshot.revision,
            hits=tuple(
                SkillManageHit(
                    skill_id=str(hit.get("skill_id") or hit.get("name") or ""),
                    version_id=str(hit.get("version_id") or ""),
                    name=str(hit.get("name") or ""),
                    graph_node_id=str(hit.get("graph_node_id") or ""),
                    score=float(hit["score"]),
                    vector_score=hit.get("vector_score"),
                    ppr_score=float(hit.get("ppr_score") or 0.0),
                    rerank_score=float(hit.get("rerank_score") or 0.0),
                )
                for hit in hits
            ),
            warnings=tuple(
                SkillManageWarning(
                    code=str(warning.get("code") or ""),
                    message=str(warning.get("message") or ""),
                    details=dict(warning.get("details") or {}),
                )
                for warning in warnings
            ),
        )


__all__ = [
    "DefaultSkillManageAlgorithm",
    "SkillManageConfig",
    "skill_manage_dependency_validator",
    "skill_memberships",
]
