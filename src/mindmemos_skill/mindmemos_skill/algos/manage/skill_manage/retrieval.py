"""Source field reranking and ID-indexed Skill/Scene PageRank."""

from collections import Counter
from collections.abc import Mapping
from math import exp, isfinite, log

from .edge_filters import filter_edges
from .linking import signature_tokens
from .query import query_schema_values
from .seeding import field_score
from .typing import SceneGraphScore


def token_overlap_score(query_tokens: set[str], values: list[str]) -> float:
    """Measure the fraction of query tokens present in candidate fields.

    Args:
        query_tokens: Normalized public query tokens.
        values: Public field strings used for tokenization or matching.
    """
    candidate_tokens = signature_tokens(values)
    if not query_tokens or not candidate_tokens:
        return 0.0
    overlap = query_tokens & candidate_tokens
    if not overlap:
        return 0.0
    return len(overlap) / max(len(query_tokens), 1)


def _field_bonus(query_tokens: set[str], values: list[str], weight: float) -> float:
    return weight * token_overlap_score(query_tokens, values)


def rerank_skill_score(
    query_schema: Mapping[str, object],
    node: Mapping[str, object],
    graph_score: float,
    semantic_rank: int | None,
) -> float:
    """Port of the engine's ``_rerank_skill_score`` field-weighted scoring.

    Args:
        query_schema: Public goal, operations, constraints, and search terms.
        node: Parsed Skill fields including its stable collection node ID.
        graph_score: Personalized PageRank evidence for this Skill.
        semantic_rank: One-based vector-neighbor rank, or None when not a vector seed.
    """

    query_tokens = signature_tokens(query_schema_values(query_schema))
    score = graph_score
    task_name = str(query_schema.get("task_name") or "")
    if task_name:
        score += _field_bonus(query_tokens, [task_name], 0.35)
    score += _field_bonus(query_tokens, [str(node.get("name") or "")], 1.25)
    score += _field_bonus(
        query_tokens,
        [str(node.get("one_line_capability") or ""), str(node.get("description") or "")],
        0.9,
    )
    score += _field_bonus(query_tokens, _as_list(node.get("domain_tags")), 1.15)
    score += _field_bonus(query_tokens, _as_list(node.get("tooling")), 0.95)
    score += _field_bonus(query_tokens, _as_list(node.get("inputs")) + _as_list(node.get("outputs")), 0.75)
    score += _field_bonus(query_tokens, _as_list(node.get("example_tasks")), 0.8)
    if semantic_rank is not None:
        score += 0.2 / float(semantic_rank)
    return score


def _as_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    return [str(value)]


REVERSE_WEIGHTS = {
    "dependency": 1.0,
    "workflow": 0.5,
    "semantic": 0.2,
    "alternative": 0.1,
    "containment": 0.1,
    "flow": 0.1,
}
"""Source reverse-propagation weights for each relationship type."""


def pagerank(nodes, edges, seeds, *, damping=0.2, max_iter=50, tolerance=1e-6):
    """Apply the source rank-seeded PPR recurrence over stable node IDs.

    Args:
        nodes: Nodes in a single immutable graph snapshot.
        edges: Directed weighted relationships within that snapshot.
        seeds: Explicit node-to-weight mapping, or legacy rank-ordered node IDs.
        damping: Restart probability.
        max_iter: Maximum iteration count.
        tolerance: L1 difference below which iteration stops.

    Returns:
        Node ID to normalized score; no seeds produce zero scores.
    """
    ids = [node.node_id for node in nodes]
    indices = {key: index for index, key in enumerate(ids)}
    adjacency = [{} for _ in ids]
    for edge in edges:
        if edge.source_node_id not in indices or edge.target_node_id not in indices:
            continue
        left, right = indices[edge.source_node_id], indices[edge.target_node_id]
        weight = edge.weight
        adjacency[left][right] = adjacency[left].get(right, 0.0) + weight
        reverse = REVERSE_WEIGHTS[edge.edge_type] * weight
        adjacency[right][left] = adjacency[right].get(left, 0.0) + reverse
    for index, row in enumerate(adjacency):
        total = sum(row.values())
        adjacency[index] = {target: weight / total for target, weight in row.items()} if total else {index: 1.0}
    weights = (
        dict(seeds)
        if isinstance(seeds, Mapping)
        else {key: 1 / rank for rank, key in enumerate(dict.fromkeys(key for key in seeds if key in indices), 1)}
    )
    if any(not isfinite(value) or value < 0 for value in weights.values()):
        raise ValueError("Seed weights must be finite and nonnegative")
    total = sum(value for key, value in weights.items() if key in indices)
    personal = [weights.get(key, 0.0) / total if total else 0.0 for key in ids]
    scores = personal.copy()
    for _ in range(max_iter):
        updated = [damping * weight for weight in personal]
        for index, row in enumerate(adjacency):
            for target, weight in row.items():
                updated[target] += (1 - damping) * scores[index] * weight
        difference = sum(abs(left - right) for left, right in zip(scores, updated, strict=True))
        scores = updated
        if difference <= tolerance:
            break
    total = sum(scores)
    return {key: value / total if total else 0.0 for key, value in zip(ids, scores, strict=True)}


def rank_joint(nodes, edges, seeds, schema, *, config, limit, walk_weights=None) -> list[SceneGraphScore]:
    """Truncate by aggregated graph evidence before field reranking Skill members.

    Args:
        nodes: Joint nodes in a single immutable snapshot.
        edges: Relationships from the same snapshot.
        seeds: Typed seeds with explicit combined personalization weights. Final
            rank bonuses use declared fused Skill ranks, if present.
        schema: Normalized public task query.
        config: Active mode, PPR settings and alpha/max/mean aggregation.
        limit: Graph candidate cap and maximum final result count.
        walk_weights: Optional collector of actual mode-specific personalization priors.

    Returns:
        Skill-only score components in final order; no valid seeds produce no results.
    """
    edges = filter_edges(nodes, edges, config)
    skills = [node for node in nodes if node.node_type == "skill"]
    scenes = [node for node in nodes if node.node_type == "scene"]
    skill_ids = {node.node_id for node in skills}
    scene_ids = {node.node_id for node in scenes}
    skill_weights = _normalize({seed.node_id: seed.weight for seed in seeds if seed.node_id in skill_ids})
    scene_weights = _normalize({seed.node_id: seed.weight for seed in seeds if seed.node_id in scene_ids})
    if config.mode in ("skill_only", "fuse"):
        scene_weights = {}
    scene_weights = _suppress_scenes(scenes, edges, scene_weights, config)
    if not skill_weights and not scene_weights:
        return []

    def walk(active, weights):
        """Run the shared sparse recurrence on an explicit projection."""
        return pagerank(
            active, edges, weights, damping=config.damping, max_iter=config.max_iter, tolerance=config.tolerance
        )

    if walk_weights is not None:
        actual_scenes = scene_weights
        if config.mode == "unified_ppr" or config.scene_ppr:
            actual_scenes = _normalize(scene_weights)
        priors = {**skill_weights, **actual_scenes}
        if config.mode == "unified_ppr":
            priors = _normalize(priors)
        walk_weights.update(priors)
    scene_scores = {}
    if config.mode == "skill_only" or not scenes:
        scores = walk(skills, skill_weights)
    elif config.mode == "separate":
        scores = walk(nodes, skill_weights)
        scene_scores = walk(scenes, scene_weights) if config.scene_ppr else scene_weights
    elif config.mode == "fuse":
        sub, full = walk(skills, skill_weights), walk(nodes, skill_weights)
        scores = {key: config.fusion_lambda * sub[key] + (1 - config.fusion_lambda) * full[key] for key in skill_ids}
    else:
        scores = walk(nodes, {**skill_weights, **_normalize(scene_weights)})
        scene_scores = {key: scores[key] for key in scene_ids}
    children = {}
    for node in scenes:
        children.setdefault(node.owner_node_id, []).append(scene_scores.get(node.node_id, 0.0))
    graph = []
    for node in skills:
        child_scores = children.get(node.node_id, [])
        boost = 0.0
        if child_scores:
            aggregate = max(child_scores) if config.aggregation == "max" else sum(child_scores) / len(child_scores)
            boost = config.alpha * aggregate
        own = scores[node.node_id]
        graph.append((node, own, boost, own + boost))
    graph.sort(key=lambda item: (-item[3], str(item[0].metadata["collection_skill_id"])))
    fused_ranks = {seed.node_id: seed.fused_rank for seed in seeds if seed.node_id in skill_ids}
    results = [
        SceneGraphScore(
            member_id=node.metadata["collection_skill_id"],
            ppr_score=own,
            scene_boost=boost,
            graph_score=total,
            rerank_score=field_score(schema, node, total, fused_ranks.get(node.node_id)),
        )
        for node, own, boost, total in graph[:limit]
    ]
    return sorted(results, key=lambda hit: (-hit.rerank_score, -hit.graph_score, hit.member_id))


def _normalize(weights):
    """Normalize one node type while preserving an absent seed route."""
    total = sum(weights.values())
    return {key: value / total for key, value in weights.items()} if total else {}


def _suppress_scenes(scenes, edges, weights, config):
    """Dampen Scene priors by sibling count and cross-family Scene connections."""
    counts = Counter(node.owner_node_id for node in scenes)
    by_id = {node.node_id: node for node in scenes}
    degree = Counter()
    for edge in edges:
        left, right = by_id.get(edge.source_node_id), by_id.get(edge.target_node_id)
        if left and right and left.metadata["family_id"] != right.metadata["family_id"]:
            degree.update((left.node_id, right.node_id))
    return {
        key: weight
        * exp(-config.scene_count_damp * log(counts[by_id[key].owner_node_id]))
        / (1 + config.scene_uniqueness * degree[key])
        for key, weight in weights.items()
    }
