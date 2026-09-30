"""Source field reranking and ID-indexed Skill/Scene PageRank."""

import re
from collections.abc import Mapping

from .linking import signature_tokens


def extract_task_name(query: str) -> str:
    """Derive the source-compatible short task slug.

    Args:
        query: Public task retrieval text.
    """
    tokens = [token for token in re.split(r"[^a-zA-Z0-9]+", query.strip()) if token]
    if not tokens:
        return ""
    slug = "-".join(token.lower() for token in tokens[:8])
    return slug[:80]


def extract_artifacts(query: str) -> list[str]:
    """Find explicit artifact filenames in the public task.

    Args:
        query: Public task retrieval text.
    """
    artifacts = re.findall(r"[A-Za-z0-9_./-]+\.(?:py|md|json|csv|stl|dot|txt|yaml|yml|bib|pptx|xlsx|docx)", query)
    return _dedupe_text([artifact.strip() for artifact in artifacts if artifact.strip()])


def fallback_query_schema(query: str) -> dict[str, list[str] | str]:
    """Derive deterministic query fields when rewriting is disabled.

    Args:
        query: Public task retrieval text.
    """
    normalized_query = re.sub(r"\s+", " ", query.strip())
    artifacts = extract_artifacts(normalized_query)
    keywords = sorted(signature_tokens([normalized_query, *artifacts]))
    return {
        "goal": normalized_query,
        "task_name": extract_task_name(normalized_query),
        "domain": [],
        "operations": [],
        "artifacts": artifacts,
        "constraints": [],
        "keywords": list(keywords),
    }


def query_schema_values(query_schema: Mapping[str, object]) -> list[str]:
    """Flatten public query fields in source order.

    Args:
        query_schema: Public goal, operations, constraints, and search terms.
    """
    values: list[str] = []
    for key in ("goal", "task_name", "domain", "operations", "artifacts", "constraints", "keywords"):
        raw = query_schema.get(key, [])
        if isinstance(raw, str):
            values.append(raw)
        elif isinstance(raw, list):
            values.extend(str(item) for item in raw)
    return [value for value in values if value]


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


def _dedupe_text(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = re.sub(r"\s+", " ", str(value or "").strip())
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(normalized)
    return result


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
        seeds: Unique seed node IDs in descending semantic relevance.
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
        # Preserve the source core's zero-weight fallback for numerical parity.
        weight = edge.weight or 1.0
        adjacency[left][right] = adjacency[left].get(right, 0.0) + weight
        reverse = REVERSE_WEIGHTS[edge.edge_type] * weight
        adjacency[right][left] = adjacency[right].get(left, 0.0) + reverse
    for index, row in enumerate(adjacency):
        total = sum(row.values())
        adjacency[index] = {target: weight / total for target, weight in row.items()} if total else {index: 1.0}
    unique_seeds = list(dict.fromkeys(key for key in seeds if key in indices))
    total = sum(1 / rank for rank in range(1, len(unique_seeds) + 1))
    personal = [0.0 for _ in ids]
    for rank, key in enumerate(unique_seeds, 1):
        personal[indices[key]] = 1 / rank / total
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


def rank_skills(nodes, edges, seeds, query_schema, *, config, limit):
    """Rank member IDs while mapping scene evidence back to its owning Skill.

    Args:
        nodes: Persisted Skill and Scene nodes.
        edges: Persisted relationships.
        seeds: Ordered database vector-neighbor node IDs.
        query_schema: Validated public task fields used for source field scoring.
        config: PageRank and scene-mode settings.
        limit: Maximum number of unique collection members to return.
    """
    active = nodes if config.mode == "unified_ppr" else [node for node in nodes if node.node_type == "skill"]
    scores = pagerank(
        active, edges, seeds, damping=config.damping, max_iter=config.max_iter, tolerance=config.tolerance
    )
    owners = {node.node_id: node.owner_node_id or node.node_id for node in active}
    aggregated = {}
    for key, score in scores.items():
        owner = owners[key]
        aggregated[owner] = aggregated.get(owner, 0.0) + score
    semantic_ranks = {}
    for rank, key in enumerate(seeds, 1):
        if key in owners:
            semantic_ranks.setdefault(owners[key], rank)
    hits = []
    for node in active:
        if node.node_type != "skill":
            continue
        graph_score = aggregated.get(node.node_id, 0.0)
        score = rerank_skill_score(query_schema, node.model_dump(), graph_score, semantic_ranks.get(node.node_id))
        hits.append((str(node.metadata["collection_skill_id"]), score, graph_score))
    hits.sort(key=lambda hit: (-hit[1], -hit[2], hit[0]))
    return [hit[0] for hit in hits[:limit]]
