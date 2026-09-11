"""Backend-neutral PPR core for the ``skill_manage`` algorithm.

Ported from the GOS ``core/retrieval.py`` pure math and kept free of any
storage driver. Uses only the NumPy ``float64`` public API.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from .errors import invalid_graph_weight
from .linking import signature_tokens


class NodeLike(Protocol):
    name: str


class EdgeLike(Protocol):
    source: str
    target: str
    type: str
    weight: float


DEFAULT_REVERSE_WEIGHTS: dict[str, float] = {
    "dependency": 1.0,
    "workflow": 0.5,
    "semantic": 0.2,
    "alternative": 0.1,
    "containment": 0.1,
    "flow": 0.1,
}

TYPE_WEIGHTS: dict[str, float] = {
    "dependency": 1.0,
    "semantic": 0.4,
    "workflow": 0.3,
    "alternative": 0.2,
}


def build_rank_distribution(count: int) -> np.ndarray:
    if count <= 0:
        return np.array([], dtype=float)
    weights = np.array([1.0 / float(index + 1) for index in range(count)], dtype=float)
    return weights / weights.sum()


def build_personalization(
    node_count: int,
    seed_indices: Sequence[int],
    seed_weights: Sequence[float] | None = None,
) -> np.ndarray:
    personalization = np.zeros(node_count, dtype=float)
    if node_count == 0 or not seed_indices:
        return personalization
    if seed_weights is None or len(seed_weights) != len(seed_indices):
        normalized_weights = build_rank_distribution(len(seed_indices))
    else:
        normalized_weights = np.array(seed_weights, dtype=float)
        total = normalized_weights.sum()
        if total <= 0:
            normalized_weights = build_rank_distribution(len(seed_indices))
        else:
            normalized_weights = normalized_weights / total
    for index, weight in zip(seed_indices, normalized_weights):
        if 0 <= index < node_count:
            personalization[index] += float(weight)
    total = personalization.sum()
    if total > 0:
        personalization = personalization / total
    return personalization


def build_transition_matrix(
    nodes: Sequence[NodeLike],
    edges: Sequence[EdgeLike],
    reverse_weights: Mapping[str, float] | None = None,
) -> tuple[np.ndarray, dict[str, int]]:
    node_count = len(nodes)
    transition = np.zeros((node_count, node_count), dtype=float)
    name_to_index = {node.name: index for index, node in enumerate(nodes)}
    if node_count == 0:
        return transition, name_to_index
    reverse_weights = reverse_weights or DEFAULT_REVERSE_WEIGHTS

    for edge in edges:
        source_index = name_to_index.get(edge.source)
        target_index = name_to_index.get(edge.target)
        if source_index is None or target_index is None:
            continue
        forward_weight = max(float(getattr(edge, "weight", 1.0) or 1.0), 0.0)
        if not np.isfinite(forward_weight):
            raise invalid_graph_weight(f"non-finite weight on edge {edge.source!r}->{edge.target!r}")
        transition[source_index, target_index] += forward_weight
        reverse_weight = reverse_weights.get(getattr(edge, "type", "semantic"), 0.0)
        if reverse_weight > 0:
            transition[target_index, source_index] += forward_weight * reverse_weight

    row_sums = transition.sum(axis=1)
    for index in range(node_count):
        if row_sums[index] > 0:
            transition[index, :] = transition[index, :] / row_sums[index]
        else:
            transition[index, index] = 1.0
    return transition, name_to_index


def personalized_pagerank(
    transition: np.ndarray,
    personalization: np.ndarray,
    *,
    damping: float = 0.2,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> np.ndarray:
    if transition.size == 0 or personalization.size == 0:
        return np.array([], dtype=float)
    if not np.all(np.isfinite(transition)):
        raise invalid_graph_weight("transition matrix contains non-finite values")
    scores = personalization.copy()
    for _ in range(max_iter):
        next_scores = damping * personalization + (1.0 - damping) * transition.T.dot(scores)
        if np.linalg.norm(next_scores - scores, ord=1) <= tol:
            scores = next_scores
            break
        scores = next_scores
    total = scores.sum()
    if total > 0:
        scores = scores / total
    return scores


__all__ = [
    "DEFAULT_REVERSE_WEIGHTS",
    "TYPE_WEIGHTS",
    "build_personalization",
    "build_rank_distribution",
    "build_transition_matrix",
    "extract_artifacts",
    "extract_task_name",
    "fallback_query_schema",
    "personalized_pagerank",
    "query_schema_values",
    "rerank_skill_score",
    "retrieve_skills",
    "token_overlap_score",
]


def extract_task_name(query: str) -> str:
    tokens = [token for token in re.split(r"[^a-zA-Z0-9]+", query.strip()) if token]
    if not tokens:
        return ""
    slug = "-".join(token.lower() for token in tokens[:8])
    return slug[:80]


def extract_artifacts(query: str) -> list[str]:
    artifacts = re.findall(r"[A-Za-z0-9_./-]+\.(?:py|md|json|csv|stl|dot|txt|yaml|yml|bib|pptx|xlsx|docx)", query)
    return _dedupe_text([artifact.strip() for artifact in artifacts if artifact.strip()])


def fallback_query_schema(query: str) -> dict[str, list[str] | str]:
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
    values: list[str] = []
    for key in ("goal", "task_name", "domain", "operations", "artifacts", "constraints", "keywords"):
        raw = query_schema.get(key, [])
        if isinstance(raw, str):
            values.append(raw)
        elif isinstance(raw, list):
            values.extend(str(item) for item in raw)
    return [value for value in values if value]


def token_overlap_score(query_tokens: set[str], values: list[str]) -> float:
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
    """Port of the engine's ``_rerank_skill_score`` field-weighted scoring."""

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


def retrieve_skills(
    *,
    nodes: Sequence[Any],
    edges: Sequence[Any],
    vector_seed: Sequence[str] = (),
    vector_similarity: Mapping[str, float] | None = None,
    lexical_seed: Sequence[str] = (),
    query: str = "",
    mode: str = "skill_only",
    has_scenes: bool = False,
    top_k: int = 10,
    damping: float = 0.2,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Deterministic PPR retrieval over an immutable, in-memory snapshot.

    Seed order encodes semantic/lexical rank; ``vector_seed`` precedes
    ``lexical_seed``. A node's 1-based position in ``vector_seed`` is its
    ``semantic_rank`` and earns the engine's ``0.2 / semantic_rank`` rerank
    term. Scene-aware modes degenerate to ``skill_only`` with a warning when
    the snapshot has no Scene nodes. Unknown modes raise
    ``SKILL_MANAGE_UNSUPPORTED_RETRIEVAL_MODE``.
    """

    warnings: list[dict[str, object]] = []
    if mode not in {"skill_only", "unified_ppr", "separate", "fuse"}:
        from .errors import unsupported_retrieval_mode

        raise unsupported_retrieval_mode(mode)
    effective_mode = mode
    if mode != "skill_only" and not has_scenes:
        warnings.append(
            {
                "code": "SKILL_MANAGE_QUERY_REWRITE_DEGRADED",
                "message": f"retrieval mode {mode!r} degenerated to skill_only without Scene nodes",
                "details": {"mode": mode},
            }
        )
        effective_mode = "skill_only"

    if not nodes:
        return [], warnings
    transition, name_to_index = build_transition_matrix(nodes, edges)
    seen: set[str] = set()
    ordered_seeds: list[str] = []
    for seed in [*vector_seed, *lexical_seed]:
        if seed in seen or seed not in name_to_index:
            continue
        seen.add(seed)
        ordered_seeds.append(seed)
    seed_indices = [name_to_index[name] for name in ordered_seeds]
    personalization = build_personalization(len(nodes), seed_indices)
    ppr = personalized_pagerank(transition, personalization, damping=damping, max_iter=max_iter, tol=tol)

    semantic_rank_by_name: dict[str, int] = {}
    for seed in vector_seed:
        if seed in name_to_index and seed not in semantic_rank_by_name:
            semantic_rank_by_name[seed] = len(semantic_rank_by_name) + 1
    similarity = vector_similarity or {}

    query_schema = fallback_query_schema(query) if query else {"goal": "", "task_name": "", "keywords": []}
    hits: list[dict[str, object]] = []
    for node in nodes:
        index = name_to_index[node.name]
        ppr_score = float(ppr[index]) if ppr.size else 0.0
        semantic_rank = semantic_rank_by_name.get(node.name)
        rerank = rerank_skill_score(query_schema, _node_mapping(node), ppr_score, semantic_rank)
        hits.append(
            {
                "name": node.name,
                "graph_node_id": getattr(node, "graph_node_id", None),
                "skill_id": getattr(node, "skill_id", None),
                "version_id": getattr(node, "version_id", None),
                "score": ppr_score,
                "ppr_score": ppr_score,
                "rerank_score": rerank,
                "semantic_rank": semantic_rank,
                "vector_score": similarity.get(node.name),
                "mode": effective_mode,
            }
        )

    hits.sort(
        key=lambda hit: (
            -float(hit["rerank_score"]),
            -float(hit["score"]),
            str(hit.get("version_id") or ""),
            str(hit.get("graph_node_id") or ""),
        )
    )
    return hits[:top_k], warnings


def _node_mapping(node: Any) -> Mapping[str, object]:
    if isinstance(node, Mapping):
        return node
    return {
        "name": getattr(node, "name", ""),
        "description": getattr(node, "description", ""),
        "one_line_capability": getattr(node, "one_line_capability", ""),
        "domain_tags": getattr(node, "domain_tags", ()),
        "tooling": getattr(node, "tooling", ()),
        "inputs": getattr(node, "inputs", ()),
        "outputs": getattr(node, "outputs", ()),
        "example_tasks": getattr(node, "example_tasks", ()),
    }
