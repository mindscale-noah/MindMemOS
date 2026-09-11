"""Deterministic Skill linking (dependency edges) without storage or LLM I/O.

Ported from the GOS engine's tokenization, schema-overlap and
``_dependency_edges_for_pair`` logic. Deterministic edges are pure functions of
the parsed Skills; optional LLM relation validation is applied separately by
the algorithm and degrades to these deterministic edges on any failure.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping

TOKEN_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "any",
        "arg",
        "args",
        "array",
        "bool",
        "boolean",
        "data",
        "dict",
        "file",
        "float",
        "for",
        "from",
        "in",
        "input",
        "int",
        "json",
        "list",
        "object",
        "of",
        "on",
        "or",
        "output",
        "path",
        "record",
        "result",
        "set",
        "str",
        "string",
        "text",
        "the",
        "to",
        "value",
    }
)


def signature_tokens(values: Iterable[str]) -> set[str]:
    tokens: set[str] = set()
    for value in values:
        lowered = str(value).lower()
        normalized = re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")
        if normalized:
            tokens.add(normalized)
        for token in re.findall(r"[a-z0-9]+", lowered):
            token = token.rstrip("s")
            if len(token) < 3 or token in TOKEN_STOPWORDS:
                continue
            tokens.add(token)
    return tokens


def schema_overlap_score(producer_values: list[str], consumer_values: list[str]) -> tuple[float, list[str]]:
    best_score = 0.0
    best_evidence: set[str] = set()
    for producer in producer_values:
        producer_norm = re.sub(r"[^a-z0-9]+", "_", str(producer).lower()).strip("_")
        producer_tokens = signature_tokens([producer])
        for consumer in consumer_values:
            consumer_norm = re.sub(r"[^a-z0-9]+", "_", str(consumer).lower()).strip("_")
            consumer_tokens = signature_tokens([consumer])
            if producer_norm and producer_norm == consumer_norm:
                return 1.0, [producer_norm]
            if producer_norm and consumer_norm and (producer_norm in consumer_norm or consumer_norm in producer_norm):
                return 0.85, sorted({producer_norm, consumer_norm})
            overlap = producer_tokens & consumer_tokens
            if not overlap:
                continue
            union = producer_tokens | consumer_tokens
            score = max(0.5, len(overlap) / max(len(union), 1))
            if score > best_score:
                best_score = score
                best_evidence = overlap
    return best_score, sorted(best_evidence)


def shared_field_score(left_values: list[str], right_values: list[str]) -> float:
    left_tokens = signature_tokens(left_values)
    right_tokens = signature_tokens(right_values)
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = left_tokens & right_tokens
    if not overlap:
        return 0.0
    return len(overlap) / max(min(len(left_tokens), len(right_tokens)), 1)


def dependency_edges_for_pair(
    node: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    dependency_match_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """Return deterministic dependency edges between two Skills, if any."""

    edges: list[dict[str, Any]] = []
    node_name = node["name"]
    candidate_name = candidate["name"]
    node_inputs = list(node.get("inputs") or [])
    node_outputs = list(node.get("outputs") or [])
    candidate_inputs = list(candidate.get("inputs") or [])
    candidate_outputs = list(candidate.get("outputs") or [])

    forward_score, forward_evidence = schema_overlap_score(node_outputs, candidate_inputs)
    if forward_score >= dependency_match_threshold:
        evidence = ", ".join(forward_evidence) or "compatible I/O"
        edges.append(
            {
                "source": node_name,
                "target": candidate_name,
                "description": f"{node_name} produces data that {candidate_name} consumes: {evidence}.",
                "type": "dependency",
                "weight": forward_score,
            }
        )

    reverse_score, reverse_evidence = schema_overlap_score(candidate_outputs, node_inputs)
    if reverse_score >= dependency_match_threshold:
        evidence = ", ".join(reverse_evidence) or "compatible I/O"
        edges.append(
            {
                "source": candidate_name,
                "target": node_name,
                "description": f"{candidate_name} produces data that {node_name} consumes: {evidence}.",
                "type": "dependency",
                "weight": reverse_score,
            }
        )
    return edges


def build_deterministic_dependency_edges(
    skills: list[Mapping[str, Any]],
    *,
    dependency_match_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """Full linking pass producing the deterministic dependency edge set.

    The candidate set is built from shared signature tokens between one Skill's
    outputs and another's inputs, then each pair is scored bidirectionally and
    deduped by the stable ``(source, target, type)`` key.
    """

    skill_nodes = [skill for skill in skills if skill.get("node_type", "skill") == "skill"]
    output_index: dict[str, set[int]] = {}
    input_index: dict[str, set[int]] = {}
    for index, skill in enumerate(skill_nodes):
        for token in signature_tokens(skill.get("inputs") or []):
            input_index.setdefault(token, set()).add(index)
        for token in signature_tokens(skill.get("outputs") or []):
            output_index.setdefault(token, set()).add(index)

    edge_map: dict[tuple[str, str, str], dict[str, Any]] = {}
    for index, node in enumerate(skill_nodes):
        candidate_indices: set[int] = set()
        for token in signature_tokens(node.get("inputs") or []):
            candidate_indices.update(output_index.get(token, set()))
        for token in signature_tokens(node.get("outputs") or []):
            candidate_indices.update(input_index.get(token, set()))
        candidate_indices.discard(index)
        for candidate_index in sorted(candidate_indices):
            candidate = skill_nodes[candidate_index]
            for edge in dependency_edges_for_pair(
                node,
                candidate,
                dependency_match_threshold=dependency_match_threshold,
            ):
                key = (edge["source"], edge["target"], edge["type"])
                edge_map[key] = edge
    return [edge_map[key] for key in sorted(edge_map)]


__all__ = [
    "TOKEN_STOPWORDS",
    "build_deterministic_dependency_edges",
    "dependency_edges_for_pair",
    "schema_overlap_score",
    "shared_field_score",
    "signature_tokens",
]
