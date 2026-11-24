"""Pure candidate selection and dependency linking without storage or model I/O."""

from __future__ import annotations

import re
from itertools import combinations
from math import fsum, hypot
from typing import Any, Iterable, Mapping

from ....persistence.db_services.graphs.typing import GraphNodeType, SkillGraphNodeRecord
from .config import JointLinkingConfig
from .constant import (
    METADATA_EXAMPLE_TASKS,
    METADATA_KEY,
    METADATA_TOOLING,
    METADATA_VALUE,
)
from .typing import DerivedNodeMetadata

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
        METADATA_VALUE,
    }
)

RELATION_TYPE_WEIGHTS = {"dependency": 1.0, "workflow": 0.7, "semantic": 0.4, "alternative": 0.3}
"""Source relation-type strengths before applying model confidence."""


def signature_tokens(values: Iterable[str]) -> set[str]:
    """Normalize English feature tokens using the source matching rules.

    Args:
        values: Public field strings used for tokenization or matching.
    """
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
    """Score compatible producer outputs and consumer inputs.

    Args:
        producer_values: Output descriptions from the producing Skill.
        consumer_values: Input descriptions from the consuming Skill.
    """
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
    """Measure token overlap between two field collections.

    Args:
        left_values: Feature strings from the first Skill.
        right_values: Feature strings from the second Skill.
    """
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
    """Return deterministic dependency edges between two Skills, if any.

    Args:
        node: Parsed Skill fields including its stable collection node ID.
        candidate: Another parsed Skill with a distinct node ID.
        dependency_match_threshold: Minimum input/output overlap score required for an edge.
    """

    edges: list[dict[str, Any]] = []
    node_name = node["node_id"]
    candidate_name = candidate["node_id"]
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


def joint_dependency_edges(nodes: list[SkillGraphNodeRecord], *, dependency_match_threshold: float):
    """Yield every positive cross-family I/O dependency independently of model budgets.

    Args:
        nodes: Final Skill and Scene nodes in the explicit snapshot.
        dependency_match_threshold: Minimum positive I/O score accepted as an edge.

    Yields:
        Directed dependencies with their deterministic overlap score and evidence.
    """
    families = {node.node_id: DerivedNodeMetadata.model_validate(node.metadata).family_id for node in nodes}
    values = {node.node_id: {"node_id": node.node_id, "inputs": node.inputs, "outputs": node.outputs} for node in nodes}
    for left, right in combinations(sorted(nodes, key=lambda node: node.node_id), 2):
        if families[left.node_id] == families[right.node_id]:
            continue
        if not ((left.outputs and right.inputs) or (right.outputs and left.inputs)):
            continue
        yield from (
            edge
            for edge in dependency_edges_for_pair(
                values[left.node_id], values[right.node_id], dependency_match_threshold=dependency_match_threshold
            )
            if edge["weight"] > 0
        )


def joint_candidates(
    nodes: list[SkillGraphNodeRecord],
    query_vectors: list[list[float]],
    config: JointLinkingConfig,
    *,
    source_type: GraphNodeType | None = None,
) -> dict[str, list[str]]:
    """Combine bounded dense, lexical and I/O routes within the explicit joint snapshot.

    Args:
        nodes: Final embedded Skill and Scene nodes; no external graph is searched.
        query_vectors: Separate linking-query vectors in node order, or empty when dense is disabled.
        config: Per-route budgets, final cap and dependency threshold.
        source_type: Optional source node type; targets remain in the supplied snapshot.

    Returns:
        Source IDs mapped to ranked cross-family candidate IDs. The selected strategy
        combines route ranks or evidence-backed field scores; ties use opaque node IDs.
    """
    if config.dense_candidates and len(query_vectors) != len(nodes):
        raise ValueError("Each node requires one linking query vector")
    if config.candidate_strategy == "field_evidence":
        return _field_evidence_candidates(nodes, query_vectors, config, source_type=source_type)
    metadata = {node.node_id: DerivedNodeMetadata.model_validate(node.metadata) for node in nodes}
    families = {key: value.family_id for key, value in metadata.items()}
    values = {
        node.node_id: [
            node.name,
            node.description,
            node.one_line_capability,
            metadata[node.node_id].intent,
            metadata[node.node_id].intent_signature,
            *metadata[node.node_id].local_scripts,
            *metadata[node.node_id].external_dependencies,
            *metadata[node.node_id].compatibility,
            *metadata[node.node_id].allowed_tools,
            *node.inputs,
            *node.outputs,
            *node.tooling,
            *node.domain_tags,
            *node.example_tasks,
        ]
        for node in nodes
    }
    result = {}
    for index, source in enumerate(nodes):
        if source_type is not None and source.node_type != source_type:
            continue
        dense, lexical, io = [], [], []
        for target in nodes:
            if families[source.node_id] == families[target.node_id]:
                continue
            if config.dense_candidates:
                query, vector = query_vectors[index], target.embedding
                similarity = fsum(a * b for a, b in zip(query, vector, strict=True)) / (hypot(*query) * hypot(*vector))
                dense.append((similarity, target.node_id))
            if config.lexical_candidates:
                score = shared_field_score(values[source.node_id], values[target.node_id])
                if score > 0:
                    lexical.append((score, target.node_id))
            if config.io_candidates:
                score = max(
                    schema_overlap_score(source.outputs, target.inputs)[0],
                    schema_overlap_score(target.outputs, source.inputs)[0],
                )
                if score > 0:
                    io.append((score, target.node_id))
        scores = {}
        for route, budget in (
            (dense, config.dense_candidates),
            (lexical, config.lexical_candidates),
            (io, config.io_candidates),
        ):
            for rank, (_, target_id) in enumerate(sorted(route, key=lambda item: (-item[0], item[1]))[:budget], 1):
                scores[target_id] = scores.get(target_id, 0.0) + 1.0 / rank
        result[source.node_id] = sorted(scores, key=lambda key: (-scores[key], key))[: config.max_candidates]
    return result


def joint_relation_payload(node: SkillGraphNodeRecord) -> dict[str, Any]:
    """Expose only public node features needed for relation inference.

    Args:
        node: Final source-grounded Skill or Scene node selected for a model pair.

    Returns:
        Relation-model fields, excluding vectors and any unvalidated metadata.
    """
    metadata = DerivedNodeMetadata.model_validate(node.metadata)
    return {
        "node_id": node.node_id,
        "node_type": node.node_type,
        "name": node.name,
        "description": node.description,
        "one_line_capability": node.one_line_capability,
        "domain_tags": node.domain_tags,
        METADATA_EXAMPLE_TASKS: node.example_tasks,
        "inputs": node.inputs,
        "outputs": node.outputs,
        METADATA_TOOLING: node.tooling,
        METADATA_KEY: metadata.model_dump(mode="json"),
    }


def _pair_feature_score(source, target, source_metadata, target_metadata):
    """Apply the reference field-evidence thresholds without using dense similarity as evidence."""
    score = 0.0
    for left, right, weight in (
        (source.domain_tags, target.domain_tags, 1.0),
        (source.tooling, target.tooling, 0.75),
        (source.example_tasks, target.example_tasks, 0.6),
        (source_metadata.local_scripts, target_metadata.local_scripts, 0.35),
        (source.inputs + source.outputs, target.inputs + target.outputs, 0.4),
    ):
        overlap = shared_field_score(left, right)
        if overlap >= 0.5:
            score += weight * overlap
    io_score = max(
        schema_overlap_score(source.outputs, target.inputs)[0],
        schema_overlap_score(target.outputs, source.inputs)[0],
    )
    return score + 0.85 * io_score, io_score


def _candidate_field_score(query_tokens, target, metadata):
    """Weight evidence in individual target fields using the reference candidate recipe."""
    score = 0.0
    for values, weight in (
        ([target.name], 1.05),
        ([target.one_line_capability, target.description], 0.8),
        (target.domain_tags, 1.0),
        (target.tooling, 0.85),
        (target.inputs + target.outputs, 0.55),
        (target.example_tasks, 0.7),
        (metadata.local_scripts, 0.45),
    ):
        score += weight * len(query_tokens & signature_tokens(values)) / max(len(query_tokens), 1)
    return score


def _field_evidence_candidates(nodes, query_vectors, config, *, source_type):
    """Retain grounded field matches and combine route scores by their maximum."""
    metadata = {node.node_id: DerivedNodeMetadata.model_validate(node.metadata) for node in nodes}
    result = {}
    for index, source in enumerate(nodes):
        if source_type is not None and source.node_type != source_type:
            continue
        source_metadata = metadata[source.node_id]
        query_tokens = signature_tokens(
            [
                source.name,
                source.one_line_capability,
                source.description,
                source_metadata.intent,
                source_metadata.intent_signature,
                *source.domain_tags,
                *source.tooling,
                *source.example_tasks,
                *source.inputs,
                *source.outputs,
                *source_metadata.local_scripts,
                *source_metadata.external_dependencies,
                *source_metadata.allowed_tools,
                *source_metadata.compatibility,
            ]
        )
        dense, lexical, io = [], [], []
        field_scores = {}
        for target in nodes:
            target_metadata = metadata[target.node_id]
            if source_metadata.family_id == target_metadata.family_id:
                continue
            if config.dense_candidates:
                query, vector = query_vectors[index], target.embedding
                similarity = fsum(a * b for a, b in zip(query, vector, strict=True)) / (hypot(*query) * hypot(*vector))
                dense.append((similarity, target.node_id))
            pair_score, io_score = _pair_feature_score(source, target, source_metadata, target_metadata)
            if pair_score <= 0:
                continue
            score = pair_score + _candidate_field_score(query_tokens, target, target_metadata)
            field_scores[target.node_id] = score
            lexical.append((score, target.node_id))
            if io_score > 0:
                io.append((score, target.node_id))
        scores = {}
        for route, budget, dense_route in (
            (dense, config.dense_candidates, True),
            (lexical, config.lexical_candidates, False),
            (io, config.io_candidates, False),
        ):
            ranked = sorted(route, key=lambda item: (-item[0], item[1]))[:budget]
            for rank, (_, target_id) in enumerate(ranked, 1):
                if target_id not in field_scores:
                    continue
                score = field_scores[target_id] + (1.2 / rank if dense_route else 0.0)
                scores[target_id] = max(score, scores.get(target_id, 0.0))
        result[source.node_id] = sorted(scores, key=lambda key: (-scores[key], key))[: config.max_candidates]
    return result


def joint_model_candidates(
    nodes: list[SkillGraphNodeRecord],
    query_vectors: list[list[float]],
    config: JointLinkingConfig,
    *,
    skill_linking: bool,
    scene_linking: bool,
    dependency_pairs: set[frozenset[str]],
) -> dict[str, list[str]]:
    """Select type-scoped model candidates without re-inferring deterministic dependencies.

    Args:
        nodes: Final Skill and Scene nodes in the explicit snapshot.
        query_vectors: Linking vectors in node order, or empty when dense is disabled.
        config: Route budgets and the per-source inference limit.
        skill_linking: Select Skill-to-Skill candidates when enabled.
        scene_linking: Select Scene-to-joint candidates when enabled.
        dependency_pairs: Unordered pairs already connected by deterministic dependencies.

    Returns:
        Ranked inference targets per source, capped after excluding dependency pairs.
    """
    candidates = {}
    if skill_linking:
        skills = [node for node in nodes if node.node_type == GraphNodeType.SKILL]
        vectors = (
            [vector for node, vector in zip(nodes, query_vectors, strict=True) if node.node_type == GraphNodeType.SKILL]
            if config.dense_candidates
            else []
        )
        candidates.update(joint_candidates(skills, vectors, config))
    if scene_linking:
        candidates.update(joint_candidates(nodes, query_vectors, config, source_type=GraphNodeType.SCENE))
    return {
        source: [target for target in targets if frozenset((source, target)) not in dependency_pairs][
            : config.model_candidates
        ]
        for source, targets in candidates.items()
    }


def joint_candidate_pairs(nodes: list[SkillGraphNodeRecord], candidates: dict[str, list[str]]):
    """Yield each selected unordered pair once in stable ID order.

    Args:
        nodes: Final snapshot nodes.
        candidates: Bounded per-source candidate lists.

    Yields:
        Node pairs for optional model inference.
    """
    by_id = {node.node_id: node for node in nodes}
    pairs = {tuple(sorted((source, target))) for source, targets in candidates.items() for target in targets}
    for left, right in sorted(pairs):
        yield by_id[left], by_id[right]


__all__ = [
    "TOKEN_STOPWORDS",
    "dependency_edges_for_pair",
    "schema_overlap_score",
    "shared_field_score",
    "signature_tokens",
]
