"""Independent dense/lexical candidate fusion and typed personalization weights."""

import re

from ....persistence.db_services.skill_management.typing import SkillGraphNodeRecord
from .linking import signature_tokens
from .query import query_values
from .typing import DerivedNodeMetadata, QuerySchema, RetrievalCandidate, RetrievalSeed


def _scripts(node):
    """Use only script paths explicitly present in persisted public fields."""
    text = "\n".join([node.rendered_snippet, *node.tooling, *node.example_tasks])
    metadata = DerivedNodeMetadata.model_validate(node.metadata)
    if metadata.local_scripts:
        return list(metadata.local_scripts)
    return re.findall(r"[A-Za-z0-9_./-]+\.(?:py|sh|bash|js|ts)\b", text)


def node_values(node: SkillGraphNodeRecord) -> list[str]:
    """Collect persisted semantic fields for lexical matching without raw SSL.

    Args:
        node: Validated joint Skill or Scene node.

    Returns:
        Public core and derived text fields, including Scene flow and conditions.
    """
    metadata = DerivedNodeMetadata.model_validate(node.metadata)
    return [
        node.name,
        node.description,
        node.one_line_capability,
        node.rendered_snippet,
        *node.inputs,
        *node.outputs,
        *node.tooling,
        *node.domain_tags,
        *node.example_tasks,
        metadata.intent,
        metadata.intent_signature,
        *metadata.local_scripts,
        *metadata.external_dependencies,
        *metadata.compatibility,
        *metadata.allowed_tools,
        metadata.scene_type,
        metadata.flow_text,
        *metadata.entry_conditions,
        *metadata.exit_conditions,
    ]


def field_score(schema: QuerySchema, node: SkillGraphNodeRecord, graph_score=0.0, rank=None) -> float:
    """Apply reference field weights using only evidence present on the candidate.

    Args:
        schema: Normalized complete public task query.
        node: Candidate Skill or Scene with persisted public fields.
        graph_score: Prior graph evidence, zero for lexical candidate ranking.
        rank: One-based dense rank during candidate fusion, or fused Skill seed
            rank during final reranking; None omits the rank bonus.

    Returns:
        Nonnegative field score including graph evidence and optional rank bonus.
    """
    values = query_values(schema)
    tokens = signature_tokens(values)
    scripts = _scripts(node)
    metadata = DerivedNodeMetadata.model_validate(node.metadata)
    fields = [
        ([node.name], 1.25),
        ([node.one_line_capability, node.description, metadata.intent_signature], 0.9),
        (node.domain_tags, 1.15),
        ([*node.tooling, *metadata.external_dependencies, *metadata.allowed_tools, *metadata.compatibility], 0.95),
        (node.inputs + node.outputs, 0.75),
        (node.example_tasks, 0.8),
        (scripts, 0.6),
    ]
    score = graph_score + sum(
        weight * len(tokens & signature_tokens(texts)) / max(len(tokens), 1) for texts, weight in fields
    )
    # Unlike the historical self-match, a task label needs candidate evidence.
    task_tokens = signature_tokens([schema.task_name]) if schema.task_name else set()
    score += 0.35 * len(task_tokens & signature_tokens(node_values(node))) / max(len(task_tokens), 1)
    name = re.sub(r"[^\w]+", " ", node.name.casefold()).strip()
    text = " " + re.sub(r"[^\w]+", " ", "\n".join(values).casefold()).strip() + " "
    if name and f" {name} " in text:
        score += 1.2
    artifacts = signature_tokens(schema.artifacts)
    script_tokens = signature_tokens(scripts)
    score += 0.9 * len(artifacts & script_tokens) / max(min(len(artifacts), len(script_tokens)), 1)
    if scripts:
        score += 0.08
    if signature_tokens(schema.domain) & signature_tokens(node.domain_tags):
        score += 0.35
    if rank is not None:
        score += 0.2 / rank
    return score


def select_seeds(
    nodes,
    dense_hits,
    schema: QuerySchema,
    *,
    candidates: int,
    seeds: int,
    threshold: float,
    diagnostics: list[RetrievalCandidate] | None = None,
):
    """Fuse one type's independently bounded dense and lexical candidates.

    Args:
        nodes: Only the requested type from one immutable snapshot.
        dense_hits: Database hits for that type and snapshot, with cosine distance.
        schema: Normalized public task query.
        candidates: Independent cap for each candidate route.
        seeds: Maximum retained seeds; zero disables this type.
        threshold: Minimum cosine similarity for dense candidates only.
        diagnostics: Optional collector of safe route ranks before seed truncation.

    Returns:
        Selected seeds in fusion order, with reciprocal-rank weights summing to one.
    """
    if not seeds:
        return []
    by_id = {node.node_id: node for node in nodes}
    distances = {}
    for hit in dense_hits:
        key = hit.record["node_id"]
        if key in by_id and 1 - hit.distance >= threshold:
            distances[key] = min(distances.get(key, float("inf")), hit.distance)
    dense = sorted(distances, key=lambda key: (distances[key], key))[:candidates]
    dense_ranks = {key: rank for rank, key in enumerate(dense, 1)}
    tokens = signature_tokens(query_values(schema))
    lexical = []
    for node in nodes:
        overlap = len(tokens & signature_tokens(node_values(node)))
        if overlap:
            lexical.append((node.node_id, overlap / len(tokens) + field_score(schema, node)))
    lexical.sort(key=lambda item: (-item[1], item[0]))
    lexical_ranks = {key: rank for rank, (key, _) in enumerate(lexical[:candidates], 1)}
    fused = []
    for key in dense_ranks.keys() | lexical_ranks.keys():
        dr, lr = dense_ranks.get(key), lexical_ranks.get(key)
        score = field_score(schema, by_id[key], 1 / dr if dr else 0, dr)
        score += 0.15 / lr if lr else 0
        score += 0.1 if dr and lr else 0
        fused.append((key, score, dr, lr))
    fused.sort(key=lambda item: (-item[1], item[2] is None, item[2] or float("inf"), item[3] or float("inf"), item[0]))
    if diagnostics is not None:
        diagnostics.extend(
            RetrievalCandidate(
                node_id=key,
                node_type=by_id[key].node_type,
                dense_rank=dr,
                lexical_rank=lr,
                fused_rank=rank,
                selected=rank <= seeds,
            )
            for rank, (key, _, dr, lr) in enumerate(fused, 1)
        )
    selected = fused[:seeds]
    total = sum(1 / rank for rank in range(1, len(selected) + 1))
    return [
        RetrievalSeed(
            node_id=key,
            dense_rank=dr,
            lexical_rank=lr,
            fused_rank=rank,
            fusion_score=score,
            weight=1 / rank / total,
        )
        for rank, (key, score, dr, lr) in enumerate(selected, 1)
    ]


def balance_seeds(*groups: list[RetrievalSeed]) -> list[RetrievalSeed]:
    """Give each nonempty node type equal personalization mass.

    Args:
        *groups: Independently normalized Skill and optionally Scene seeds.

    Returns:
        Seeds with explicit combined weights; an only active type receives all mass.
    """
    active = [group for group in groups if group]
    return [seed.model_copy(update={"weight": seed.weight / len(active)}) for group in active for seed in group]
