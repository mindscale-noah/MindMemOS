"""Dependency ablations over an immutable query snapshot."""

import math
import re
from collections import Counter, defaultdict

from .linking import signature_tokens


def _evidence(edge):
    """Read only deterministic linker evidence, retaining unknown formats conservatively."""
    match = re.search(r"consumes:\s*(.+?)\.?\s*$", edge.metadata.get("evidence", ""))
    if not match or match[1].casefold() == "compatible i/o":
        return []
    return [value.strip() for value in match[1].split(",") if value.strip()]


def _path(edge, evidence):
    """Identify the deterministic exact, substring, or shared-token match path."""
    if edge.weight >= 0.999 and len(evidence) == 1:
        return "exact"
    return "substring" if abs(edge.weight - 0.85) < 1e-6 else "jaccard"


def filter_edges(nodes, edges, config):
    """Return a filtered/reweighted list without changing records or storage.

    Args:
        nodes: Nodes from one pinned joint snapshot.
        edges: Directed edges from that snapshot.
        config: Validated retrieval mode and dependency filter settings.

    Returns:
        Query-view edges; non-dependencies and unparseable evidence are retained.
        The top-k cap uses the union of both Scene endpoints' choices.
    """
    scenes = {node.node_id for node in nodes if node.node_type == "scene"}
    mode = config.dependency_filter
    if config.mode == "skill_only" or not scenes or mode == "none":
        return list(edges)
    dependencies = [edge for edge in edges if edge.edge_type == "dependency"]
    if not dependencies:
        return list(edges)
    frequency = Counter(token for node in nodes for token in signature_tokens(node.inputs + node.outputs))
    evidence = {edge.edge_id: _evidence(edge) for edge in dependencies}
    paths = {edge.edge_id: _path(edge, evidence[edge.edge_id]) for edge in dependencies}
    drivers = {
        edge.edge_id: [min(values, key=len)] if paths[edge.edge_id] == "substring" and values else values
        for edge in dependencies
        for values in [evidence[edge.edge_id]]
    }
    yields = Counter(
        drivers[edge.edge_id][0] for edge in dependencies if drivers[edge.edge_id] and paths[edge.edge_id] != "jaccard"
    )
    dropped, factors = set(), {}
    for edge in dependencies:
        values = drivers[edge.edge_id]
        if not values:
            continue
        if mode in ("hub_df", "hub_df_topk"):
            hub = all(frequency[value] >= config.dependency_hub_fraction * len(nodes) for value in values)
            if paths[edge.edge_id] != "jaccard":
                hub |= yields[values[0]] >= config.dependency_hub_fraction * len(dependencies)
            if hub:
                dropped.add(edge.edge_id)
        elif mode == "no_substring" and paths[edge.edge_id] == "substring":
            dropped.add(edge.edge_id)
        elif mode == "idf_reweight":
            effective = max(max(frequency[value], yields[value] * len(nodes) / len(dependencies)) for value in values)
            factors[edge.edge_id] = max(0.0, math.log((len(nodes) + 1) / (1 + effective)) / math.log(len(nodes) + 1))
    if mode in ("topk", "hub_df_topk"):
        incident = defaultdict(list)
        for edge in dependencies:
            if edge.edge_id not in dropped and evidence[edge.edge_id]:
                for endpoint in (edge.source_node_id, edge.target_node_id):
                    if endpoint in scenes:
                        incident[endpoint].append(edge)

        def specificity(edge):
            """Break equal weights by evidence specificity and stable endpoint IDs."""
            values = evidence[edge.edge_id]
            return (
                -edge.weight,
                min(frequency[value] for value in values),
                -max(len(value) for value in values),
                edge.source_node_id,
                edge.target_node_id,
            )

        kept = {
            edge.edge_id
            for group in incident.values()
            for edge in sorted(group, key=specificity)[: config.dependency_top_k]
        }
        dropped.update(edge.edge_id for group in incident.values() for edge in group if edge.edge_id not in kept)
    return [
        edge.model_copy(update={"weight": edge.weight * factors[edge.edge_id]}) if edge.edge_id in factors else edge
        for edge in edges
        if edge.edge_id not in dropped
    ]
