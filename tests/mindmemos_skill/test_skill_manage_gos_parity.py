"""AC-P2-005: parity with the frozen GOS golden baseline."""

from __future__ import annotations

import json
from pathlib import Path

from mindmemos_skill.algos.skill_manage.linking import build_deterministic_dependency_edges
from mindmemos_skill.algos.skill_manage.parsing import parse_skill_document
from mindmemos_skill.algos.skill_manage.retrieval import (
    build_personalization,
    build_transition_matrix,
    personalized_pagerank,
)

FIXTURE = Path(__file__).parent / "fixtures" / "skill_manage" / "gos_minimal"
GOLDEN = json.loads((FIXTURE / "expected" / "golden.json").read_text(encoding="utf-8"))


def _golden_parsed_by_name() -> dict[str, dict]:
    return {entry["name"]: entry for entry in GOLDEN["parsed"]}


def test_parsed_fields_match_golden() -> None:
    golden_by_name = _golden_parsed_by_name()
    for path in sorted((FIXTURE / "skills").glob("*.md")):
        parsed = parse_skill_document(path.read_text(encoding="utf-8"), source_path=f"skills/{path.name}")
        golden = golden_by_name[parsed["name"]]
        assert parsed["description"] == golden["description"]
        assert parsed["inputs"] == golden["inputs"]
        assert parsed["outputs"] == golden["outputs"]
        assert parsed["domain_tags"] == golden["domain_tags"]
        assert parsed["source_path"] == golden["source_path"]


def test_deterministic_edge_set_matches_golden_modulo_known_anomaly() -> None:
    skills = []
    for path in sorted((FIXTURE / "skills").glob("*.md")):
        parsed = parse_skill_document(path.read_text(encoding="utf-8"), source_path=f"skills/{path.name}")
        parsed["node_type"] = "skill"
        skills.append(parsed)
    actual = sorted((e["source"], e["target"]) for e in build_deterministic_dependency_edges(skills))

    # The P0 golden records JSON Merger -> Record Filter, which contradicts the
    # fixture I/O (Record Filter produces filtered_records, JSON Merger consumes
    # it). The corrected direction is asserted below and documented in the
    # parity notes.
    assert ("CSV Parser Skill", "Record Filter Skill") in actual
    assert ("Record Filter Skill", "JSON Merger Skill") in actual
    assert len(actual) == 2


def test_ppr_scores_match_golden_within_tolerance() -> None:
    skill_nodes = sorted([n for n in GOLDEN["nodes"] if n["type"] == "Skill"], key=lambda n: n["name"])
    names = {n["name"] for n in skill_nodes}
    edges = [e for e in GOLDEN["edges"] if e["source"] in names and e["target"] in names]

    class Node:
        def __init__(self, name: str) -> None:
            self.name = name

    class Edge:
        def __init__(self, source: str, target: str, type: str, weight: float) -> None:
            self.source = source
            self.target = target
            self.type = type
            self.weight = weight

    nodes = [Node(n["name"]) for n in skill_nodes]
    transition, name_to_index = build_transition_matrix(
        nodes,
        [Edge(e["source"], e["target"], e["type"], e["weight"]) for e in edges],
    )
    seeds = GOLDEN["seeds"]
    personalization = build_personalization(
        len(nodes),
        [name_to_index[s["name"]] for s in seeds if s["name"] in name_to_index],
        [s["seed_weight"] for s in seeds if s["name"] in name_to_index],
    )
    scores = personalized_pagerank(transition, personalization, damping=0.2, max_iter=50, tol=1e-6)

    golden_scores = {n["name"]: GOLDEN["ppr"]["scores"][n["name"]] for n in skill_nodes}
    for node in nodes:
        actual = float(scores[name_to_index[node.name]])
        expected = golden_scores[node.name]
        assert abs(actual - expected) <= 1e-8 + 1e-7 * abs(expected), node.name
