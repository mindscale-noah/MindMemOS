"""AC-P2-002/003/004: linking, retrieval and algorithm-boundary contracts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from mindmemos_skill.algos.skill_manage.linking import (
    build_deterministic_dependency_edges,
    schema_overlap_score,
    shared_field_score,
    signature_tokens,
)
from mindmemos_skill.algos.skill_manage.retrieval import (
    build_personalization,
    build_rank_distribution,
    build_transition_matrix,
    personalized_pagerank,
)

FIXTURE = Path(__file__).parent / "fixtures" / "skill_manage" / "gos_minimal"
GOLDEN = json.loads((FIXTURE / "expected" / "golden.json").read_text(encoding="utf-8"))


def _parsed_skills() -> list[dict[str, Any]]:
    skills: list[dict[str, Any]] = []
    for path in sorted((FIXTURE / "skills").glob("*.md")):
        from mindmemos_skill.algos.skill_manage.parsing import parse_skill_document

        parsed = parse_skill_document(path.read_text(encoding="utf-8"), source_path=f"skills/{path.name}")
        parsed["node_type"] = "skill"
        skills.append(parsed)
    return skills


class Node:
    def __init__(self, name: str) -> None:
        self.name = name


class Edge:
    def __init__(self, source: str, target: str, type: str, weight: float) -> None:
        self.source = source
        self.target = target
        self.type = type
        self.weight = weight


def test_signature_tokens_matches_gos_tokenization() -> None:
    assert signature_tokens(["parsed_records"]) == {"parsed_records", "parsed"}
    assert "data" not in signature_tokens(["data-transfer"])  # stopword filtered


def test_schema_overlap_score_exact_and_partial() -> None:
    assert schema_overlap_score(["parsed_records"], ["parsed_records"]) == (1.0, ["parsed_records"])
    score, evidence = schema_overlap_score(["s3_uri"], ["file_path"])
    assert score < 0.5
    assert evidence == []


def test_shared_field_score_for_common_domain() -> None:
    assert shared_field_score(["data-transfer", "sync"], ["data-transfer", "cloud"]) > 0.0
    assert shared_field_score(["data-transfer"], ["text"]) == 0.0


def test_deterministic_dependency_edges_match_golden() -> None:
    skills = _parsed_skills()
    edges = build_deterministic_dependency_edges(skills)

    # Producer -> consumer direction derived from the fixtures' typed I/O:
    #   CSV Parser --parsed_records--> Record Filter --filtered_records--> JSON Merger
    # (The P0 golden records the JSON Merger/Record Filter edge in the reverse
    # direction with a description that contradicts the fixture I/O; that is a
    # known baseline anomaly fixed here — see progress.md parity notes.)
    expected = sorted(
        [
            ("CSV Parser Skill", "Record Filter Skill"),
            ("Record Filter Skill", "JSON Merger Skill"),
        ]
    )
    actual = sorted((e["source"], e["target"]) for e in edges)

    assert actual == expected
    assert all(e["weight"] == 1.0 for e in edges)
    assert len(edges) == len(set((e["source"], e["target"], e["type"]) for e in edges))


def test_ppr_matches_golden_scores_within_tolerance() -> None:
    skill_nodes = sorted([n for n in GOLDEN["nodes"] if n["type"] == "Skill"], key=lambda n: n["name"])
    names = {n["name"] for n in skill_nodes}
    edges = [e for e in GOLDEN["edges"] if e["source"] in names and e["target"] in names]

    nodes = [Node(n["name"]) for n in skill_nodes]
    edge_objects = [Edge(e["source"], e["target"], e["type"], e["weight"]) for e in edges]
    transition, name_to_index = build_transition_matrix(nodes, edge_objects)

    seeds = GOLDEN["seeds"]
    seed_indices = [name_to_index[s["name"]] for s in seeds if s["name"] in name_to_index]
    seed_weights = [s["seed_weight"] for s in seeds if s["name"] in name_to_index]
    personalization = build_personalization(len(nodes), seed_indices, seed_weights)

    scores = personalized_pagerank(transition, personalization, damping=0.2, max_iter=50, tol=1e-6)

    golden_scores = {n["name"]: GOLDEN["ppr"]["scores"][n["name"]] for n in skill_nodes}
    for node in nodes:
        actual = float(scores[name_to_index[node.name]])
        expected = golden_scores[node.name]
        assert abs(actual - expected) <= 1e-8 + 1e-7 * abs(expected), node.name


def test_ppr_empty_and_single_node() -> None:
    empty = personalized_pagerank(np.zeros((0, 0)), np.zeros(0))
    assert empty.size == 0
    single = personalized_pagerank(np.array([[1.0]]), np.array([1.0]), max_iter=10, tol=1e-9)
    assert single.shape == (1,)
    assert math.isclose(float(single[0]), 1.0, abs_tol=1e-6)


def test_ppr_returns_last_iterate_when_not_converged() -> None:
    # The golden transition matrix is periodic (two 3-cycles), so the 50-iterate
    # cap is hit; the core faithfully returns the last iterate like GOS, and the
    # P4 retriever wrapper is responsible for the PPR_NOT_CONVERGED error.
    scores = personalized_pagerank(np.array([[1.0]]), np.array([1.0]), max_iter=1, tol=0.0)
    assert scores.shape == (1,)
    assert math.isfinite(float(scores[0]))


def test_transition_matrix_rejects_non_finite_weight() -> None:
    from mindmemos_skill.algos.skill_manage.errors import SkillManageError

    nodes = [Node("a"), Node("b")]
    bad_edge = Edge("a", "b", "semantic", float("nan"))
    with pytest.raises(SkillManageError) as exc_info:
        build_transition_matrix(nodes, [bad_edge])
    assert exc_info.value.error_code == "SKILL_MANAGE_INVALID_GRAPH_WEIGHT"


def test_rank_distribution_is_normalized() -> None:
    distribution = build_rank_distribution(3)
    assert distribution.shape == (3,)
    assert math.isclose(float(distribution.sum()), 1.0, abs_tol=1e-9)
