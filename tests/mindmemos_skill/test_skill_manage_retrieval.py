"""AC-P2-003: retrieval modes, PPR edge cases and deterministic ordering."""

from __future__ import annotations

import pytest
from mindmemos_skill.algos.skill_manage.errors import SkillManageError
from mindmemos_skill.algos.skill_manage.retrieval import retrieve_skills


class Node:
    def __init__(self, name: str) -> None:
        self.name = name
        self.graph_node_id = name
        self.version_id = name
        self.description = name
        self.one_line_capability = name
        self.domain_tags: tuple[str, ...] = ()
        self.tooling: tuple[str, ...] = ()
        self.inputs: tuple[str, ...] = ()
        self.outputs: tuple[str, ...] = ()
        self.example_tasks: tuple[str, ...] = ()


class Edge:
    def __init__(self, source: str, target: str, type: str, weight: float) -> None:
        self.source = source
        self.target = target
        self.type = type
        self.weight = weight


def test_empty_graph_returns_empty_hits() -> None:
    hits, warnings = retrieve_skills(nodes=[], edges=[])
    assert hits == []
    assert warnings == []


def test_single_node_graph_is_stable() -> None:
    hits, _ = retrieve_skills(nodes=[Node("only")], edges=[])
    assert len(hits) == 1
    assert hits[0]["name"] == "only"
    assert float(hits[0]["score"]) >= 0.0


def test_isolated_nodes_do_not_leak_seed_score() -> None:
    nodes = [Node("a"), Node("b")]
    hits, _ = retrieve_skills(nodes=nodes, edges=[], vector_seed=["a"])
    by_name = {hit["name"]: hit for hit in hits}
    assert float(by_name["a"]["score"]) > 0.0
    assert float(by_name["b"]["score"]) == 0.0


def test_unknown_mode_raises_typed_error() -> None:
    with pytest.raises(SkillManageError) as exc_info:
        retrieve_skills(nodes=[Node("a")], edges=[], mode="bogus")
    assert exc_info.value.error_code == "SKILL_MANAGE_UNSUPPORTED_RETRIEVAL_MODE"


def test_scene_aware_modes_degenerate_to_skill_only_without_scenes() -> None:
    hits, warnings = retrieve_skills(
        nodes=[Node("a")],
        edges=[],
        mode="unified_ppr",
        has_scenes=False,
    )
    assert hits[0]["mode"] == "skill_only"
    assert any("degenerated" in str(w["message"]) for w in warnings)


def test_scene_aware_mode_with_scenes_is_not_degenerated() -> None:
    hits, warnings = retrieve_skills(
        nodes=[Node("a")],
        edges=[],
        mode="unified_ppr",
        has_scenes=True,
    )
    assert hits[0]["mode"] == "unified_ppr"
    assert warnings == []


def test_tie_break_order_is_deterministic() -> None:
    nodes = [Node("b"), Node("a"), Node("c")]
    hits, _ = retrieve_skills(nodes=nodes, edges=[], top_k=3)
    assert [hit["name"] for hit in hits] == ["a", "b", "c"]


def test_hits_are_ordered_by_rerank_then_score_like_the_engine() -> None:
    # CSV-matching skill reranks above a higher-PPR but poorly-matching skill,
    # mirroring the engine's final ordering (rerank desc, then score desc).
    nodes = [Node("csv parser"), Node("json merger"), Node("record filter")]
    hits, _ = retrieve_skills(
        nodes=nodes,
        edges=[],
        query="parse a csv file",
        vector_seed=["csv parser", "json merger", "record filter"],
        top_k=3,
    )
    names = [hit["name"] for hit in hits]
    assert names[0] == "csv parser"
    # The ordering must be monotone by (rerank, score) descending.
    scores = [(float(hit["rerank_score"]), float(hit["score"])) for hit in hits]
    assert scores == sorted(scores, reverse=True)
