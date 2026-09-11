"""Vector-seeded retrieval: semantic rank, vector score and GOS golden parity."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from mindmemos_skill.algos.skill_manage.algorithm import DefaultSkillManageAlgorithm
from mindmemos_skill.algos.skill_manage.config import SkillManageConfig
from mindmemos_skill.algos.skill_manage.retrieval import (
    fallback_query_schema,
    rerank_skill_score,
    retrieve_skills,
)
from mindmemos_skill.typing import (
    SkillGraphEdge,
    SkillGraphNode,
    SkillGraphSnapshot,
    SkillManageRetrieveInput,
    SkillManageScope,
)

FIXTURE = Path(__file__).parent / "fixtures" / "skill_manage" / "gos_minimal"
GOLDEN = json.loads((FIXTURE / "expected" / "golden.json").read_text(encoding="utf-8"))


class Node:
    def __init__(
        self,
        name: str,
        *,
        skill_id: str | None = None,
        description: str = "",
        one_line_capability: str = "",
        inputs: tuple[str, ...] = (),
        outputs: tuple[str, ...] = (),
        domain_tags: tuple[str, ...] = (),
        tooling: tuple[str, ...] = (),
    ) -> None:
        self.name = name
        self.graph_node_id = f"skill:{name}"
        self.skill_id = skill_id
        self.version_id = f"{name}@v1"
        self.description = description
        self.one_line_capability = one_line_capability
        self.inputs = inputs
        self.outputs = outputs
        self.domain_tags = domain_tags
        self.tooling = tooling
        self.example_tasks: tuple[str, ...] = ()


class Edge:
    def __init__(self, source: str, target: str, type: str, weight: float) -> None:
        self.source = source
        self.target = target
        self.type = type
        self.weight = weight


def _mapping(node: Node) -> dict[str, object]:
    return {
        "name": node.name,
        "description": node.description,
        "one_line_capability": node.one_line_capability,
        "domain_tags": node.domain_tags,
        "tooling": node.tooling,
        "inputs": node.inputs,
        "outputs": node.outputs,
        "example_tasks": node.example_tasks,
    }


def _by_name(hits: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    return {str(hit["name"]): hit for hit in hits}


def _close(actual: float, expected: float) -> bool:
    return abs(actual - expected) <= 1e-8 + 1e-7 * abs(expected)


def test_hits_carry_skill_id_vector_score_and_semantic_rank() -> None:
    nodes = [Node("a", skill_id="sk-a"), Node("b", skill_id="sk-b"), Node("c", skill_id="sk-c")]
    hits, _ = retrieve_skills(
        nodes=nodes,
        edges=[],
        vector_seed=["b", "unknown", "a", "b"],
        vector_similarity={"b": 0.9, "a": 0.7, "unknown": 0.5},
        top_k=3,
    )
    by_name = _by_name(hits)

    assert by_name["a"]["skill_id"] == "sk-a"
    assert by_name["b"]["skill_id"] == "sk-b"
    assert by_name["c"]["skill_id"] == "sk-c"
    assert by_name["b"]["semantic_rank"] == 1
    assert by_name["a"]["semantic_rank"] == 2
    assert by_name["c"]["semantic_rank"] is None
    assert by_name["b"]["vector_score"] == 0.9
    assert by_name["a"]["vector_score"] == 0.7
    assert by_name["c"]["vector_score"] is None
    assert {hit["name"] for hit in hits} == {"a", "b", "c"}


def test_vector_score_is_none_without_similarity_map() -> None:
    hits, _ = retrieve_skills(nodes=[Node("a", skill_id="sk-a")], edges=[], vector_seed=["a"])
    assert hits[0]["semantic_rank"] == 1
    assert hits[0]["vector_score"] is None


def test_seeded_rerank_adds_semantic_rank_bonus_only_for_seeds() -> None:
    query = "parse a csv file"
    nodes = [
        Node("csv parser", skill_id="csv", description="parse csv text"),
        Node("json merger", skill_id="json", description="merge json documents"),
        Node("record filter", skill_id="filter", description="filter records"),
    ]
    edges = [Edge("csv parser", "record filter", "dependency", 1.0)]
    hits, _ = retrieve_skills(
        nodes=nodes,
        edges=edges,
        vector_seed=["json merger", "csv parser"],
        query=query,
        top_k=3,
    )
    by_name = _by_name(hits)
    schema = fallback_query_schema(query)
    ranks = {"json merger": 1, "csv parser": 2}

    for node in nodes:
        hit = by_name[node.name]
        base = rerank_skill_score(schema, _mapping(node), float(hit["ppr_score"]), None)
        rank = ranks.get(node.name)
        if rank is None:
            assert hit["semantic_rank"] is None
            assert float(hit["rerank_score"]) == pytest.approx(base)
        else:
            assert hit["semantic_rank"] == rank
            assert float(hit["rerank_score"]) == pytest.approx(base + 0.2 / rank)


def test_golden_top3_rerank_and_ppr_match_gos_baseline() -> None:
    skill_entries = sorted((n for n in GOLDEN["nodes"] if n["type"] == "Skill"), key=lambda n: n["name"])
    names = {entry["name"] for entry in skill_entries}
    nodes = [
        Node(
            entry["name"],
            skill_id=entry["name"],
            description=entry["description"],
            one_line_capability=entry["one_line_capability"],
            inputs=tuple(entry["inputs"]),
            outputs=tuple(entry["outputs"]),
            domain_tags=tuple(entry["domain_tags"]),
            tooling=tuple(entry["tooling"]),
        )
        for entry in skill_entries
    ]
    edges = [
        Edge(e["source"], e["target"], e["type"], e["weight"])
        for e in GOLDEN["edges"]
        if e["source"] in names and e["target"] in names
    ]
    seeds = [seed["name"] for seed in GOLDEN["seeds"]]

    hits, warnings = retrieve_skills(
        nodes=nodes,
        edges=edges,
        vector_seed=seeds,
        query=GOLDEN["query"],
        top_k=len(nodes),
    )

    assert warnings == []
    assert len(hits) == len(nodes)
    golden_top = GOLDEN["top_k"][:3]
    assert [hit["name"] for hit in hits[:3]] == [entry["name"] for entry in golden_top]
    assert [entry["name"] for entry in golden_top] == ["CSV Parser Skill", "JSON Merger Skill", "Record Filter Skill"]
    for hit, expected in zip(hits[:3], golden_top, strict=True):
        assert _close(float(hit["rerank_score"]), expected["rerank_score"]), (
            hit["name"],
            hit["rerank_score"],
            expected["rerank_score"],
        )
        assert _close(float(hit["score"]), expected["score"]), (hit["name"], hit["score"], expected["score"])

    by_name = _by_name(hits)
    for seed in GOLDEN["seeds"]:
        assert by_name[seed["name"]]["semantic_rank"] == seed["semantic_rank"]
    for name, expected_ppr in GOLDEN["ppr"]["scores"].items():
        actual = float(by_name[name]["score"])
        assert _close(actual, expected_ppr), (name, actual, expected_ppr)


def _snapshot() -> SkillGraphSnapshot:
    nodes = (
        SkillGraphNode(graph_node_id="n-csv", node_type="skill", skill_id="csv-family", version_id="v7", name="csv"),
        SkillGraphNode(
            graph_node_id="n-filter", node_type="skill", skill_id="filter-family", version_id="v3", name="filter"
        ),
        SkillGraphNode(graph_node_id="n-orphan", node_type="skill", skill_id=None, version_id="v1", name="orphan"),
    )
    edges = (SkillGraphEdge(edge_id="e1", edge_type="dependency", source_id="n-csv", target_id="n-filter", weight=1.0),)
    return SkillGraphSnapshot(generation_id="gen-1", revision=4, nodes=nodes, edges=edges)


@pytest.mark.asyncio
async def test_algorithm_retrieve_maps_skill_id_and_vector_score() -> None:
    algorithm = DefaultSkillManageAlgorithm(config=SkillManageConfig(), context=object())
    output = await algorithm.retrieve(
        SkillManageRetrieveInput(
            scope=SkillManageScope(project_id="p", graph_id="g"),
            snapshot=_snapshot(),
            query="filter csv records",
            top_k=3,
            vector_seed=("filter", "csv"),
            vector_similarity={"filter": 0.91, "csv": 0.42},
        )
    )

    assert output.generation_id == "gen-1"
    assert output.revision == 4
    by_name = {hit.name: hit for hit in output.hits}
    assert set(by_name) == {"csv", "filter", "orphan"}

    assert by_name["csv"].skill_id == "csv-family"
    assert by_name["csv"].version_id == "v7"
    assert by_name["csv"].graph_node_id == "n-csv"
    assert by_name["filter"].skill_id == "filter-family"
    assert by_name["filter"].version_id == "v3"
    # No skill_id on the node: the hit falls back to the node name, never the version.
    assert by_name["orphan"].skill_id == "orphan"
    assert by_name["orphan"].version_id == "v1"

    assert by_name["filter"].vector_score == 0.91
    assert by_name["csv"].vector_score == 0.42
    assert by_name["orphan"].vector_score is None
    assert by_name["filter"].ppr_score is not None and by_name["filter"].ppr_score > 0.0
    assert by_name["csv"].ppr_score is not None and by_name["csv"].ppr_score > 0.0
    assert by_name["orphan"].ppr_score == 0.0
