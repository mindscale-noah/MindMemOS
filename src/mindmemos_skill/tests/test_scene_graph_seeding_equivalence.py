"""Exact seed-ranking oracles captured before request-local feature reuse."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from mindmemos_skill.algos.manage.skill_manage import seeding
from mindmemos_skill.algos.manage.skill_manage.typing import QuerySchema, SceneGraphContent


@pytest.fixture
def nodes():
    """Load the shared immutable graph fixture."""
    payload = json.loads((Path(__file__).parent / "fixtures/scene_graph_d0.json").read_text())
    return SceneGraphContent.model_validate(payload["content"]).nodes


@pytest.mark.parametrize(
    "goal,task_name,seeds,threshold,expected_scores,expected_ranks",
    [
        (
            "csv validate report",
            "validate",
            3,
            0,
            [
                ('["scene","d","sketch"]', 1, None, 1, "0x1.3333333333333p+0", "0x1.1745d1745d175p-1"),
                ('["scene","a::csv","validate"]', None, 1, 2, "0x1.0000000000000p+0", "0x1.1745d1745d175p-2"),
                ('["skill","d"]', 2, None, 3, "0x1.3333333333333p-1", "0x1.745d1745d1746p-3"),
            ],
            [
                ('["scene","d","sketch"]', 1, None, 1, True),
                ('["scene","a::csv","validate"]', None, 1, 2, True),
                ('["skill","d"]', 2, None, 3, True),
                ('["skill","a::csv"]', None, 2, 4, False),
                ('["scene","c","notes"]', 3, None, 5, False),
                ('["skill","b"]', None, 3, 6, False),
                ('["skill","c"]', 4, None, 7, False),
                ('["scene","a::csv","export"]', None, 4, 8, False),
            ],
        ),
        (
            "zzzzunmatched",
            "",
            3,
            0,
            [
                ('["scene","d","sketch"]', 1, None, 1, "0x1.3333333333333p+0", "0x1.1745d1745d175p-1"),
                ('["skill","d"]', 2, None, 2, "0x1.3333333333333p-1", "0x1.1745d1745d175p-2"),
                ('["scene","c","notes"]', 3, None, 3, "0x1.9999999999999p-2", "0x1.745d1745d1746p-3"),
            ],
            [
                ('["scene","d","sketch"]', 1, None, 1, True),
                ('["skill","d"]', 2, None, 2, True),
                ('["scene","c","notes"]', 3, None, 3, True),
                ('["skill","c"]', 4, None, 4, False),
            ],
        ),
        (
            "数据分析",
            "",
            3,
            0,
            [
                ('["scene","d","sketch"]', 1, None, 1, "0x1.3333333333333p+0", "0x1.1745d1745d175p-1"),
                ('["skill","d"]', 2, None, 2, "0x1.3333333333333p-1", "0x1.1745d1745d175p-2"),
                ('["scene","c","notes"]', 3, None, 3, "0x1.9999999999999p-2", "0x1.745d1745d1746p-3"),
            ],
            [
                ('["scene","d","sketch"]', 1, None, 1, True),
                ('["skill","d"]', 2, None, 2, True),
                ('["scene","c","notes"]', 3, None, 3, True),
                ('["skill","c"]', 4, None, 4, False),
            ],
        ),
        (
            "the",
            "",
            3,
            1,
            [('["scene","d","sketch"]', 1, None, 1, "0x1.3333333333333p+0", "0x1.0000000000000p+0")],
            [('["scene","d","sketch"]', 1, None, 1, True)],
        ),
        ("csv", "", 0, 0, [], []),
    ],
)
def test_seed_results_match_exact_preoptimization_oracles(
    nodes, goal, task_name, seeds, threshold, expected_scores, expected_ranks
):
    """Keep ranks, diagnostics, and every floating-point bit unchanged."""
    hits = [
        SimpleNamespace(record={"node_id": node.node_id}, distance=index / 10)
        for index, node in enumerate(reversed(nodes))
    ]
    diagnostics = []
    result = seeding.select_seeds(
        nodes,
        hits,
        QuerySchema(goal=goal, task_name=task_name),
        candidates=4,
        seeds=seeds,
        threshold=threshold,
        diagnostics=diagnostics,
    )
    assert [
        (hit.node_id, hit.dense_rank, hit.lexical_rank, hit.fused_rank, hit.fusion_score.hex(), hit.weight.hex())
        for hit in result
    ] == expected_scores
    assert [
        (hit.node_id, hit.dense_rank, hit.lexical_rank, hit.fused_rank, hit.selected) for hit in diagnostics
    ] == expected_ranks


def test_query_preparation_does_not_grow_with_node_count(nodes):
    """Prepare query values a fixed number of times within each selection."""
    schema = QuerySchema(goal="csv validate report")
    counts = []
    for count in (1, 100):
        candidates = [nodes[0].model_copy(update={"node_id": f"node-{index}"}) for index in range(count)]
        with patch.object(seeding, "query_values", wraps=seeding.query_values) as query_values:
            seeding.select_seeds(candidates, [], schema, candidates=4, seeds=3, threshold=0)
            counts.append(query_values.call_count)
    assert counts == [2, 2]


def test_prepared_query_features_are_not_shared_across_calls(nodes):
    """A later query must use its own fields, even after an earlier match."""
    matching = QuerySchema(goal="csv validate report")
    unrelated = QuerySchema(goal="zzzzunmatched")
    initial = seeding.select_seeds(nodes, [], matching, candidates=4, seeds=3, threshold=0)
    assert initial
    assert seeding.select_seeds(nodes, [], unrelated, candidates=4, seeds=3, threshold=0) == []
    assert seeding.select_seeds(nodes, [], matching, candidates=4, seeds=3, threshold=0) == initial
