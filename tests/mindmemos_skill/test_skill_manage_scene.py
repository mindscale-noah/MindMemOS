"""AC-P2-002 (scene portion): Scene node/edge construction from typed SSL records."""

from __future__ import annotations

import json
from pathlib import Path

from mindmemos_skill.algos.skill_manage.scene import (
    build_scene_edges,
    build_scene_graph,
    build_scene_nodes,
    scene_node_name,
)

FIXTURE = Path(__file__).parent / "fixtures" / "skill_manage" / "gos_minimal"
SSL_RECORD = json.loads((FIXTURE / "ssl_records" / "CSV Parser Skill.json").read_text(encoding="utf-8"))


def test_scene_node_name_format() -> None:
    assert scene_node_name("CSV Parser Skill", "scene_parse") == "CSV Parser Skill::scene_parse"


def test_scene_nodes_match_golden() -> None:
    nodes = build_scene_nodes("CSV Parser Skill", SSL_RECORD)

    assert len(nodes) == 1
    node = nodes[0]
    assert node["name"] == "CSV Parser Skill::scene_parse"
    assert node["node_type"] == "scene"
    assert node["description"] == "Parse raw CSV text into records"
    assert node["one_line_capability"] == "Parse CSV"
    assert node["parent_skill_id"] == "CSV Parser Skill"


def test_containment_edge_matches_golden() -> None:
    edges = build_scene_edges("CSV Parser Skill", SSL_RECORD, enable_flow_edges=False)

    assert len(edges) == 1
    assert edges[0] == {
        "source": "CSV Parser Skill",
        "target": "CSV Parser Skill::scene_parse",
        "type": "containment",
        "weight": 1.0,
    }


def test_scene_graph_builds_nodes_and_edges() -> None:
    nodes, edges = build_scene_graph({"CSV Parser Skill": SSL_RECORD}, enable_flow_edges=False)
    assert [n["name"] for n in nodes] == ["CSV Parser Skill::scene_parse"]
    assert [(e["source"], e["target"], e["type"]) for e in edges] == [
        ("CSV Parser Skill", "CSV Parser Skill::scene_parse", "containment")
    ]


def test_flow_edges_only_when_enabled() -> None:
    record = {
        "scenes": [
            {
                "scene_id": "scene_a",
                "scene_name": "A",
                "scene_goal": "Do A",
                "next_scene_rules": [{"next_scene_id": "scene_b"}],
            },
            {"scene_id": "scene_b", "scene_name": "B", "scene_goal": "Do B", "next_scene_rules": []},
        ]
    }
    no_flow = build_scene_edges("Skill", record, enable_flow_edges=False)
    with_flow = build_scene_edges("Skill", record, enable_flow_edges=True)
    assert len(no_flow) == 2
    assert ("Skill::scene_a", "Skill::scene_b", "flow") in {(e["source"], e["target"], e["type"]) for e in with_flow}
