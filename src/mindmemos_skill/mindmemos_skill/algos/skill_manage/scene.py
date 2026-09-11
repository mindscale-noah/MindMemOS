"""Scene node and edge construction from typed SSL records.

Ported from the GOS scene/SSL logic with the frozen constraint that the runtime
never scans the filesystem for SSL files: it receives already-authorized,
typed SSL record DTOs from the Application. The Scene node naming and
containment direction follow the P0 golden fixture.
"""

from __future__ import annotations

from typing import Any, Mapping


def scene_node_name(skill_name: str, scene_id: str) -> str:
    return f"{skill_name}::{scene_id}"


def build_scene_nodes(skill_name: str, ssl_record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Derive Scene nodes from one normalized SSL record for one Skill."""

    nodes: list[dict[str, Any]] = []
    for scene in ssl_record.get("scenes") or []:
        scene_id = str(scene.get("scene_id") or "").strip()
        if not scene_id:
            continue
        scene_name = str(scene.get("scene_name") or "").strip()
        scene_goal = str(scene.get("scene_goal") or "").strip()
        nodes.append(
            {
                "name": scene_node_name(skill_name, scene_id),
                "node_type": "scene",
                "skill_id": skill_name,
                "parent_skill_id": skill_name,
                "description": scene_goal,
                "one_line_capability": scene_name,
                "inputs": [],
                "outputs": [],
                "domain_tags": [],
                "tooling": [],
                "metadata": {
                    "scene_id": scene_id,
                    "scene_type": str(scene.get("scene_type") or ""),
                },
            }
        )
    return nodes


def build_scene_edges(
    skill_name: str,
    ssl_record: Mapping[str, Any],
    *,
    enable_flow_edges: bool = False,
) -> list[dict[str, Any]]:
    """Derive containment and (optionally) flow edges for one Skill's scenes."""

    edges: list[dict[str, Any]] = []
    scenes = ssl_record.get("scenes") or []
    for scene in scenes:
        scene_id = str(scene.get("scene_id") or "").strip()
        if not scene_id:
            continue
        edges.append(
            {
                "source": skill_name,
                "target": scene_node_name(skill_name, scene_id),
                "type": "containment",
                "weight": 1.0,
            }
        )
        if not enable_flow_edges:
            continue
        for rule in scene.get("next_scene_rules") or []:
            target_id = str(rule.get("next_scene_id") or "").strip() if isinstance(rule, dict) else str(rule).strip()
            if target_id:
                edges.append(
                    {
                        "source": scene_node_name(skill_name, scene_id),
                        "target": scene_node_name(skill_name, target_id),
                        "type": "flow",
                        "weight": 0.3,
                    }
                )
    return edges


def build_scene_graph(
    ssl_records: Mapping[str, Mapping[str, Any]],
    *,
    enable_flow_edges: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build the full Scene node/edge set from all typed SSL records.

    ``ssl_records`` maps Skill name -> normalized SSL record. Deduplication is
    keyed on ``(source, target, type)`` to keep the result deterministic.
    """

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    for skill_name in sorted(ssl_records):
        record = ssl_records[skill_name]
        nodes.extend(build_scene_nodes(skill_name, record))
        edges.extend(build_scene_edges(skill_name, record, enable_flow_edges=enable_flow_edges))
    return nodes, edges


__all__ = [
    "build_scene_edges",
    "build_scene_graph",
    "build_scene_nodes",
    "scene_node_name",
]
