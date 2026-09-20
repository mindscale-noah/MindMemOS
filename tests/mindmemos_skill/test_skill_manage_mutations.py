"""AC-P5-002 (plan level): graph-only remove excludes target + incident edges."""

from __future__ import annotations

import pytest
from mindmemos_skill.algos.skill_manage.algorithm import DefaultSkillManageAlgorithm
from mindmemos_skill.algos.skill_manage.config import SkillManageConfig
from mindmemos_skill.typing import (
    EdgeType,
    EmbeddingProfile,
    SkillGraphEdge,
    SkillGraphMembership,
    SkillGraphNode,
    SkillGraphSnapshot,
    SkillManageRemoveInput,
    SkillManageRemoveSelector,
    SkillManageScope,
    SkillManageUpsertInput,
    SkillVersionRef,
)


def _snapshot() -> SkillGraphSnapshot:
    nodes = (
        SkillGraphNode(graph_node_id="n-csv", node_type="skill", skill_id="csv", version_id="v1", name="CSV"),
        SkillGraphNode(graph_node_id="n-filter", node_type="skill", skill_id="filter", version_id="v1", name="Filter"),
        SkillGraphNode(graph_node_id="n-scene", node_type="scene", skill_id="csv", parent_skill_id="csv"),
    )
    edges = (
        SkillGraphEdge(
            edge_id="e1", edge_type=EdgeType.DEPENDENCY, source_id="n-csv", target_id="n-filter", weight=1.0
        ),
        SkillGraphEdge(
            edge_id="e2", edge_type=EdgeType.CONTAINMENT, source_id="n-csv", target_id="n-scene", weight=1.0
        ),
    )
    memberships = (
        SkillGraphMembership(skill_id="csv", version_id="v1", graph_node_id="n-csv"),
        SkillGraphMembership(skill_id="filter", version_id="v1", graph_node_id="n-filter"),
    )
    return SkillGraphSnapshot(generation_id="g", revision=0, nodes=nodes, edges=edges, memberships=memberships)


@pytest.mark.asyncio
async def test_remove_by_version_excludes_node_incident_edges_and_membership() -> None:
    algorithm = DefaultSkillManageAlgorithm(config=SkillManageConfig(), context=object())
    plan = await algorithm.plan_remove(
        SkillManageRemoveInput(
            scope=SkillManageScope(project_id="p", graph_id="g"),
            selector=SkillManageRemoveSelector(kind="version", skill_id="csv", version_id="v1"),
            snapshot=_snapshot(),
        )
    )

    remaining_ids = {node.graph_node_id for node in plan.nodes}
    assert "n-csv" not in remaining_ids
    assert "n-scene" not in remaining_ids
    assert "n-filter" in remaining_ids
    assert len(plan.edges) == 0  # both edges were incident to the removed CSV node
    assert [m.skill_id for m in plan.memberships] == ["filter"]


@pytest.mark.asyncio
async def test_remove_by_family_excludes_whole_family() -> None:
    algorithm = DefaultSkillManageAlgorithm(config=SkillManageConfig(), context=object())
    plan = await algorithm.plan_remove(
        SkillManageRemoveInput(
            scope=SkillManageScope(project_id="p", graph_id="g"),
            selector=SkillManageRemoveSelector(kind="family", skill_id="csv"),
            snapshot=_snapshot(),
        )
    )

    remaining_ids = {node.graph_node_id for node in plan.nodes}
    assert "n-csv" not in remaining_ids
    assert "n-scene" not in remaining_ids
    assert [m.skill_id for m in plan.memberships] == ["filter"]


@pytest.mark.asyncio
async def test_remove_absent_skill_is_noop() -> None:
    algorithm = DefaultSkillManageAlgorithm(config=SkillManageConfig(), context=object())
    plan = await algorithm.plan_remove(
        SkillManageRemoveInput(
            scope=SkillManageScope(project_id="p", graph_id="g"),
            selector=SkillManageRemoveSelector(kind="family", skill_id="absent"),
            snapshot=_snapshot(),
        )
    )

    assert len(plan.nodes) == 3
    assert len(plan.edges) == 2
    assert len(plan.memberships) == 2


@pytest.mark.asyncio
async def test_upsert_without_replacement_document_preserves_snapshot() -> None:
    algorithm = DefaultSkillManageAlgorithm(config=SkillManageConfig(), context=object())
    snapshot = _snapshot()
    plan = await algorithm.plan_upsert(
        SkillManageUpsertInput(
            scope=SkillManageScope(project_id="p", graph_id="g"),
            version_ref=SkillVersionRef(skill_id="csv", version_id="v2"),
            snapshot=snapshot,
            embedding_profile=EmbeddingProfile(model="m", dimension=3072),
        )
    )

    assert plan.nodes == snapshot.nodes
    assert plan.edges == snapshot.edges
    assert plan.memberships == snapshot.memberships


@pytest.mark.asyncio
async def test_upsert_replaces_target_node_and_repoints_membership() -> None:
    """AC-P5-001: upsert with a document replaces the Skill node, not a no-op.

    The target node is rebuilt from the new version's content, its membership is
    repointed to the new ``version_id``, Scene nodes are preserved, and the
    dependency-only edge set is recomputed.
    """

    from mindmemos_skill.typing import (
        EmbeddingProfile,
        SkillDocumentInput,
        SkillManageUpsertInput,
        SkillVersionRef,
    )

    algorithm = DefaultSkillManageAlgorithm(config=SkillManageConfig(), context=object())
    # The build convention assigns ``graph_node_id=f"skill:{skill_id}"``; upsert
    # must target the same ID to replace the node in place.
    snapshot = SkillGraphSnapshot(
        generation_id="g",
        revision=0,
        nodes=(
            SkillGraphNode(graph_node_id="skill:csv", node_type="skill", skill_id="csv", version_id="v1", name="CSV"),
            SkillGraphNode(
                graph_node_id="skill:filter", node_type="skill", skill_id="filter", version_id="v1", name="Filter"
            ),
            SkillGraphNode(graph_node_id="n-scene", node_type="scene", skill_id="csv", parent_skill_id="csv"),
        ),
        edges=(
            SkillGraphEdge(
                edge_id="e1", edge_type=EdgeType.DEPENDENCY, source_id="skill:csv", target_id="skill:filter", weight=1.0
            ),
            SkillGraphEdge(
                edge_id="e2", edge_type=EdgeType.CONTAINMENT, source_id="skill:csv", target_id="n-scene", weight=1.0
            ),
        ),
        memberships=(
            SkillGraphMembership(skill_id="csv", version_id="v1", graph_node_id="skill:csv"),
            SkillGraphMembership(skill_id="filter", version_id="v1", graph_node_id="skill:filter"),
        ),
    )
    document = SkillDocumentInput(
        skill_id="csv",
        version_id="v2",
        content=(
            "---\nname: CSV Parser\n"
            "description: Parses raw CSV text into records.\n"
            "inputs: [raw_csv_text]\n"
            "outputs: [parsed_records]\n"
            "---\n\n# CSV Parser\n\nv2 body.\n"
        ),
    )
    plan = await algorithm.plan_upsert(
        SkillManageUpsertInput(
            scope=SkillManageScope(project_id="p", graph_id="g"),
            version_ref=SkillVersionRef(skill_id="csv", version_id="v2"),
            snapshot=snapshot,
            embedding_profile=EmbeddingProfile(model="m", dimension=3072),
            document=document,
        )
    )

    # The target node is replaced with the re-parsed v2 content.
    by_id = {node.graph_node_id: node for node in plan.nodes}
    assert by_id["skill:csv"].version_id == "v2"
    assert by_id["skill:csv"].name == "CSV Parser"
    # Membership for the family is repointed to v2.
    memberships = {m.skill_id: m for m in plan.memberships}
    assert memberships["csv"].version_id == "v2"
    # Scene node is preserved.
    assert "n-scene" in by_id
    # Containment edge is preserved (non-dependency, intact endpoints).
    assert any(edge.edge_type == EdgeType.CONTAINMENT for edge in plan.edges)
