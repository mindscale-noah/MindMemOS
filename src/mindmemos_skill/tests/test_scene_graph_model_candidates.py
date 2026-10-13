"""Model candidate budgets preserve type scope and exclude existing dependencies."""

from collections import Counter

from mindmemos_skill.algos.manage.skill_manage.algorithm import node_id
from mindmemos_skill.algos.manage.skill_manage.config import JointLinkingConfig
from mindmemos_skill.algos.manage.skill_manage.linking import (
    joint_candidate_pairs,
    joint_candidates,
    joint_dependency_edges,
    joint_model_candidates,
)
from mindmemos_skill.algos.manage.skill_manage.typing import DerivedNodeMetadata
from mindmemos_skill.persistence.db_services.graphs.typing import GraphNodeType, SkillGraphNodeRecord


def _node(member_index, scene_index=None, *, inputs=(), outputs=()):
    """Create a validated node with a shared vector and explicit family ownership."""
    member = f"member-{member_index:02d}"
    scene = None if scene_index is None else f"step-{scene_index:02d}"
    return SkillGraphNodeRecord(
        project_id="candidate-test",
        graph_id="model-candidates",
        graph_version_id="skgv-" + "0" * 31 + "1",
        node_id=node_id(member, scene),
        node_type=GraphNodeType.SKILL if scene is None else GraphNodeType.SCENE,
        skill_version_id=f"version-{member}" if scene is None else None,
        owner_node_id=node_id(member) if scene is not None else None,
        name=scene or member,
        inputs=list(inputs),
        outputs=list(outputs),
        metadata=DerivedNodeMetadata(
            collection_skill_id=member, family_id=f"family-{member}", scene_id=scene
        ).model_dump(mode="json"),
        embedding=[1, 0, 0],
        embedding_input_hash="1" * 64,
    )


def test_scene_crowding_cannot_consume_the_skill_model_candidate_budget():
    skills = [_node(index) for index in range(3)]
    scenes = [_node(index, scene_index) for index in range(3) for scene_index in range(10)]
    nodes = skills + scenes
    vectors = [[1, 0, 0] for _ in nodes]
    by_id = {node.node_id: node for node in nodes}
    config = JointLinkingConfig(dense_candidates=64, lexical_candidates=0, io_candidates=0, max_candidates=1)
    unscoped = joint_candidates(nodes, vectors, config)
    assert all(by_id[unscoped[skill.node_id][0]].node_type == GraphNodeType.SCENE for skill in skills)

    scoped = joint_model_candidates(
        nodes, vectors, config, skill_linking=True, scene_linking=False, dependency_pairs=set()
    )
    assert set(scoped) == {skill.node_id for skill in skills}
    assert all(len(targets) == 1 and by_id[targets[0]].node_type == GraphNodeType.SKILL for targets in scoped.values())
    assert scoped == joint_model_candidates(
        list(reversed(nodes)),
        list(reversed(vectors)),
        config,
        skill_linking=True,
        scene_linking=False,
        dependency_pairs=set(),
    )


def test_scene_only_model_candidates_cover_joint_targets_without_skill_sources():
    nodes = [_node(index) for index in range(3)] + [_node(index, 0) for index in range(3)]
    by_id = {node.node_id: node for node in nodes}
    config = JointLinkingConfig(dense_candidates=20, lexical_candidates=0, io_candidates=0, max_candidates=20)
    selected = joint_model_candidates(
        nodes, [[1, 0, 0] for _ in nodes], config, skill_linking=False, scene_linking=True, dependency_pairs=set()
    )
    assert set(selected) == {node.node_id for node in nodes if node.node_type == GraphNodeType.SCENE}
    assert all(len(targets) == 4 for targets in selected.values())
    pairs = list(joint_candidate_pairs(nodes, selected))
    assert {tuple(sorted((left.node_type, right.node_type))) for left, right in pairs} == {
        (GraphNodeType.SCENE, GraphNodeType.SCENE),
        (GraphNodeType.SCENE, GraphNodeType.SKILL),
    }
    assert all(
        by_id[source].metadata["family_id"] != by_id[target].metadata["family_id"]
        for source, targets in selected.items()
        for target in targets
    )


def test_per_source_limit_applies_after_dependency_filtering_before_pair_union():
    nodes = [
        _node(index, inputs=("shared_rows",) if index == 0 else (), outputs=("shared_rows",) if 1 <= index <= 4 else ())
        for index in range(15)
    ]
    dependencies = list(joint_dependency_edges(nodes, dependency_match_threshold=0.6))
    assert {(edge["source"], edge["target"]) for edge in dependencies} == {
        (nodes[index].node_id, nodes[0].node_id) for index in range(1, 5)
    }
    dependency_pairs = {frozenset((edge["source"], edge["target"])) for edge in dependencies}
    config = JointLinkingConfig(dense_candidates=20, lexical_candidates=0, io_candidates=0, max_candidates=20)
    selected = joint_model_candidates(
        nodes,
        [[1, 0, 0] for _ in nodes],
        config,
        skill_linking=True,
        scene_linking=False,
        dependency_pairs=dependency_pairs,
    )
    assert all(len(targets) == 8 for targets in selected.values())
    assert selected[nodes[0].node_id] == [node.node_id for node in nodes[5:13]]
    assert all(
        frozenset((source, target)) not in dependency_pairs
        for source, targets in selected.items()
        for target in targets
    )
    pairs = list(joint_candidate_pairs(nodes, selected))
    unique_pairs = {frozenset((left.node_id, right.node_id)) for left, right in pairs}
    assert len(pairs) == len(unique_pairs)
    assert 8 < len(pairs) < sum(len(targets) for targets in selected.values())
    incident_counts = Counter(node.node_id for pair in pairs for node in pair)
    assert incident_counts[nodes[0].node_id] > 8
