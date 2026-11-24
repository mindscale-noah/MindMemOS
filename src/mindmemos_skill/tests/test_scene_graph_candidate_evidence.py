"""Relation inputs and evidence-based candidates preserve useful reference semantics."""

import pytest
from mindmemos_skill.algos.manage.skill_manage.config import JointLinkingConfig
from mindmemos_skill.algos.manage.skill_manage.linking import joint_candidates, joint_model_candidates
from mindmemos_skill.algos.manage.skill_manage.typing import DerivedNodeMetadata

from .test_scene_graph_model_candidates import _node


def _features(index, *, metadata=None, **fields):
    """Add explicit public features to a node with a stable family identity."""
    node = _node(index)
    if metadata:
        fields["metadata"] = DerivedNodeMetadata.model_validate({**node.metadata, **metadata}).model_dump(mode="json")
    return node.model_copy(update=fields)


def _rank(nodes, strategy, **settings):
    """Select candidates using fixed controlled vector order and explicit budgets."""
    config = JointLinkingConfig(candidate_strategy=strategy, **settings)
    vectors = [[1, 0, 0] for _ in nodes] if config.dense_candidates else []
    return joint_candidates(nodes, vectors, config)[nodes[0].node_id]


def test_evidence_gate_rejects_dense_only_similarity_and_weak_field_overlap():
    nodes = [
        _features(0, domain_tags=["alpha", "bravo", "charlie", "delta"]),
        _features(1, domain_tags=["alpha", "echo", "foxtrot", "golf"]),
        _features(2),
    ]
    assert _rank(nodes, "rrf")
    assert _rank(nodes, "field_evidence") == []


@pytest.mark.parametrize(
    "features",
    [
        {"domain_tags": ["cardiology"]},
        {"tooling": ["pandas"]},
        {"example_tasks": ["compare randomized trials"]},
        {"inputs": ["trial_records"]},
        {"metadata": {"local_scripts": ["scripts/compare_trials.py"]}},
    ],
)
def test_each_supported_field_can_independently_supply_evidence(features):
    nodes = [_features(index, **features) for index in range(2)]
    assert _rank(nodes, "field_evidence", dense_candidates=0) == [nodes[1].node_id]


def test_field_weights_can_outrank_a_dense_neighbor_even_when_rrf_prefers_it():
    nodes = [
        _features(0, domain_tags=["cardiology"], tooling=["convert"]),
        _features(1, tooling=["convert"]),
        _features(2, domain_tags=["cardiology"], tooling=["convert"], embedding=[0, 1, 0]),
    ]
    assert _rank(nodes, "rrf", max_candidates=1, io_candidates=0) == [nodes[1].node_id]
    assert _rank(nodes, "field_evidence", max_candidates=1, io_candidates=0) == [nodes[2].node_id]


def test_field_strategy_preserves_dependency_pair_exclusion_and_per_source_cap():
    nodes = [_features(index, domain_tags=["cardiology"]) for index in range(5)]
    pair = frozenset((nodes[0].node_id, nodes[1].node_id))
    selected = joint_model_candidates(
        nodes,
        [[1, 0, 0] for _ in nodes],
        JointLinkingConfig(candidate_strategy="field_evidence", model_candidates=2),
        skill_linking=True,
        scene_linking=False,
        dependency_pairs={pair},
    )
    assert all(len(targets) == 2 for targets in selected.values())
    assert all(pair != frozenset((source, target)) for source, targets in selected.items() for target in targets)


def test_rank_fusion_remains_the_default_without_claiming_measured_quality():
    assert JointLinkingConfig().candidate_strategy == "rrf"
    assert JointLinkingConfig().model_candidates == 8


@pytest.mark.parametrize(
    "metadata",
    [
        {"intent_signature": "quasar normalization"},
        {"local_scripts": ["scripts/ephemeris.py"]},
        {"external_dependencies": ["astropy"]},
        {"compatibility": ["Python 3.11"]},
        {"allowed_tools": ["Bash"]},
    ],
)
def test_default_rrf_lexical_route_uses_independent_derived_features(metadata):
    nodes = [_features(index, name=name) for index, name in enumerate(("alpha", "bravo", "charlie"))]
    assert _rank(nodes, "rrf", dense_candidates=0, io_candidates=0) == []
    nodes = [
        _features(0, name="alpha", metadata=metadata),
        _features(1, name="bravo", metadata=metadata),
        nodes[2],
    ]
    assert _rank(nodes, "rrf", dense_candidates=0, io_candidates=0) == [nodes[1].node_id]
