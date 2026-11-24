"""Original numeric retrieval acceptance oracles migrated without changing calculations."""

import json
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from mindmemos_skill.algos.manage.skill_manage.algorithm import SkillManageAlgorithm, node_id
from mindmemos_skill.algos.manage.skill_manage.config import EmbeddingProfile, SkillManageConfig
from mindmemos_skill.algos.manage.skill_manage.query import normalize_query, public_task_text, query_text
from mindmemos_skill.algos.manage.skill_manage.retrieval import pagerank, rank_joint
from mindmemos_skill.algos.manage.skill_manage.seeding import balance_seeds, field_score, select_seeds
from mindmemos_skill.algos.manage.skill_manage.typing import QuerySchema, RetrievalSeed, SceneGraphContent
from mindmemos_skill.envs.typing import Task
from mindmemos_skill.infra.embedding import OpenAIEmbeddingConfig
from mindmemos_skill.persistence.db_services.graphs.typing import GraphNodeType
from pydantic import ValidationError


@pytest.fixture
def reference():
    return json.loads((Path(__file__).parent / "fixtures/scene_graph_d0.json").read_text())


@pytest.fixture
def content(reference):
    return SceneGraphContent.model_validate(reference["content"])


@pytest.fixture
def neutral_skills(content):
    return [
        node.model_copy(
            update={
                "name": "ordinary",
                "description": "",
                "one_line_capability": "",
                "inputs": [],
                "outputs": [],
                "tooling": [],
                "domain_tags": [],
                "example_tasks": [],
                "rendered_snippet": "",
                "metadata": {**node.metadata, "intent": ""},
            }
        )
        for node in content.nodes
        if node.node_type == GraphNodeType.SKILL
    ]


def config(**updates):
    """Create validated retrieval settings without external model access.

    Args:
        **updates: Explicit options to exercise.
    """
    return SkillManageConfig(
        embedding=OpenAIEmbeddingConfig(model="fixture", dimensions=3, base_url="http://fixture.invalid"),
        llm_linking=False,
        query_rewrite=False,
        **updates,
    )


def hit(node, distance=0):
    """Represent a database cosine-distance result.

    Args:
        node: Candidate node identity.
        distance: Reported cosine distance.
    """
    return SimpleNamespace(record={"node_id": node}, distance=distance)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "owners,include_scenes,tied,expected_offsets",
    [
        (list(range(100)), False, False, [0]),
        ([0, 1], False, False, [0]),
        (list(range(100)), False, True, [0]),
        ([0] * 200 + list(range(1, 100)), True, False, [0, 100, 200]),
        ([0] * 100 + [1] * 100, True, False, [0, 100, 200]),
        ([0] * 99 + list(range(1, 100)), True, True, [0, 100]),
        ([0] * 500 + list(range(1, 100)), True, False, [0, 100, 200, 300, 400]),
        ([0] * 499 + list(range(1, 100)), True, False, [0, 100, 200, 300, 400]),
        ([0] * 496 + list(range(1, 100)), True, False, [0, 100, 200, 300, 400]),
    ],
    ids=[
        "skill-only-top-k",
        "skill-only-fewer-than-limit",
        "skill-only-ties",
        "deduplicate-across-pages",
        "exhaust-before-limit",
        "ties-across-pages",
        "stop-at-node-budget",
        "include-last-budgeted-node",
        "reach-owner-limit-at-budget",
    ],
)
async def test_retrieve_vector_stops_at_unique_owner_limit(owners, include_scenes, tied, expected_offsets):
    """Stop after five unique owners while preserving best hits and stable ties."""
    hits = [
        SimpleNamespace(
            distance=0.25 if tied else index / len(owners),
            record={
                "node_id": f"node-{index:04d}",
                "node_type": GraphNodeType.SCENE if include_scenes else GraphNodeType.SKILL,
                "skill_version_id": None if include_scenes else f"version-{owner}",
                "metadata": {"collection_skill_id": str(owner)},
            },
        )
        for index, owner in enumerate(owners)
    ]

    def search_hits(snapshot_id, vector, *, top_k, node_type=None, offset=0):
        """Return a sorted page using the store's default query arguments."""
        return hits[offset : offset + top_k]

    store = SimpleNamespace(search=AsyncMock(side_effect=search_hits))
    algorithm = SkillManageAlgorithm(config(), store=store)
    algorithm.embedding_model = SimpleNamespace(embed=AsyncMock(return_value=[[1.0, 0.0, 0.0]]))
    algorithm.bind_vector_snapshot(
        "snapshot",
        skill_ids={str(owner) for owner in owners},
        skill_versions={str(owner): f"version-{owner}" for owner in owners},
    )

    result = await algorithm.retrieve_vector(
        task=Task(id="query", instruction="Find a Skill"), limit=5, include_scenes=include_scenes
    )

    expected_owners = list(dict.fromkeys(owners[:500] if include_scenes else owners))[:5]
    assert [score.member_id for score in result.scores] == [str(owner) for owner in expected_owners]
    assert [score.node_id for score in result.scores] == [
        f"node-{owners.index(owner):04d}" for owner in expected_owners
    ]
    assert [score.similarity for score in result.scores] == pytest.approx(
        [1 - hits[owners.index(owner)].distance for owner in expected_owners]
    )
    assert [score.skill_version_id for score in result.scores] == [f"version-{owner}" for owner in expected_owners]
    assert [call.kwargs.get("offset", 0) for call in store.search.await_args_list] == expected_offsets
    assert store.search.await_count <= 5
    assert all(call.kwargs["top_k"] == (100 if include_scenes else 5) for call in store.search.await_args_list)
    if not include_scenes:
        assert "offset" not in store.search.await_args.kwargs
    assert all(
        call.kwargs.get("node_type") == (None if include_scenes else GraphNodeType.SKILL)
        for call in store.search.await_args_list
    )


def seed(node, weight=1):
    """Create an explicit personalization seed.

    Args:
        node: Candidate node identity.
        weight: Positive prior mass.
    """
    return RetrievalSeed(node_id=node, weight=weight, fusion_score=0)


@pytest.mark.parametrize("aggregation,alpha", [("max", 0), ("max", 0.5), ("mean", 0.5), ("mean", 2)])
def test_exact_ppr_and_alpha_reduction(reference, content, aggregation, alpha):
    seeds = [seed(reference["scene_only_query"]["seeds"][0])]
    expected = {key: float(Fraction(value)) for key, value in reference["expected"]["stationary_ppr_fraction"].items()}
    scores = pagerank(content.nodes, content.edges, {seeds[0].node_id: 1}, max_iter=500, tolerance=1e-12)
    assert scores == pytest.approx(expected, abs=1e-11)
    results = rank_joint(
        content.nodes,
        content.edges,
        seeds,
        normalize_query("nebula dedupe stage"),
        config=config(mode="unified_ppr", alpha=alpha, aggregation=aggregation, max_iter=500, tolerance=1e-12),
        limit=4,
    )
    assert {result.member_id for result in results} == {"a::csv", "b", "c", "d"}
    for result in results:
        child = [expected[node.node_id] for node in content.nodes if node.owner_node_id == node_id(result.member_id)]
        aggregate = (max(child) if aggregation == "max" else sum(child) / len(child)) if child else 0
        assert result.ppr_score == pytest.approx(expected[node_id(result.member_id)], abs=1e-11)
        assert result.scene_boost == pytest.approx(alpha * aggregate, abs=1e-11)
        assert result.graph_score == pytest.approx(result.ppr_score + result.scene_boost)


def test_explicit_prior_zero_edges_dangling_and_no_seeds(content):
    nodes = [node for node in content.nodes if node.node_type == GraphNodeType.SKILL][:2]
    a, b = [node.node_id for node in nodes]
    edge = SimpleNamespace(source_node_id=a, target_node_id=b, edge_type="dependency", weight=0)
    assert pagerank(nodes, [edge], {a: 3, b: 1}) == pytest.approx({a: 0.75, b: 0.25})
    assert pagerank(nodes, [edge], {a: 1}) == {a: 1, b: 0}
    assert pagerank(nodes, [edge], {a: 0, "outside": 1}) == {a: 0, b: 0}
    assert rank_joint(nodes, [edge], [], normalize_query(nodes[0].name), config=config(), limit=2) == []


@pytest.mark.parametrize("weight", [-1, float("nan"), float("inf")])
def test_invalid_ppr_weights_rejected(content, weight):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        pagerank(content.nodes, content.edges, {content.nodes[0].node_id: weight})


def test_type_budgets_union_ranks_and_balanced_mass(content):
    skills = [node for node in content.nodes if node.node_type == GraphNodeType.SKILL]
    # Force lexical evidence onto a different candidate from the best dense hit.
    skills = [
        node.model_copy(
            update={
                "name": "ordinary",
                "description": "",
                "one_line_capability": "",
                "inputs": [],
                "outputs": [],
                "tooling": [],
                "example_tasks": [],
                "rendered_snippet": "",
                "metadata": {**node.metadata, "intent": ""},
            }
        )
        for node in skills
    ]
    skills[1] = skills[1].model_copy(update={"description": "quasar"})
    schema = normalize_query("quasar")
    hits = [hit(skills[0].node_id), hit(skills[2].node_id, 0.1), hit("outside", -1)]
    selected = select_seeds(skills, hits, schema, candidates=1, seeds=2, threshold=0.5)
    assert {entry.node_id for entry in selected} == {skills[0].node_id, skills[1].node_id}
    assert {(entry.dense_rank, entry.lexical_rank) for entry in selected} == {(1, None), (None, 1)}
    assert [entry.fused_rank for entry in selected] == [1, 2]
    assert [entry.weight for entry in selected] == pytest.approx([2 / 3, 1 / 3])
    assert sum(entry.weight for entry in selected) == pytest.approx(1)
    scene = seed(node_id("a::csv", "validate"))
    balanced = balance_seeds(selected, [scene])
    assert [entry.weight for entry in balanced] == pytest.approx([1 / 3, 1 / 6, 1 / 2])
    assert [entry.fused_rank for entry in balanced] == [1, 2, None]
    assert balance_seeds([], selected) == selected
    assert balance_seeds([], []) == []


def test_dual_route_fusion_formula_and_deterministic_ties(content):
    nodes = [node for node in content.nodes if node.node_type == GraphNodeType.SKILL]
    schema = normalize_query(nodes[0].name)
    selected = select_seeds(nodes, [hit(nodes[0].node_id)], schema, candidates=4, seeds=4, threshold=0)
    entry = next(entry for entry in selected if entry.node_id == nodes[0].node_id)
    assert entry.dense_rank == 1 and entry.lexical_rank is not None
    expected = field_score(schema, nodes[0]) + 1 + 0.2 + 0.15 / entry.lexical_rank + 0.1
    assert entry.fusion_score == pytest.approx(expected)
    unknown = normalize_query("zzzzunmatched")
    tied = select_seeds(
        nodes, [hit(node.node_id) for node in reversed(nodes)], unknown, candidates=4, seeds=2, threshold=0
    )
    assert [entry.node_id for entry in tied] == sorted(node.node_id for node in nodes)[:2]
    assert select_seeds(nodes, [], unknown, candidates=4, seeds=2, threshold=0) == []


@pytest.mark.parametrize("include_dense", [False, True], ids=["lexical-only", "reversed-dense-order"])
def test_final_rank_bonus_uses_fused_skill_order(neutral_skills, include_dense):
    nodes = neutral_skills[:3]
    nodes[0] = nodes[0].model_copy(update={"description": "quasar", "tooling": ["quasar"]})
    schema = QuerySchema(goal="quasar")
    dense = [hit(nodes[1].node_id)]
    if include_dense:
        dense.append(hit(nodes[0].node_id, 0.1))
    selected = select_seeds(nodes, dense, schema, candidates=2, seeds=2, threshold=0)
    assert [entry.node_id for entry in selected] == [nodes[0].node_id, nodes[1].node_id]
    assert [entry.dense_rank for entry in selected] == [2 if include_dense else None, 1]
    assert [entry.fused_rank for entry in selected] == [1, 2]
    assert [entry.fusion_score for entry in selected] == pytest.approx([2.7 if include_dense else 2.0, 1.2])

    result = rank_joint(nodes, [], selected, schema, config=config(mode="skill_only"), limit=3)
    assert [entry.member_id for entry in result] == ["a::csv", "b", "c"]
    assert [entry.graph_score for entry in result] == pytest.approx([2 / 3, 1 / 3, 0])
    # Description/tooling contribute 0.9 + 0.95; only selected Skills get rank bonuses.
    assert [entry.rerank_score for entry in result] == pytest.approx([2 / 3 + 1.85 + 0.2, 1 / 3 + 0.1, 0])


def test_fused_rank_bonus_changes_final_order_after_dual_ppr(neutral_skills, content):
    nodes = neutral_skills[:3]
    nodes[0] = nodes[0].model_copy(update={"description": "quasar"})
    scene = next(node for node in content.nodes if node.owner_node_id == nodes[1].node_id)
    schema = QuerySchema(goal="quasar", keywords=["nebula"])
    skills = select_seeds(
        nodes, [hit(nodes[1].node_id), hit(nodes[0].node_id, 0.1)], schema, candidates=2, seeds=2, threshold=0
    )
    scenes = select_seeds([scene], [hit(scene.node_id)], schema, candidates=1, seeds=1, threshold=0)
    assert [entry.node_id for entry in skills] == [nodes[0].node_id, nodes[1].node_id]
    assert [entry.fusion_score for entry in skills] == pytest.approx([1.3, 1.2])
    selected = balance_seeds(scenes, skills)
    assert [entry.fused_rank for entry in selected] == [1, 1, 2]
    result = rank_joint(
        [*nodes, scene],
        [],
        selected,
        schema,
        config=config(mode="separate", scene_ppr=True, alpha=0.8),
        limit=3,
    )
    assert [entry.member_id for entry in result] == ["a::csv", "b", "c"]
    assert [entry.ppr_score for entry in result] == pytest.approx([2 / 3, 1 / 3, 0])
    assert [entry.scene_boost for entry in result] == pytest.approx([0, 0.8, 0])
    assert [entry.rerank_score for entry in result] == pytest.approx([79 / 60, 37 / 30, 0])
    # Dense-rank bonuses instead produce 73/60 and 4/3, incorrectly placing B first.
    assert result[0].graph_score < result[1].graph_score


def test_manual_priors_without_fused_rank_have_no_rank_bonus(neutral_skills):
    nodes = neutral_skills[:2]
    selected = [
        RetrievalSeed(node_id=nodes[1].node_id, dense_rank=1, weight=0.5, fusion_score=0),
        seed(nodes[0].node_id, 0.5),
    ]
    result = rank_joint(nodes, [], selected, QuerySchema(goal="quasar"), config=config(mode="skill_only"), limit=2)
    assert [entry.member_id for entry in result] == ["a::csv", "b"]
    assert [entry.graph_score for entry in result] == pytest.approx([0.5, 0.5])
    assert [entry.rerank_score for entry in result] == pytest.approx([0.5, 0.5])


def test_graph_truncation_precedes_fields_and_isolated_ties(content):
    nodes = [node for node in content.nodes if node.node_type == GraphNodeType.SKILL]
    nodes[1] = nodes[1].model_copy(update={"name": "quasar", "description": "quasar", "tooling": ["quasar"]})
    schema = normalize_query("quasar")
    seeds = [seed(nodes[0].node_id, 0.9), seed(nodes[1].node_id, 0.1)]
    assert field_score(schema, nodes[1]) > field_score(schema, nodes[0])
    assert rank_joint(nodes, [], seeds, schema, config=config(), limit=1)[0].member_id == "a::csv"
    assert rank_joint(nodes, [], seeds, schema, config=config(), limit=2)[0].member_id == "b"
    tied = rank_joint(
        nodes,
        [],
        [seed(node_id("d"), 0.5), seed(node_id("c"), 0.5)],
        normalize_query("zzzzunmatched"),
        config=config(),
        limit=2,
    )
    assert [item.member_id for item in tied] == ["c", "d"]


def test_field_bonuses_require_candidate_evidence(content):
    node = next(node for node in content.nodes if node.node_type == GraphNodeType.SKILL)
    bare = node.model_copy(
        update={
            "name": "other",
            "description": "",
            "one_line_capability": "",
            "inputs": [],
            "outputs": [],
            "tooling": [],
            "domain_tags": [],
            "example_tasks": [],
            "rendered_snippet": "",
            "metadata": {**node.metadata, "intent": ""},
        }
    )
    schema = QuerySchema(goal="run compile.py", task_name="compile", domain=["manufacturing"], artifacts=["compile.py"])
    assert field_score(schema, bare) == 0
    script = bare.model_copy(update={"rendered_snippet": "Run scripts/compile.py"})
    assert field_score(schema, script) > 0.98
    domain = bare.model_copy(update={"domain_tags": ["manufacturing"]})
    assert field_score(schema, domain) >= 0.35
    named = bare.model_copy(update={"name": "compile.py"})
    assert field_score(schema, named) >= 1.2
    assert field_score(QuerySchema(goal="otherworld"), bare) == 0


def test_full_task_normalization_rule_merge_and_templates():
    task = PublicTask(
        id="excluded-private-dataset-id",
        instruction=" Build   charts ",
        artifacts=["input.csv"],
        constraints=["offline only"],
    )
    text = public_task_text(task)
    assert "input.csv" in text and "offline only" in text and task.id not in text
    schema = normalize_query(
        text, QuerySchema(goal=" charts ", domain=[" data ", "DATA", ""], artifacts=["output.csv"], keywords=["plots"])
    )
    assert schema.goal == "charts" and schema.domain == ["data"] and schema.task_name
    assert schema.artifacts == ["output.csv", "input.csv"]
    assert "offline" in schema.keywords and "plot" in schema.keywords
    assert "input.csv" in query_text(schema, "legacy_query_v1")
    assert "[ARTIFACTS] output.csv" in query_text(schema, "scene_graph_query_v2")
    with pytest.raises(ValueError):
        query_text(schema, "unknown")


@pytest.mark.parametrize(
    "options",
    [{"skill_candidates": 2}, {"scene_seeds": 21}, {"alpha": -1}, {"alpha": float("nan")}, {"aggregation": "sum"}],
)
def test_public_query_options_are_validated(options):
    with pytest.raises(ValidationError):
        config(**options)


def test_unknown_query_profile_is_rejected():
    with pytest.raises(ValidationError, match="query embedding template"):
        config(
            embedding_profile=EmbeddingProfile(
                space_id="fixture", model="fixture", dimensions=3, query_template="unknown"
            )
        )


class PublicTask(Task):
    """Public artifact and constraint extension used for retrieval acceptance."""

    artifacts: list[str]
    """Files supplied publicly to the task executor."""

    constraints: list[str]
    """Public execution restrictions."""
