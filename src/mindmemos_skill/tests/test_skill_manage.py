"""Graph management integration with real SQLite and deterministic model doubles."""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from mindmemos_skill.agents import AgentResult, OpenAIChatConfig
from mindmemos_skill.algos import Algorithm
from mindmemos_skill.algos.manage.skill_manage import (
    SceneRecord,
    SkillManageAlgorithm,
    SkillManageConfig,
    SkillManageSkillSet,
)
from mindmemos_skill.algos.manage.skill_manage.algorithm import node_id
from mindmemos_skill.envs import Environment, Reward, RolloutResult, Task
from mindmemos_skill.envs.data import RewardData
from mindmemos_skill.infra.database import Database, SQLiteConfig, SQLiteDatabase
from mindmemos_skill.infra.embedding import OpenAIEmbeddingConfig
from mindmemos_skill.persistence import SkillVersionRecord, create_persistence, ensure_collection
from mindmemos_skill.persistence.constants import (
    SKILL_GRAPH_EDGES_TABLE,
    SKILL_GRAPH_NODES_TABLE,
    SKILL_GRAPH_VERSIONS_TABLE,
    SKILL_GRAPHS_TABLE,
    SKILL_VERSIONS_TABLE,
    SKILLS_TABLE,
)
from mindmemos_skill.persistence.db_services.skill_management.writer import SkillGraphStore
from mindmemos_skill.persistence.db_services.skills.typing import SkillRecord
from mindmemos_skill.skill import Skill, SkillContent
from pydantic import BaseModel


class Embeddings:
    def __init__(self, config):
        self.config = config
        self.calls = []
        self.fail = False

    async def embed(self, texts):
        if self.fail:
            raise RuntimeError("Embedding failed")
        self.calls.append(list(texts))
        return [[1.0, 0.0, 0.0] if "csv" in text.lower() else [0.0, 1.0, 0.0] for text in texts]


class Chat:
    def __init__(self, config):
        self.config = config
        self.calls = []
        self.fail = False

    async def run(self, *, system, user, response_format):
        if self.fail:
            raise RuntimeError("Chat failed")
        payload = json.loads(user)
        self.calls.append(payload)
        if "pairs" in payload:
            pair = payload["pairs"][0]
            response = {
                "relations": [
                    {
                        "source_id": pair[0]["node_id"],
                        "target_id": pair[1]["node_id"],
                        "edge_type": "semantic",
                        "weight": 0.4,
                        "evidence": "Related data tools",
                    }
                ]
            }
        else:
            response = {"goal": payload["instruction"], "keywords": ["csv"]}
        return AgentResult(output=json.dumps(response))


@pytest.fixture
async def setup(tmp_path):
    database = SQLiteDatabase(SQLiteConfig(url=f"sqlite:///{tmp_path / 'graph.db'}"))
    persistence = create_persistence(database)
    (await ensure_collection(persistence))
    skills = {}
    for key, desc, io in [
        ("a", "Parse CSV tables", "outputs: [table_records]"),
        ("b", "Filter records", "inputs: [table_records]"),
        ("c", "Draw image pictures", "outputs: [pictures]"),
    ]:
        skill = Skill(
            SkillContent(
                content=f"---\nname: Same name\ndescription: {desc}\n{io}\n---\n{desc}",
                name="test-skill",
                description="Test instructions",
                standard_format=False,
            )
        )
        (
            await persistence.skills.add_skill(
                skill,
                SkillVersionRecord(
                    project_id="p",
                    skill_id=key,
                    version_id=f"v-{key}",
                    **skill.snapshot().model_dump(),
                ),
            )
        )
        (
            await database.insert(
                SKILLS_TABLE,
                [
                    SkillRecord(
                        project_id="p", skill_id=key, name="Same name", active_version_id=f"v-{key}"
                    ).model_dump()
                ],
            )
        )
        skills[key] = skill
    store = SkillGraphStore(
        database, project_id="p", graph_id="g", actor_id="test", version_ids={key: f"v-{key}" for key in skills}
    )
    config = SkillManageConfig(
        embedding=OpenAIEmbeddingConfig(
            model="test-embedding", base_url="http://test.invalid/v1", dimensions=3, api_key="secret"
        ),
        chat=OpenAIChatConfig(model="test-chat", api_key="secret", base_url="http://test.invalid/v1"),
        llm_batch_size=1,
    )
    scenes = {
        "a": SceneRecord.model_validate(
            {
                "skill": {"name": "ignored display"},
                "logic_steps": [],
                "scenes": [
                    {
                        "scene_id": "parse",
                        "scene_name": "CSV parse",
                        "scene_goal": "Load CSV",
                        "next_scene_rules": [{"next_scene_id": "check"}],
                    },
                    {"scene_id": "check", "scene_name": "Check rows", "scene_goal": "Validate CSV records"},
                ],
            }
        )
    }
    embedding = Embeddings(config.embedding)
    chat = Chat(config.chat)

    def manager(**updates):
        settings = config.model_copy(update=updates)
        return SkillManageAlgorithm(
            settings, store=store, scene_records=scenes, embedding_model=embedding, chat_model=chat
        )

    yield database, persistence, skills, store, config, embedding, chat, manager
    (await database.close())


async def test_registered_end_to_end_build_scene_link_rewrite_and_id_identity(setup):
    database, persistence, skills, store, config, embedding, chat, factory = setup
    assert Algorithm.resolve("skill_manage") is SkillManageAlgorithm
    manager = factory()
    collection = SkillManageSkillSet(skills, algorithm=manager, limit=2, max_concurrency=2)
    assert not embedding.calls and not chat.calls
    selected = await collection.select(Task(id="task", instruction="Parse CSV records"))
    assert len(selected) == len(set(selected)) == 2 and set(selected) <= set(skills)
    assert (await database.count(SKILL_GRAPH_NODES_TABLE)) == 5
    nodes = await database.select(SKILL_GRAPH_NODES_TABLE)
    assert len({node["node_id"] for node in nodes}) == 5
    assert {node["name"] for node in nodes if node["node_type"] == "skill"} == {"Same name"}
    edges = await database.select(SKILL_GRAPH_EDGES_TABLE)
    assert {edge["edge_type"] for edge in edges} >= {"dependency", "semantic", "containment", "flow"}
    dependency = next(edge for edge in edges if edge["edge_type"] == "dependency")
    assert (dependency["source_node_id"], dependency["target_node_id"]) == (node_id("a"), node_id("b"))
    assert len([call for call in chat.calls if "pairs" in call]) == 3
    assert chat.calls[-1]["instruction"] == "Parse CSV records"
    assert embedding.calls[-1] == ["Parse CSV records\ncsv"]
    assert "secret" in json.dumps((await database.select(SKILL_GRAPH_VERSIONS_TABLE)), default=str)
    assert (await database.count(SKILL_VERSIONS_TABLE)) == 3
    assert (await database.count(SKILL_GRAPH_VERSIONS_TABLE)) == 1
    calls = len(embedding.calls)
    restored = factory()
    (await restored.organize(skills=skills))
    assert len(embedding.calls) == calls
    assert (await database.count(SKILL_GRAPH_VERSIONS_TABLE)) == 1
    assert (await restored.recommend(task=Task(id="again", instruction="Parse CSV records"), limit=2)) == selected


async def test_disabled_model_features_do_not_call_chat(setup):
    database, _, skills, _, _, embedding, chat, factory = setup
    manager = factory(llm_linking=False, query_rewrite=False, build_scene=False, mode="skill_only")
    chat.fail = True
    (await manager.organize(skills=skills))
    assert await manager.recommend(task=Task(id="task", instruction="CSV"))
    assert not chat.calls
    assert (await database.count(SKILL_GRAPH_NODES_TABLE)) == 3


async def test_atomic_interruption_rolls_back_and_retry_preserves_existing_versions(setup, monkeypatch):
    database, _, skills, _, _, _, _, factory = setup
    original = Database.update

    def interrupted(self, table, values, *, filters):
        if table == SKILL_GRAPHS_TABLE and "current_version_id" in values:
            raise RuntimeError("Interrupted before publication")
        return original(self, table, values, filters=filters)

    monkeypatch.setattr(Database, "update", interrupted)
    manager = factory()
    with pytest.raises(RuntimeError, match="Interrupted"):
        (await manager.organize(skills=skills))
    for table in (
        SKILL_GRAPHS_TABLE,
        SKILL_GRAPH_VERSIONS_TABLE,
        SKILL_GRAPH_NODES_TABLE,
        SKILL_GRAPH_EDGES_TABLE,
    ):
        assert (await database.count(table)) == 0
    assert (await database.count(SKILL_VERSIONS_TABLE)) == 3
    with pytest.raises(RuntimeError, match="Organize"):
        (await manager.recommend(task=Task(id="task", instruction="CSV")))
    monkeypatch.setattr(Database, "update", original)
    (await manager.organize(skills=skills))
    assert (await database.count(SKILL_GRAPH_VERSIONS_TABLE)) == 1
    assert (await database.count(SKILL_VERSIONS_TABLE)) == 3


async def test_failed_new_build_preserves_previous_published_snapshot(setup):
    database, _, skills, _, _, embedding, _, factory = setup
    (await factory().organize(skills=skills))
    old = (await database.select(SKILL_GRAPHS_TABLE))[0]
    embedding.fail = True
    manager = factory(mode="skill_only")
    with pytest.raises(RuntimeError, match="Embedding failed"):
        (await manager.organize(skills=skills))
    assert (await database.select(SKILL_GRAPHS_TABLE))[0] == old
    assert (await database.count(SKILL_GRAPH_VERSIONS_TABLE)) == 1


async def test_project_bindings_and_content_cannot_expand_scope(setup):
    database, _, skills, _, config, embedding, chat, factory = setup
    changed = dict(skills)
    changed["a"] = Skill(
        SkillContent(content="different", name="test-skill", description="Test instructions", standard_format=False)
    )
    with pytest.raises(ValueError, match="differs"):
        (await factory().organize(skills=changed))
    bad_store = SkillGraphStore(
        database, project_id="other", graph_id="g", actor_id="test", version_ids={key: f"v-{key}" for key in skills}
    )
    manager = SkillManageAlgorithm(
        config, store=bad_store, scene_records={}, embedding_model=embedding, chat_model=chat
    )
    with pytest.raises(Exception, match="project"):
        (await manager.organize(skills=skills))
    assert not embedding.calls and not chat.calls


async def test_corrupt_persisted_graph_is_rejected(setup):
    database, _, skills, _, _, _, _, factory = setup
    (await factory().organize(skills=skills))
    (
        await database.update(
            SKILL_GRAPH_NODES_TABLE, {"description": "corrupted"}, filters={"project_id": "p", "node_id": node_id("a")}
        )
    )
    with pytest.raises(ValueError, match="integrity"):
        (await factory().organize(skills=skills))


async def test_concurrent_identical_builds_publish_only_once(setup):
    database, _, skills, _, _, _, _, factory = setup
    await asyncio.gather(*(factory().organize(skills=skills) for _ in range(2)))
    assert (await database.count(SKILL_GRAPH_VERSIONS_TABLE)) == 1
    assert (await database.count(SKILL_GRAPH_VERSIONS_TABLE)) == 1


class Options(BaseModel):
    pass


class Env(Environment[Options]):
    config_type = Options

    async def _run(self, *, task, reward_data, skills, workspace):
        return RolloutResult(
            output=[skill.content for skill in skills],
            trajectory=[],
            reward=Reward(score=1, success=True, detail="OK"),
        )


async def test_skillset_environment_resume_does_not_call_models(setup, tmp_path):
    _, _, skills, _, _, embedding, chat, factory = setup
    collection = SkillManageSkillSet(skills, algorithm=factory(), limit=2, max_concurrency=2)
    env = Env(Options())
    task = Task(id="task", instruction="Parse CSV")
    result = await env.rollout(task, tmp_path / "rollout", reward_data=RewardData(), skill_set=collection)
    embedding.fail = chat.fail = True
    new_collection = SkillManageSkillSet(skills, algorithm=factory(), limit=2, max_concurrency=2)
    assert (await env.rollout(task, tmp_path / "rollout", reward_data=RewardData(), skill_set=new_collection)) == result


async def test_empty_scope_uses_no_models(setup):
    database, _, _, _, config, embedding, chat, _ = setup
    store = SkillGraphStore(database, project_id="p", graph_id="empty", actor_id="test", version_ids={})
    manager = SkillManageAlgorithm(config, store=store, scene_records={}, embedding_model=embedding, chat_model=chat)
    (await manager.organize(skills={}))
    assert (await manager.recommend(task=Task(id="task", instruction="CSV"))) == []
    assert not embedding.calls and not chat.calls


async def test_scene_vector_seed_changes_recommended_owner(setup):
    database, _, skills, store, config, embedding, chat, _ = setup
    scenes = {
        "b": SceneRecord.model_validate(
            {"scenes": [{"scene_id": "special", "scene_name": "special", "scene_goal": "special scenario"}]}
        )
    }

    async def embed(texts):
        return [[0, 0, 1] if "special" in text else [1, 0, 0] for text in texts]

    embedding.embed = embed
    settings = config.model_copy(
        update={"llm_linking": False, "query_rewrite": False, "seed_top_k": 1, "score_threshold": 0.9}
    )
    manager = SkillManageAlgorithm(
        settings, store=store, scene_records=scenes, embedding_model=embedding, chat_model=chat
    )
    (await manager.organize(skills=skills))
    assert (await manager.recommend(task=Task(id="task", instruction="special"), limit=1)) == ["b"]


async def test_durable_reload_uses_vectors_without_rebuilding(setup, tmp_path):
    _, _, skills, _, config, embedding, chat, factory = setup
    (await factory().organize(skills=skills))
    database = SQLiteDatabase(SQLiteConfig(url=f"sqlite:///{tmp_path / 'graph.db'}"))
    create_persistence(database)
    try:
        store = SkillGraphStore(
            database, project_id="p", graph_id="g", actor_id="test", version_ids={key: f"v-{key}" for key in skills}
        )
        original = factory()
        manager = SkillManageAlgorithm(
            config, store=store, scene_records=original.scene_records, embedding_model=embedding, chat_model=chat
        )
        embedding.fail = chat.fail = True
        (await manager.organize(skills=skills))
        assert (await database.count(SKILL_GRAPH_VERSIONS_TABLE)) == 1
    finally:
        (await database.close())


async def test_invalid_llm_endpoints_never_publish(setup, monkeypatch):
    database, _, skills, _, _, _, chat, factory = setup

    def run(**kwargs):
        return AgentResult(
            output=json.dumps(
                {
                    "relations": [
                        {
                            "source_id": "foreign",
                            "target_id": node_id("a"),
                            "edge_type": "semantic",
                            "weight": 0.8,
                            "evidence": "bad",
                        }
                    ]
                }
            )
        )

    monkeypatch.setattr(chat, "run", AsyncMock(side_effect=run))
    with pytest.raises(ValueError, match="outside"):
        (await factory().organize(skills=skills))
    assert (await database.count(SKILL_GRAPHS_TABLE)) == 0


async def test_new_configuration_adds_snapshot_without_rewriting_history(setup):
    database, _, skills, _, _, _, _, factory = setup
    (await factory().organize(skills=skills))
    old_nodes = await database.select(SKILL_GRAPH_NODES_TABLE)
    (await factory(mode="skill_only").organize(skills=skills))
    assert (await database.count(SKILL_GRAPH_VERSIONS_TABLE)) == 2
    assert (await database.count(SKILL_GRAPH_VERSIONS_TABLE)) == 2
    current_nodes = await database.select(SKILL_GRAPH_NODES_TABLE)
    assert all(row in current_nodes for row in old_nodes)


def test_pagerank_matches_source_frozen_golden():
    from types import SimpleNamespace

    from mindmemos_skill.algos.manage.skill_manage.retrieval import pagerank

    golden = json.loads((Path(__file__).parent / "fixtures/skill_manage/golden.json").read_text())
    raw_nodes = sorted((node for node in golden["nodes"] if node["type"] == "Skill"), key=lambda node: node["name"])
    nodes = [SimpleNamespace(node_id=node["name"]) for node in raw_nodes]
    names = {node.node_id for node in nodes}
    edges = [
        SimpleNamespace(
            source_node_id=edge["source"], target_node_id=edge["target"], edge_type=edge["type"], weight=edge["weight"]
        )
        for edge in golden["edges"]
        if edge["source"] in names and edge["target"] in names
    ]
    scores = pagerank(nodes, edges, [seed["name"] for seed in golden["seeds"]])
    for name, value in scores.items():
        assert value == pytest.approx(golden["ppr"]["scores"][name], rel=1e-7, abs=1e-8)


async def test_scope_replacement_and_empty_scope_never_expand_from_bindings(setup):
    database, _, skills, store, config, embedding, chat, _ = setup
    settings = config.model_copy(update={"build_scene": False, "llm_linking": False, "query_rewrite": False})
    manager = SkillManageAlgorithm(settings, store=store, embedding_model=embedding, chat_model=chat)
    (await manager.organize(skills=skills))
    (await manager.organize(skills={"c": skills["c"]}))
    assert (await manager.recommend(task=Task(id="task", instruction="CSV"))) == ["c"]
    calls = len(embedding.calls)
    (await manager.organize(skills={}))
    assert (await manager.recommend(task=Task(id="task", instruction="CSV"))) == []
    assert len(embedding.calls) == calls
    assert (await database.count(SKILL_GRAPH_VERSIONS_TABLE)) == 3


async def test_failed_scope_replacement_invalidates_recommendation(setup):
    database, _, skills, _, _, _, _, factory = setup
    manager = factory()
    (await manager.organize(skills=skills))
    changed = dict(skills)
    changed["a"] = Skill(
        SkillContent(content="changed", name="test-skill", description="Test instructions", standard_format=False)
    )
    with pytest.raises(ValueError, match="differs"):
        (await manager.organize(skills=changed))
    with pytest.raises(RuntimeError, match="Organize"):
        (await manager.recommend(task=Task(id="task", instruction="CSV")))
    assert (await database.count(SKILL_GRAPH_VERSIONS_TABLE)) == 1


async def test_republishing_historical_build_selects_its_existing_snapshot(setup):
    """Reusing a historical build advances the head without duplicating history."""
    database, _, skills, _, _, _, _, factory = setup
    original = factory()
    (await original.organize(skills=skills))
    first = original.snapshot_id
    different = factory(mode="skill_only")
    (await different.organize(skills=skills))
    assert different.snapshot_id != first
    repeated = factory()
    (await repeated.organize(skills=skills))
    assert repeated.snapshot_id == first
    assert (await database.get(SKILL_GRAPHS_TABLE, {"project_id": "p", "graph_id": "g"}))["current_version_id"] == first
    assert (await database.count(SKILL_GRAPH_VERSIONS_TABLE)) == 2


async def test_interrupted_historical_republication_preserves_current_head(setup, monkeypatch):
    """A failed cached-head switch leaves the current graph and history intact."""
    database, _, skills, _, _, _, _, factory = setup
    (await factory().organize(skills=skills))
    current = factory(mode="skill_only")
    (await current.organize(skills=skills))
    original_update = Database.update

    def interrupt(self, table, values, *, filters):
        if table == SKILL_GRAPHS_TABLE and "current_version_id" in values:
            raise RuntimeError("Interrupted cached publication")
        return original_update(self, table, values, filters=filters)

    monkeypatch.setattr(Database, "update", interrupt)
    with pytest.raises(RuntimeError, match="Interrupted cached"):
        (await factory().organize(skills=skills))
    assert (await database.get(SKILL_GRAPHS_TABLE, {"project_id": "p", "graph_id": "g"}))[
        "current_version_id"
    ] == current.snapshot_id
    assert (await database.count(SKILL_GRAPH_VERSIONS_TABLE)) == 2
