"""Frozen original prompts, parameter defaults and graph scores."""

import json
from pathlib import Path

import pytest
from mindmemos_skill.algos.manage.skill_manage import prompts
from mindmemos_skill.algos.manage.skill_manage.config import (
    JointLinkingConfig,
    RetrievalStrategyConfig,
    SkillManageConfig,
)
from mindmemos_skill.algos.manage.skill_manage.retrieval import rank_joint
from mindmemos_skill.algos.manage.skill_manage.typing import QuerySchema, RetrievalSeed, SceneGraphContent
from mindmemos_skill.persistence.db_services.graphs.typing import GraphNodeType

FIXTURES = Path(__file__).parent / "fixtures"
BASELINE = json.loads((FIXTURES / "scene_graph_parity.json").read_text())


def test_original_prompt_bytes():
    assert BASELINE["prompts"] == {
        "SSL_SYSTEM": prompts.SSL_SYSTEM,
        "RELATION_SYSTEM": prompts.RELATION_SYSTEM,
        "_JOINT_RELATION_SYSTEM": prompts._JOINT_RELATION_SYSTEM,
        "JOINT_RELATION_SYSTEM": prompts.JOINT_RELATION_SYSTEM,
        "JOINT_RELATION_WEIGHT_SYSTEM": prompts.JOINT_RELATION_WEIGHT_SYSTEM,
        "QUERY_SYSTEM": prompts.QUERY_SYSTEM,
    }


def test_original_algorithm_defaults():
    omitted = {"scene_source", "dependency_match_threshold", "seed_top_k"}
    for name, model in (
        ("SkillManageConfig", SkillManageConfig),
        ("RetrievalStrategyConfig", RetrievalStrategyConfig),
        ("JointLinkingConfig", JointLinkingConfig),
    ):
        for field, expected in BASELINE["defaults"][name].items():
            if name == "SkillManageConfig" and field in omitted:
                continue
            assert model.model_fields[field].get_default() == expected, f"{name}.{field}"


@pytest.mark.parametrize("case", BASELINE["cases"])
def test_original_four_mode_scores(case):
    content = SceneGraphContent.model_validate(json.loads((FIXTURES / "scene_graph_d0.json").read_text())["content"])
    config = SkillManageConfig(
        embedding={"model": "fake", "base_url": "https://example.invalid", "dimensions": 3},
        llm_linking=False,
        **case["options"],
    )
    actual = rank_joint(
        content.nodes,
        content.edges,
        [RetrievalSeed.model_validate(seed) for seed in BASELINE["seeds"]],
        QuerySchema.model_validate(BASELINE["query"]),
        config=config,
        limit=4,
    )
    assert [score.member_id for score in actual] == [score["member_id"] for score in case["scores"]]
    for score, expected in zip(actual, case["scores"], strict=True):
        for field in ("ppr_score", "scene_boost", "graph_score", "rerank_score"):
            assert score.model_dump()[field] == expected[field]


BUILD_BASELINE = json.loads((FIXTURES / "scene_graph_build_parity.json").read_text())


@pytest.mark.parametrize("case", BUILD_BASELINE["cases"])
async def test_original_complete_build_and_model_payloads(case, monkeypatch):
    """Compare native async build output with the original synchronous method output."""
    from mindmemos_skill.agents import AgentResult
    from mindmemos_skill.algos.manage.skill_manage.algorithm import SkillManageAlgorithm
    from mindmemos_skill.persistence.db_services.graphs.typing import SkillGraphNodeRecord
    from mindmemos_skill.scene.typing import ProvidedSceneRecord
    from mindmemos_skill.skill import Skill, SkillContent

    config = SkillManageConfig(
        embedding={"model": "fake", "base_url": "https://example.invalid", "dimensions": 3},
        chat={"model": "fake"},
        build_scene=case["build_scene"],
        scene_llm_linking=True,
        joint_linking={"relation_weighting": case["weighting"]},
    )

    class Store:
        graph_id = "skg-parity"
        scope = {"project_id": "parity-project", "graph_id": graph_id}

        async def resolve_families(self, keys):
            return {key: key for key in keys}

        def bind_node(self, snapshot, key, values):
            return SkillGraphNodeRecord(
                **self.scope,
                graph_version_id=snapshot,
                skill_version_id=f"version-{key}"
                if values.get("node_type", GraphNodeType.SKILL) == GraphNodeType.SKILL
                else None,
                **values,
            )

    class Embeddings:
        def __init__(self):
            self.config = config.embedding
            self.calls = []

        async def embed(self, texts):
            self.calls.append(list(texts))
            return [[1.0, 0.2 if "csv" in value.lower() else 0.8, 0.1] for value in texts]

    class Chat:
        def __init__(self):
            self.config = config.chat
            self.calls = []

        async def run(self, **request):
            self.calls.append(request)
            pairs = json.loads(request["user"])["pairs"]
            relations = []
            if pairs:
                pair = pairs[0]
                relations = [
                    {
                        "source_id": pair[0]["node_id"],
                        "target_id": pair[1]["node_id"],
                        "edge_type": "semantic",
                        "confidence": 0.75,
                        "weight": 0.8,
                        "evidence": "Controlled source-grounded overlap",
                    }
                ]
            return AgentResult(output=json.dumps({"relations": relations}))

    from mindmemos_skill.agents import Agent
    from mindmemos_skill.infra.embedding import EmbeddingModel

    store, embeddings, chat = Store(), Embeddings(), Chat()
    monkeypatch.setattr(EmbeddingModel, "from_config", lambda options: embeddings)
    monkeypatch.setattr(Agent, "from_config", lambda options: chat)
    algorithm = SkillManageAlgorithm(
        config,
        store=store,
        provided={key: ProvidedSceneRecord.model_validate(record) for key, record in BUILD_BASELINE["records"].items()},
    )
    generated = await algorithm._build_joint(
        {key: Skill(SkillContent(content=document)) for key, document in BUILD_BASELINE["documents"].items()},
        "a" * 64,
        None,
    )
    assert [node.model_dump(mode="json", exclude={"graph_version_id"}) for node in generated.nodes] == case["nodes"]
    assert [edge.model_dump(mode="json", exclude={"graph_version_id", "edge_id"}) for edge in generated.edges] == case[
        "edges"
    ]
    assert embeddings.calls == case["embedding_calls"]
    assert chat.calls == case["chat_calls"]


def test_original_scene_schemas_and_enrichment_prompt():
    from mindmemos_skill.algos.manage.skill_manage.enrichment import ENRICHMENT_SYSTEM
    from mindmemos_skill.algos.manage.skill_manage.typing import SkillEnrichment
    from mindmemos_skill.scene.typing import ProvidedSceneRecord, SceneExtractionResponse

    assert ENRICHMENT_SYSTEM == BASELINE["enrichment_prompt"]
    assert {
        "ProvidedSceneRecord": ProvidedSceneRecord.model_json_schema(),
        "SceneExtractionResponse": SceneExtractionResponse.model_json_schema(),
        "SkillEnrichment": SkillEnrichment.model_json_schema(),
    } == BASELINE["schemas"]
