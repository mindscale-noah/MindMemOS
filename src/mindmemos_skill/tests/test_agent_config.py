"""Verify named Agent configuration across service loading and algorithm boundaries."""

from pathlib import Path
from typing import Annotated

import pytest
import yaml
from mindmemos_skill.agents import AgentCatalog, AgentConfig, CodexConfig, OpenAIChatConfig
from mindmemos_skill.algos.train.treeskill.config import TreeSkillTrainerConfig
from mindmemos_skill.config.loader import load_skill_config, resolve_service_config
from mindmemos_skill.config.schema import skill_config_schema
from pydantic import BaseModel, ValidationError

EXAMPLE = Path(__file__).resolve().parents[3] / "config/mindmemos_skill/dev.example.yaml"
"""Public service config exercising every current algorithm's named Agent fields."""


def test_presets_expand_through_all_algorithm_and_pipeline_boundaries():
    raw = yaml.safe_load(EXAMPLE.read_text())
    deployment = load_skill_config(EXAMPLE.read_text())
    service = deployment.service
    tree = service.training_pipelines["treeskill"].config.trainer.config
    forest = service.training_pipelines["skill_forest"].config.trainer.config
    evolution = service.evolution_pipelines["simple"].config.evolver.config
    assert tree.optimizer == service.agents["treeskill_qwen3_5_thinking"].config
    assert tree.failure_optimizer == tree.router == service.agents["treeskill_qwen3_5_fast"].config
    assert forest.optimizer.model == service.agents["skill_forest_gpt_5_4_mini"].config.model
    assert evolution.optimizer == service.agents["simple_evolve_gpt_5_4_mini"].config
    assert (
        service.manage_pipeline.config.algorithm.chat
        == service.agents[raw["service"]["manage_pipeline"]["config"]["algorithm"]["chat"]].config
    )
    assert tree.failure_optimizer is not tree.router
    tree.failure_optimizer.extra_body["demo_mutation"] = True
    assert "demo_mutation" not in tree.router.extra_body
    assert "demo_mutation" not in service.agents["treeskill_qwen3_5_fast"].config.extra_body
    assert "demo_mutation" not in raw["service"]["agents"]["treeskill_qwen3_5_fast"]["config"]["extra_body"]


def test_expanded_runtime_options_survive_reconstruction_without_names():
    deployment = load_skill_config(EXAMPLE.read_text())
    expanded = deployment.service.model_dump(mode="json")
    assert isinstance(expanded["training_pipelines"]["treeskill"]["config"]["trainer"]["config"]["router"], dict)
    deployment.service.agents.clear()
    reconstructed = resolve_service_config(deployment.service)
    assert reconstructed.training_pipelines["treeskill"].config.trainer.config.router.model == "qwen3.5-9b"


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("treeskill_missing", "Unknown Agent preset"),
        ("skill_forest_gpt_5_4_mini", "must belong to algorithm 'treeskill'"),
        ("treeskill_codex", "allowed backends"),
    ],
)
def test_invalid_names_backends_and_cross_algorithm_references_fail_in_preflight(name, message):
    value = yaml.safe_load(EXAMPLE.read_text())
    value["service"]["agents"]["treeskill_codex"] = {"class_name": "codex", "config": {"model": "offline"}}
    value["service"]["training_pipelines"]["treeskill"]["config"]["trainer"]["config"]["optimizer"] = name
    with pytest.raises(ValidationError, match=message):
        load_skill_config(yaml.safe_dump(value))


def test_inline_config_and_none_fallback_work_without_a_catalog():
    config = TreeSkillTrainerConfig(
        optimizer=OpenAIChatConfig(model="inline"),
        failure_optimizer=None,
        router={"model": "inline-router"},
    )
    assert config.optimizer.model == "inline"
    assert config.failure_optimizer is None
    with pytest.raises(ValidationError, match="service agent catalog"):
        TreeSkillTrainerConfig(optimizer="treeskill_inline", router={"model": "inline"})


def test_general_agent_config_preserves_backend_identity_for_multiple_allowed_types():
    class MultiBackendConfig(BaseModel):
        """Exercise backend selection separately from Chat-only production algorithms."""

        agent: Annotated[
            OpenAIChatConfig | CodexConfig, AgentConfig(allowed=("openai_chat", "codex"), namespace="demo")
        ]
        """One concrete Chat or Codex configuration selected by its preset name."""

    catalog = AgentCatalog(
        agents={
            "demo_chat": {"class_name": "openai_chat", "config": {"model": "chat"}},
            "demo_codex": {"class_name": "codex", "config": {"model": "codex", "extra_args": ["--offline"]}},
        }
    )
    chat = MultiBackendConfig.model_validate({"agent": "demo_chat"}, context=catalog)
    codex = MultiBackendConfig.model_validate({"agent": "demo_codex"}, context=catalog)
    assert type(chat.agent) is OpenAIChatConfig
    assert type(codex.agent) is CodexConfig
    codex.agent.extra_args.append("--changed")
    assert catalog.agents["demo_codex"].config.extra_args == ["--offline"]


def test_unused_invalid_presets_fail_before_resource_construction():
    value = yaml.safe_load(EXAMPLE.read_text())
    value["service"]["agents"]["unused_invalid"] = {
        "class_name": "openai_chat",
        "config": {"model": "offline", "temperature": 3},
    }
    with pytest.raises(ValidationError, match="temperature"):
        load_skill_config(yaml.safe_dump(value))
    value["service"]["agents"]["unused_invalid"] = {"class_name": "unregistered_backend", "config": {}}
    with pytest.raises(ValidationError, match="Unknown Agent"):
        load_skill_config(yaml.safe_dump(value))


def test_presets_do_not_chain_names_or_accept_override_objects():
    with pytest.raises(ValidationError):
        AgentCatalog(agents={"treeskill_alias": {"class_name": "openai_chat", "config": "treeskill_other"}})
    with pytest.raises(ValidationError):
        TreeSkillTrainerConfig(optimizer={"ref": "treeskill_other", "temperature": 0.5}, router={"model": "offline"})


@pytest.mark.parametrize("name", ["shared", "treeskill bad", "treeskill_", "TreeSkill_chat"])
def test_preset_names_require_algorithm_and_llm_name(name):
    with pytest.raises(ValidationError, match="pattern"):
        AgentCatalog(agents={name: {"class_name": "openai_chat", "config": {"model": "offline"}}})


def test_current_algorithm_schema_declares_chat_only_agent_fields():
    schema = skill_config_schema()
    for model, fields in {
        "TreeSkillTrainerConfig": ("optimizer", "failure_optimizer", "router"),
        "SkillForestTrainerConfig": ("optimizer",),
        "SimpleEvolverConfig": ("optimizer",),
        "SkillManageConfig": ("chat",),
    }.items():
        for field in fields:
            definition = schema["$defs"][model]["properties"][field]
            branches = definition.get("anyOf", [definition])
            agent = next((branch for branch in branches if "x-agent-backends" in branch), definition)
            assert agent["x-agent-backends"] == ["openai_chat"]
            assert "x-agent-namespace" in agent


def test_service_example_uses_only_algorithm_owned_named_references():
    raw = yaml.safe_load(EXAMPLE.read_text())["service"]
    consumed = set()
    for pipeline in raw["training_pipelines"].values():
        trainer = pipeline["config"]["trainer"]
        for field in ("optimizer", "failure_optimizer", "router"):
            if field in trainer["config"]:
                name = trainer["config"][field]
                if isinstance(name, str):
                    assert name.startswith(trainer["class_name"] + "_")
                    consumed.add(name)
                else:
                    assert name.get("class_name", "openai_chat") == "openai_chat"
    name = raw["evolution_pipelines"]["simple"]["config"]["evolver"]["config"]["optimizer"]
    assert name.startswith("simple_evolve_")
    consumed.add(name)
    name = raw["manage_pipeline"]["config"]["algorithm"]["chat"]
    assert name.startswith("skill_manage_")
    consumed.add(name)
    assert consumed <= raw["agents"].keys()


@pytest.mark.parametrize("backend", ["openai_chat", "codex", "claude", "deepseek_harness"])
def test_shared_agent_builder_uses_backend_identity_and_creates_independent_instances(backend):
    from mindmemos_skill.agents import Agent, AgentDefinition

    definition = AgentDefinition(class_name=backend, config={"model": "offline"})
    config = definition.config
    restored = type(config).model_validate_json(config.model_dump_json())
    first = Agent.from_config(restored)
    second = Agent.from_config(restored)
    assert type(first) is Agent.resolve(backend)
    assert first is not second
    assert first.config.class_name == backend


def test_inline_multi_backend_selection_survives_serialization_without_a_catalog():
    from mindmemos_skill.agents import Agent

    class Options(BaseModel):
        """Select one supported backend explicitly in inline configuration."""

        agent: Annotated[OpenAIChatConfig | CodexConfig, AgentConfig(allowed=("openai_chat", "codex"))]
        """Backend settings with an explicit registration name."""

    for backend in ("openai_chat", "codex"):
        options = Options(agent={"class_name": backend, "model": "offline"})
        restored = Options.model_validate_json(options.model_dump_json())
        assert restored.agent.class_name == backend
        assert type(Agent.from_config(restored.agent)) is Agent.resolve(backend)
    with pytest.raises(ValidationError, match="requires class_name"):
        Options(agent={"model": "ambiguous"})


@pytest.mark.parametrize("value", [{"class_name": "codex", "model": "offline"}, CodexConfig(model="offline")])
def test_inline_backends_cannot_bypass_algorithm_whitelist(value):
    with pytest.raises(ValidationError, match="allowed backends"):
        TreeSkillTrainerConfig(optimizer=value, router={"model": "offline"})


def test_agent_definition_rejects_conflicting_backend_identity():
    from mindmemos_skill.agents import AgentDefinition

    with pytest.raises(ValidationError):
        AgentDefinition(class_name="openai_chat", config={"class_name": "codex", "model": "offline"})


def test_shared_connection_settings_cannot_construct_an_agent():
    from mindmemos_skill.agents import Agent
    from mindmemos_skill.agents.config import AgentConnectionConfig

    with pytest.raises(ValueError, match="requires class_name"):
        Agent.from_config(AgentConnectionConfig())


def test_algorithms_reject_constructor_factory_overrides():
    from mindmemos_skill.algos.evolve.simple_evolve import SimpleEvolverConfig, SimpleSkillEvolver
    from mindmemos_skill.algos.train.skill_forest import SkillForestTrainer, SkillForestTrainerConfig
    from mindmemos_skill.algos.train.treeskill import TreeSkillTrainer

    for implementation, config in (
        (SkillForestTrainer, SkillForestTrainerConfig(optimizer={"model": "offline"})),
        (TreeSkillTrainer, TreeSkillTrainerConfig(optimizer={"model": "offline"}, router={"model": "offline"})),
        (SimpleSkillEvolver, SimpleEvolverConfig(optimizer={"model": "offline"})),
    ):
        with pytest.raises(TypeError):
            implementation(config, agent_factory=lambda settings: None)


def test_registered_agents_can_share_a_config_model_with_explicit_backend_selection(monkeypatch):
    from mindmemos_skill.agents import Agent, AgentDefinition
    from mindmemos_skill.agents.config import AgentConnectionConfig

    monkeypatch.setattr(Agent, "_registry", Agent.registered_types())

    @Agent.register("test_first")
    class First(Agent[AgentConnectionConfig]):
        """First backend sharing the common configuration model."""

        config_type = AgentConnectionConfig
        """Connection and explicit backend identity used by this test."""

    @Agent.register("test_second")
    class Second(Agent[AgentConnectionConfig]):
        """Second backend sharing exactly the same configuration model."""

        config_type = AgentConnectionConfig
        """Connection and explicit backend identity used by this test."""

    class Options(BaseModel):
        """Validate a shared model while restricting its selectable backends."""

        agent: Annotated[AgentConnectionConfig, AgentConfig(allowed=("test_first", "test_second"))]
        """Explicit selection among two implementations sharing one model."""

    for name, implementation in (("test_first", First), ("test_second", Second)):
        definition = AgentDefinition(class_name=name)
        options = Options(agent=definition.config)
        restored = Options.model_validate_json(options.model_dump_json())
        assert type(Agent.from_config(restored.agent)) is implementation


def test_skill_manage_presets_expand_and_reconstruct_without_a_catalog():
    value = yaml.safe_load(EXAMPLE.read_text())
    value["service"]["agents"]["skill_manage_offline"] = {
        "class_name": "openai_chat",
        "config": {"model": "offline"},
    }
    value["service"]["manage_pipeline"]["config"]["algorithm"]["chat"] = "skill_manage_offline"
    deployment = load_skill_config(yaml.safe_dump(value))
    algorithm = deployment.service.manage_pipeline.config.algorithm
    assert algorithm.chat.class_name == "openai_chat"
    assert algorithm.chat.model == "offline"
    assert algorithm.embedding.class_name == "openai_embedding"
    deployment.service.agents.clear()
    restored = resolve_service_config(deployment.service).manage_pipeline.config.algorithm
    assert restored.chat == algorithm.chat
    assert restored.embedding == algorithm.embedding


@pytest.mark.parametrize("named", [True, False])
def test_skill_manage_chat_whitelist_rejects_unsupported_backend(named):
    value = yaml.safe_load(EXAMPLE.read_text())
    chat = {"class_name": "codex", "config": {"model": "offline"}}
    if named:
        value["service"]["agents"]["skill_manage_codex"] = chat
        selected = "skill_manage_codex"
    else:
        selected = {"class_name": "codex", "model": "offline"}
    value["service"]["manage_pipeline"]["config"]["algorithm"]["chat"] = selected
    with pytest.raises(ValidationError, match="allowed backends"):
        load_skill_config(yaml.safe_dump(value))


def test_skill_manage_models_are_created_from_config_and_cannot_be_injected(monkeypatch):
    from mindmemos_skill.agents import Agent
    from mindmemos_skill.algos.manage.skill_manage import SkillManageAlgorithm, SkillManageConfig
    from mindmemos_skill.infra.embedding import EmbeddingModel

    settings = SkillManageConfig(
        embedding={"model": "offline", "dimensions": 2, "base_url": "http://test.invalid/v1"},
        chat={"model": "offline"},
    )
    calls = []
    chat_model, embedding_model = object(), object()

    def chat(config):
        calls.append(("chat", config))
        return chat_model

    def embedding(config):
        calls.append(("embedding", config))
        return embedding_model

    monkeypatch.setattr(Agent, "from_config", chat)
    monkeypatch.setattr(EmbeddingModel, "from_config", embedding)
    algorithm = SkillManageAlgorithm(settings)
    assert algorithm.chat_model is chat_model
    assert algorithm.embedding_model is embedding_model
    assert calls == [("embedding", settings.embedding), ("chat", settings.chat)]
    for dependencies in ({"chat_model": chat_model}, {"embedding_model": embedding_model}):
        with pytest.raises(TypeError):
            SkillManageAlgorithm(settings, **dependencies)
    calls.clear()
    rules_only = SkillManageAlgorithm(settings.model_copy(update={"chat": None, "llm_linking": False}))
    assert rules_only.chat_model is None
    assert calls == [("embedding", settings.embedding)]
