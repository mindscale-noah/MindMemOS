"""Shared configuration contracts for Skill Manage and existing algorithms."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from mindmemos_skill.config import SkillConfigCompiler
from mindmemos_skill.errors import SkillConfigurationError
from mindmemos_skill.infra.vector_store import BackendRegistry
from mindmemos_skill.registry import ComponentRequirements, ComponentType, get_component


def _joint_config(tmp_path: Path) -> dict:
    return {
        "local": {
            "root_dir": str(tmp_path / "skill"),
            "database": {"provider": "postgres"},
            "vector_store": {"provider": "pgvector"},
        },
        "runtime": {
            "models": {
                "chat": {"model": "openai/test-chat"},
                "embedding": {"model": "openai/test-embedding", "kind": "embedding", "dimension": 3},
            },
            "algorithms": {
                "evolution": {
                    "type": "offline_evolution",
                    "model_roles": {"chat": "chat", "embedding": "embedding"},
                },
                "graphs": {"type": "skill_manage", "model_roles": {"embedding": "embedding"}},
            },
        },
    }


def test_offline_evolution_and_skill_manage_share_embedding_model(tmp_path: Path) -> None:
    compiled = SkillConfigCompiler().compile(_joint_config(tmp_path))

    assert compiled.runtime.algorithms["evolution"].type == "offline_evolution"
    assert compiled.runtime.algorithms["graphs"].type == "skill_manage"
    assert compiled.runtime.algorithms["evolution"].model_roles["embedding"] == "embedding"
    assert compiled.runtime.algorithms["graphs"].model_roles["embedding"] == "embedding"
    assert compiled.runtime.models["embedding"].kind == "embedding"
    assert not (tmp_path / "skill").exists()


def test_offline_evolution_preserves_legacy_model_config_without_kind(tmp_path: Path) -> None:
    config = _joint_config(tmp_path)
    config["runtime"]["models"]["legacy_embedding"] = {"model": "openai/test-embedding"}
    config["runtime"]["algorithms"]["evolution"]["model_roles"]["embedding"] = "legacy_embedding"

    compiled = SkillConfigCompiler().compile(config)

    assert compiled.runtime.algorithms["evolution"].model_roles["embedding"] == "legacy_embedding"


@pytest.mark.parametrize("role", ["chat", "embedding"])
def test_offline_evolution_still_requires_untyped_roles(tmp_path: Path, role: str) -> None:
    config = _joint_config(tmp_path)
    del config["runtime"]["algorithms"]["evolution"]["model_roles"][role]

    with pytest.raises(SkillConfigurationError, match=f"missing model roles: {role}"):
        SkillConfigCompiler().compile(config)


def test_skill_manage_still_rejects_chat_model_for_embedding(tmp_path: Path) -> None:
    config = _joint_config(tmp_path)
    config["runtime"]["algorithms"]["graphs"]["model_roles"]["embedding"] = "chat"

    with pytest.raises(SkillConfigurationError, match="role 'embedding' requires a 'embedding' model"):
        SkillConfigCompiler().compile(config)


def test_skill_manage_dynamic_chat_role_rejects_embedding_model(tmp_path: Path) -> None:
    config = _joint_config(tmp_path)
    config["runtime"]["algorithms"]["graphs"].update(config={"llm_linking": True})
    config["runtime"]["algorithms"]["graphs"]["model_roles"]["chat"] = "embedding"

    with pytest.raises(SkillConfigurationError, match="role 'chat' requires a 'chat' model"):
        SkillConfigCompiler().compile(config)


def test_agent_model_ref_still_requires_chat(tmp_path: Path) -> None:
    config = _joint_config(tmp_path)
    config["runtime"]["agents"] = {"executor": {"type": "react", "model_ref": "embedding"}}

    with pytest.raises(SkillConfigurationError, match="model_ref must reference a chat model"):
        SkillConfigCompiler().compile(config)


def test_model_kind_declaration_requires_role_even_without_legacy_role_set(tmp_path: Path) -> None:
    component = replace(
        get_component(type=ComponentType.ALGO, name="offline_evolution"),
        requirements=ComponentRequirements(required_model_kinds={"embedding": "embedding"}),
    )
    compiler = SkillConfigCompiler(component_resolver=lambda **kwargs: component)
    config = {"runtime": {"algorithms": {"custom": {"type": component.name}}}}

    with pytest.raises(SkillConfigurationError, match="missing model roles: embedding"):
        compiler.compile(config)


def test_custom_vector_registry_controls_available_providers() -> None:
    def unexpected_create(options, tables):
        raise AssertionError("configuration compilation must not construct a vector store")

    registry = BackendRegistry()
    registry.register("custom", unexpected_create)
    compiler = SkillConfigCompiler(vector_store_registry=registry)
    compiled = compiler.compile({"local": {"vector_store": {"provider": "custom"}}})

    assert compiled.local.vector_store.provider == "custom"
    with pytest.raises(SkillConfigurationError, match="available providers: custom"):
        compiler.compile({"local": {"vector_store": {"provider": "pgvector"}}})
