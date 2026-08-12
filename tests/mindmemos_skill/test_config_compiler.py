"""Phase 4A contracts for typed Skill application configuration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from mindmemos_skill.agents.react import ReactAgentConfig
from mindmemos_skill.config import SkillApplicationConfig, SkillConfigCompiler
from mindmemos_skill.errors import SkillConfigurationError
from mindmemos_skill.registry import ComponentRequirements, ComponentType, get_component, register
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError


class DemoOptimizerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_iterations: int = Field(default=2, ge=1)


@register(
    type=ComponentType.ALGO,
    name="test_config_compiler_optimizer",
    config_model=DemoOptimizerConfig,
    capabilities={"optimize"},
    requirements=ComponentRequirements(required_model_roles=frozenset({"proposer"})),
)
class DemoOptimizer:
    pass


def _application_config(tmp_path: Path, *, api_key: str = "model-secret") -> dict:
    return {
        "local": {
            "root_dir": str(tmp_path / "skill"),
            "database": {"provider": "SQLITE", "path": "db/state.db"},
            "artifacts_dir": "artifacts",
        },
        "runtime": {
            "models": {
                "target": {
                    "model": "openai/test-model",
                    "api_base": "https://models.example/v1",
                    "api_key": api_key,
                    "temperature": 0.2,
                    "options": {"num_retries": 3, "timeout": 45},
                }
            },
            "agents": {
                "executor": {
                    "type": "react",
                    "model_ref": "target",
                    "skill_injection_mode": "tool",
                    "config": {"max_turns": 4},
                }
            },
            "algorithms": {
                "optimizer": {
                    "type": "test_config_compiler_optimizer",
                    "model_roles": {"proposer": "target"},
                    "config": {"max_iterations": 3},
                }
            },
            "execution": {"max_concurrent_rollouts": 3, "attempt_limit": 2},
        },
    }


def test_component_catalog_keeps_factory_config_capabilities_and_requirements() -> None:
    react = get_component(type=ComponentType.AGENT, name="react")
    optimizer = get_component(type=ComponentType.ALGO, name="test_config_compiler_optimizer")

    assert react.config_model is ReactAgentConfig
    assert react.capabilities == frozenset({"execute"})
    assert react.requirements.requires_model_ref is True
    assert react.requirements.supported_skill_injection_modes == frozenset(
        {"system_prompt", "tool", "tree_routed_system_prompt"}
    )
    assert optimizer.factory is DemoOptimizer
    assert optimizer.config_model is DemoOptimizerConfig
    assert optimizer.capabilities == frozenset({"optimize"})
    assert optimizer.requirements.required_model_roles == frozenset({"proposer"})


def test_compiler_resolves_paths_references_and_typed_component_configs(tmp_path: Path) -> None:
    compiled = SkillConfigCompiler().compile(_application_config(tmp_path))

    assert compiled.local.root_dir == (tmp_path / "skill").resolve()
    assert compiled.local.artifacts_dir == (tmp_path / "skill" / "artifacts").resolve()
    assert compiled.local.database.provider == "sqlite"
    assert compiled.local.database.options == {"path": str((tmp_path / "skill" / "db/state.db").resolve())}
    assert compiled.local.database.required.transactions is True
    assert compiled.local.database.required.compare_and_swap is True
    assert compiled.local.database.required.atomic_batch_write is True
    assert compiled.runtime.models["target"].model == "openai/test-model"
    assert compiled.runtime.models["target"].provider == "openai"
    assert compiled.runtime.models["target"].options == {"num_retries": 3, "timeout": 45}
    assert isinstance(compiled.runtime.models["target"].api_key, SecretStr)
    assert isinstance(compiled.runtime.agents["executor"].config, ReactAgentConfig)
    assert compiled.runtime.agents["executor"].config.model == "openai/test-model"
    assert compiled.runtime.agents["executor"].config.max_turns == 4
    assert isinstance(compiled.runtime.algorithms["optimizer"].config, DemoOptimizerConfig)
    assert compiled.runtime.algorithms["optimizer"].config.max_iterations == 3
    assert len(compiled.config_hash) == 64


def test_compiler_snapshot_and_hash_never_persist_plaintext_credentials(tmp_path: Path) -> None:
    first = SkillConfigCompiler().compile(_application_config(tmp_path, api_key="first-plaintext-secret"))
    second = SkillConfigCompiler().compile(_application_config(tmp_path, api_key="second-plaintext-secret"))

    serialized = json.dumps(first.config_snapshot, sort_keys=True)
    assert "first-plaintext-secret" not in serialized
    assert "first-plaintext-secret" not in repr(first)
    assert first.config_snapshot["runtime"]["models"]["target"]["api_key_configured"] is True
    assert first.config_snapshot["runtime"]["models"]["target"]["provider"] == "openai"
    assert first.config_hash == second.config_hash


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda config: config["runtime"]["agents"]["executor"].update(model_ref="missing"), "unknown model"),
        (
            lambda config: config["runtime"]["algorithms"]["optimizer"].update(model_roles={}),
            "missing model roles",
        ),
        (lambda config: config["local"]["database"].update(provider="missing"), "unknown database provider"),
        (
            lambda config: config["runtime"]["models"]["target"].update(options={"model": "shadow"}),
            "duplicate explicit fields",
        ),
        (lambda config: config["runtime"]["models"]["target"].update(model="test-model"), "invalid model"),
        (lambda config: config["runtime"]["models"]["target"].update(provider="openai"), "extra_forbidden"),
        (lambda config: config["runtime"]["agents"]["executor"]["config"].update(unknown=True), "invalid config"),
    ],
)
def test_compiler_rejects_dangling_references_unknown_providers_and_component_fields(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    config = _application_config(tmp_path)
    mutation(config)

    with pytest.raises(SkillConfigurationError, match=message):
        SkillConfigCompiler().compile(config)


def test_compiler_rejects_agent_family_injection_mode_mismatch(tmp_path: Path) -> None:
    config = _application_config(tmp_path)
    config["runtime"]["agents"]["executor"] = {
        "type": "claude",
        "skill_injection_mode": "tool",
    }

    with pytest.raises(SkillConfigurationError, match="invalid config"):
        SkillConfigCompiler().compile(config)


def test_compiler_rejects_unknown_application_fields_and_file_loading(tmp_path: Path) -> None:
    with pytest.raises(SkillConfigurationError, match="extra_forbidden"):
        SkillConfigCompiler().compile({"unknown": True})
    with pytest.raises(SkillConfigurationError, match="config loader must read"):
        SkillConfigCompiler().compile(tmp_path / "config.yaml")


def test_typed_application_config_is_frozen() -> None:
    config = SkillApplicationConfig()

    with pytest.raises(ValidationError):
        config.local = config.local
