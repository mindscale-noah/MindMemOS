from __future__ import annotations

import pytest
from mindmemos_skill.envs import (
    ALFWorldBoundedHistoryEnv,
    ALFWorldEnv,
    BaseEnv,
    LiveMathEnv,
    PreparedRollout,
    get_env,
    list_envs,
)
from mindmemos_skill.registry import ComponentType, create, get_component, register
from mindmemos_skill.typing import EnvConfig, Reward, Trajectory


class ExternalEnvConfig(EnvConfig):
    marker: str = "external"


@register(type=ComponentType.ENV, name="test_external_env")
class ExternalEnv(BaseEnv[ExternalEnvConfig]):
    config_type = ExternalEnvConfig

    async def _evaluate(self, *, trajectory: Trajectory, prepared: PreparedRollout) -> Reward:
        del trajectory, prepared
        return Reward(score=1.0)


def test_registry_rejects_plain_string_component_types() -> None:
    with pytest.raises(TypeError, match="ComponentType enum member"):
        register(type="env", name="plain_string_type")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="ComponentType enum member"):
        create(type="env", name="test_external_env")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="ComponentType enum member"):
        get_component(type="env", name="test_external_env")  # type: ignore[arg-type]


def test_builtin_and_package_external_envs_use_the_same_registry() -> None:
    assert "livemath" in list_envs()
    assert "alfworld" in list_envs()
    assert "alfworld_bounded_history" in list_envs()
    assert "alfworld_skillopt" not in list_envs()
    assert "alfworld_lean_history" not in list_envs()
    assert "test_external_env" in list_envs()

    env = get_env(name="test_external_env", config={"marker": "selected-by-name"})

    assert isinstance(env, ExternalEnv)
    assert env.config.marker == "selected-by-name"


def test_builtin_envs_live_in_independent_registered_env_packages() -> None:
    assert ALFWorldEnv.__module__ == "mindmemos_skill.envs.registered_envs.alfworld.env"
    assert ALFWorldBoundedHistoryEnv.__module__ == (
        "mindmemos_skill.envs.registered_envs.alfworld_bounded_history.env"
    )
    assert LiveMathEnv.__module__ == "mindmemos_skill.envs.registered_envs.livemath.env"
