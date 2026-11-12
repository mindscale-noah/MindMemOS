"""Validate native backend selection through the existing service configuration."""

from pathlib import Path

import pytest
import yaml
from mindmemos_skill.agents import Agent, ClaudeConfig, CodexConfig
from mindmemos_skill.algos.evolve.offline_evolution.config import OfflineEvolutionConfig
from mindmemos_skill.config.loader import load_skill_config
from mindmemos_skill.pipelines.builtin_pipelines.offline_evolution.config import OfflineTrajectoryConfig
from pydantic import ValidationError

EXAMPLE = Path(__file__).resolve().parents[4] / "config/mindmemos_skill/dev.example.yaml"
"""Shared public service configuration containing both native presets."""


@pytest.mark.parametrize("backend", ["claude", "codex"])
def test_offline_preset_and_serialized_config_preserve_backend_identity(backend):
    raw = yaml.safe_load(EXAMPLE.read_text())
    name = next(name for name, agent in raw["service"]["agents"].items() if agent["class_name"] == backend)
    raw["service"]["evolution_pipelines"]["offline"]["config"]["evolver"]["config"]["agent"] = name
    service = load_skill_config(yaml.safe_dump(raw)).service
    config = service.evolution_pipelines["offline"].config.evolver.config
    restored = OfflineEvolutionConfig.model_validate_json(config.model_dump_json())
    assert restored.agent.class_name == backend
    assert restored.agent == service.agents[name].config
    assert type(Agent.from_config(restored.agent)) is Agent.resolve(backend)
    assert set(service.evolution_pipelines) == {"offline"}


def test_offline_default_and_explicit_backend_validation():
    assert isinstance(OfflineEvolutionConfig().agent, ClaudeConfig)
    selected = OfflineEvolutionConfig(agent={"class_name": "codex", "model": "offline"})
    assert isinstance(selected.agent, CodexConfig)
    with pytest.raises(ValidationError, match="requires class_name"):
        OfflineEvolutionConfig(agent={"model": "ambiguous"})
    with pytest.raises(ValidationError, match="allowed backends"):
        OfflineEvolutionConfig(agent={"class_name": "openai_chat", "model": "unsupported"})


def test_offline_evidence_limits_are_optional():
    """Use all unconsumed exact-version evidence by default."""
    assert OfflineTrajectoryConfig().trajectory_limit is None
    assert OfflineTrajectoryConfig().exclude_consumed is True
    assert OfflineTrajectoryConfig(exclude_consumed=False).exclude_consumed is False


@pytest.mark.parametrize("field", ["trajectory_limit", "max_page_size"])
def test_offline_evidence_limits_reject_zero(field):
    with pytest.raises(ValidationError):
        OfflineTrajectoryConfig(**{field: 0})


@pytest.mark.parametrize("field", ["sample_size", "seed"])
def test_offline_rejects_sampling_settings(field):
    """Reject removed options instead of retaining a second sampling contract."""
    with pytest.raises(ValidationError, match="Extra inputs"):
        OfflineTrajectoryConfig(**{field: 3})


def test_example_selects_offline_as_default():
    """Use the same collection pipeline for omitted and explicit HTTP selections."""
    deployment = load_skill_config(EXAMPLE.read_text())
    assert deployment.fastapi.evolution_pipelines == {"offline"}
    assert deployment.fastapi.default_evolution_pipeline == "offline"
    pipeline = deployment.service.evolution_pipelines["offline"]
    assert pipeline.class_name == "offline_evolution"
    assert pipeline.config.evolver.class_name == "offline_evolution"
    assert pipeline.config.traj_collect.trajectory_limit is None
    assert pipeline.config.traj_collect.exclude_consumed is True
