"""Validate native backend selection through the existing service configuration."""

from pathlib import Path

import pytest
import yaml
from mindmemos_skill.agents import Agent, ClaudeConfig, CodexConfig
from mindmemos_skill.algos.evolve.offline_evolution.config import OfflineEvolutionConfig
from mindmemos_skill.config.loader import load_skill_config
from mindmemos_skill.pipelines.builtin_pipelines.evolve.config import TrajCollectConfig
from mindmemos_skill.pipelines.builtin_pipelines.offline_evolution.config import OfflineTrajectoryConfig
from pydantic import ValidationError

EXAMPLE = Path(__file__).resolve().parents[4] / "config/mindmemos_skill/dev.example.yaml"
"""Shared public service configuration containing both native presets."""


@pytest.mark.parametrize("backend", ["claude", "codex"])
def test_offline_preset_and_serialized_config_preserve_backend_identity(backend):
    raw = yaml.safe_load(EXAMPLE.read_text())
    raw["service"]["evolution_pipelines"]["offline"]["config"]["evolver"]["config"]["agent"] = (
        "offline_evolution_" + backend
    )
    service = load_skill_config(yaml.safe_dump(raw)).service
    config = service.evolution_pipelines["offline"].config.evolver.config
    restored = OfflineEvolutionConfig.model_validate_json(config.model_dump_json())
    assert restored.agent.class_name == backend
    assert restored.agent == service.agents["offline_evolution_" + backend].config
    assert type(Agent.from_config(restored.agent)) is Agent.resolve(backend)
    assert service.evolution_pipelines["simple"].config.evolver.class_name == "simple_evolve"


def test_offline_default_and_explicit_backend_validation():
    assert isinstance(OfflineEvolutionConfig().agent, ClaudeConfig)
    selected = OfflineEvolutionConfig(agent={"class_name": "codex", "model": "offline"})
    assert isinstance(selected.agent, CodexConfig)
    with pytest.raises(ValidationError, match="requires class_name"):
        OfflineEvolutionConfig(agent={"model": "ambiguous"})
    with pytest.raises(ValidationError, match="allowed backends"):
        OfflineEvolutionConfig(agent={"class_name": "openai_chat", "model": "unsupported"})


def test_offline_evidence_limits_are_optional_and_independent_of_simple_evolution():
    """Keep all offline evidence by default without changing fixed Simple sampling."""
    offline = OfflineTrajectoryConfig()
    assert offline.trajectory_limit is None
    simple = TrajCollectConfig()
    assert (simple.sample_size, simple.trajectory_limit) == (40, 100)
    with pytest.raises(ValidationError):
        TrajCollectConfig(sample_size=40, trajectory_limit=2)


@pytest.mark.parametrize("field", ["trajectory_limit", "max_page_size"])
def test_offline_evidence_limits_reject_zero(field):
    with pytest.raises(ValidationError):
        OfflineTrajectoryConfig(**{field: 0})


@pytest.mark.parametrize("field", ["sample_size", "seed"])
def test_offline_rejects_sampling_settings(field):
    """Reject removed options instead of retaining a second sampling contract."""
    with pytest.raises(ValidationError, match="Extra inputs"):
        OfflineTrajectoryConfig(**{field: 3})
