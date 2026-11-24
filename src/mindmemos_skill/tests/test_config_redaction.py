"""Verify safe experiment snapshots without changing runtime credentials."""

from copy import deepcopy

import pytest
from mindmemos_skill.agents import OpenAIChatConfig
from mindmemos_skill.algos.train.treeskill import TreeSkillTrainerConfig
from mindmemos_skill.algos.train.treeskill.artifacts import experiment_config
from mindmemos_skill.common.json import read_json, write_json
from mindmemos_skill.common.secrets import is_sensitive_key, redact_credentials


@pytest.mark.parametrize(
    "key", ["api_key", "API-Key", "provider_api_key", "access_token", "password", "jwt_secret", "authorization"]
)
def test_recognizes_credential_names(key):
    assert is_sensitive_key(key)


def test_nested_snapshot_redaction_preserves_inputs_and_generation_settings(tmp_path):
    value = {
        "agent": {"api_key": "private", "model": "model", "base_url": "https://provider.invalid/v1"},
        "models": [{"API-Key": "another-private", "max_completion_tokens": 10}],
        "unused": {"api_key": None, "password": ""},
        "instruction": "Keep api_key in your local configuration.",
    }
    before = deepcopy(value)
    path = tmp_path / "config.json"
    write_json(path, redact_credentials(value))
    saved = read_json(path)
    assert value == before
    assert "private" not in path.read_text()
    assert saved["agent"] == "[REDACTED]"
    assert saved["models"] == [{"API-Key": "[REDACTED]", "max_completion_tokens": 10}]
    assert saved["unused"] == value["unused"]
    assert saved["instruction"] == value["instruction"]


def test_treeskill_snapshot_masks_all_model_roles_without_changing_runtime_config():
    model = OpenAIChatConfig(model="test", api_key="private", base_url="https://provider.invalid/v1")
    config = TreeSkillTrainerConfig(optimizer=model, router=model, failure_optimizer=model)
    snapshot = experiment_config(config)
    for role in ("optimizer", "router", "failure_optimizer"):
        assert snapshot["algorithm"][role]["api_key"] == "[REDACTED]"
        assert snapshot["algorithm"][role]["base_url"] == "[REDACTED]"
        assert snapshot["algorithm"][role]["model"] == "[REDACTED]"
    assert config.optimizer.api_key == config.router.api_key == config.failure_optimizer.api_key == "private"
