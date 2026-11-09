"""Verify explicit credential injection into the no-skill evaluation policy."""

import yaml
from mindmemos_skill.common.json import write_json

from scripts.treeskill.no_skill import load_policy


def test_policy_uses_local_credentials_and_recorded_generation_settings(tmp_path):
    reference = tmp_path / "comparison"
    recorded = {
        "optimizer": {"model": "recorded-optimizer", "api_key": "[REDACTED]"},
        "router": {"model": "recorded-router", "api_key": "[REDACTED]"},
        "failure_optimizer": {
            "model": "recorded-policy",
            "api_key": "[REDACTED]",
            "base_url": "https://recorded.invalid/v1",
            "max_completion_tokens": 1234,
        },
    }
    write_json(reference / "training/config.json", {"algorithm": recorded})
    local = {
        "optimizer": {"model": "local", "api_key": "optimizer-private", "base_url": "https://local.invalid/v1"},
        "router": {"model": "local"},
        "failure_optimizer": {"model": "local", "api_key": "policy-private", "base_url": "https://local.invalid/v1"},
    }
    path = tmp_path / "service.yaml"
    path.write_text(
        yaml.safe_dump({"service": {"training_pipelines": {"treeskill": {"config": {"trainer": {"config": local}}}}}})
    )
    policy = load_policy(reference, path)
    assert policy.api_key == "policy-private"
    assert policy.model == "recorded-policy"
    assert policy.base_url == "https://recorded.invalid/v1"
    assert policy.max_completion_tokens == 1234
    assert policy.temperature == 0.7 and policy.stop == ["Observation:"]
    assert "policy-private" not in (reference / "training/config.json").read_text()
