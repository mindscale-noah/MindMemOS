"""Effective configuration hashes distinguish behavior without retaining credentials."""

from __future__ import annotations

import json

import pytest
import yaml
from mindmemos_skill.algos.trace2skill.offline_evolution.algorithm import _resolve_vendor_config_hash
from mindmemos_skill.algos.trace2skill.offline_evolution.config import OfflineEvolutionConfig
from mindmemos_skill.algos.trace2skill.offline_evolution.impl import config


@pytest.fixture(autouse=True)
def isolated_config(monkeypatch):
    monkeypatch.setattr(config, "_CONFIG_PATH", None)
    monkeypatch.setattr(config, "_CONFIG_CACHE", None)
    monkeypatch.setattr(config, "_ENV_LOADED", True)
    for name in (
        "LLM_TEMPERATURE",
        "LLM_MAX_OUTPUT_TOKENS",
        "LLM_CONTEXT_WINDOW_TOKENS",
        "LLM_OUTPUT_TOKEN_PARAMETER",
        "LLM_REASONING_EFFORT",
        "PATTERN_KEYWORD_CLUSTER_THRESHOLD",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("llm", "temperature", "0.8"),
        ("llm", "max_output_tokens", "1024"),
        ("llm", "context_window_tokens", "8192"),
        ("llm", "output_token_parameter", "max_completion_tokens"),
        ("pattern", "keyword_cluster_threshold", "0.5"),
    ],
)
def test_environment_override_changes_effective_snapshot_and_hash(monkeypatch, section, key, value):
    before = _resolve_vendor_config_hash(OfflineEvolutionConfig())
    monkeypatch.setenv(f"{section}_{key}".upper(), value)
    assert config.resolved_config_snapshot()[section][key] == config.setting_raw(f"{section}.{key}") == value
    assert _resolve_vendor_config_hash(OfflineEvolutionConfig()) != before


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("llm", "max_output_tokens", 1024),
        ("llm", "output_token_parameter", "max_completion_tokens"),
        ("pattern", "keyword_cluster_threshold", 0.5),
    ],
)
def test_overlay_token_and_keyword_settings_change_hash(tmp_path, section, key, value):
    before = _resolve_vendor_config_hash(OfflineEvolutionConfig())
    path = tmp_path / "overlay.yaml"
    path.write_text(yaml.safe_dump({section: {key: value}}), encoding="utf-8")
    config.use_config_file(path)
    assert _resolve_vendor_config_hash(OfflineEvolutionConfig()) != before


def test_empty_environment_override_masks_yaml_default(monkeypatch):
    before = _resolve_vendor_config_hash(OfflineEvolutionConfig())
    monkeypatch.setenv("LLM_REASONING_EFFORT", "")
    assert config.setting_optional_str("llm.reasoning_effort") is None
    assert config.resolved_config_snapshot()["llm"]["reasoning_effort"] is None
    assert _resolve_vendor_config_hash(OfflineEvolutionConfig()) != before


def test_equivalent_raw_environment_and_yaml_values_have_same_hash(monkeypatch):
    before = _resolve_vendor_config_hash(OfflineEvolutionConfig())
    monkeypatch.setenv("LLM_MAX_OUTPUT_TOKENS", config.setting_raw("llm.max_output_tokens"))
    assert _resolve_vendor_config_hash(OfflineEvolutionConfig()) == before


def test_credentials_are_excluded_and_do_not_change_hash(monkeypatch, tmp_path):
    path = tmp_path / "overlay.yaml"
    path.write_text(
        yaml.safe_dump(
            {"llm": {"api_key": "test-secret", "access_token": "test-secret"}, "LLM_API_KEY": "test-secret"}
        ),
        encoding="utf-8",
    )
    config.use_config_file(path)
    before = _resolve_vendor_config_hash(OfflineEvolutionConfig())
    monkeypatch.setenv("LLM_API_KEY", "replacement-secret")
    monkeypatch.setenv("EMBEDDING_API_KEY", "replacement-secret")
    snapshot = config.resolved_config_snapshot()
    assert "api_key" not in snapshot["llm"] and "access_token" not in snapshot["llm"]
    assert "LLM_API_KEY" not in snapshot
    assert "secret" not in json.dumps(snapshot)
    assert _resolve_vendor_config_hash(OfflineEvolutionConfig()) == before


def test_list_settings_keep_yaml_semantics_and_do_not_mutate_cache(monkeypatch, tmp_path):
    path = tmp_path / "overlay.yaml"
    path.write_text(yaml.safe_dump({"test": {"names": ["one", "two"]}}), encoding="utf-8")
    config.use_config_file(path)
    monkeypatch.setenv("TEST_NAMES", "ignored")
    snapshot = config.resolved_config_snapshot()
    assert snapshot["test"]["names"] == list(config.setting_optional_list("test.names")) == ["one", "two"]
    snapshot["test"]["names"].append("three")
    assert config.setting_optional_list("test.names") == ("one", "two")
