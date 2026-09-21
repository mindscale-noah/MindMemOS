"""Tests for the centralized STATE-Bench configuration entry point."""

from __future__ import annotations

import json
from copy import deepcopy

import pytest
from mindmemos_eval.memory.envs.statebench.config_runner import (
    DEFAULT_CONFIG,
    _load_config,
    build_argv,
)
from mindmemos_eval.memory.envs.statebench.repair_runner import run as run_repair


def test_dry_run_metadata_is_redacted_and_reports_credentials():
    config = _load_config(DEFAULT_CONFIG)
    argv, metadata = build_argv(config, 42, require_credentials=False)

    assert argv[argv.index("--experiment-id") + 1].endswith("-s42")
    assert metadata["output_dir"].endswith("outputs/statebench/customer_support/seed42")
    assert all(values["credential_status"] == "available" for values in metadata["branches"].values())
    assert "api_key" not in json.dumps(metadata)


def test_unknown_seed_is_rejected():
    config = _load_config(DEFAULT_CONFIG)
    with pytest.raises(ValueError, match="not listed"):
        build_argv(config, 99, require_credentials=False)


def test_strict_run_requires_provisioned_key():
    config = deepcopy(_load_config(DEFAULT_CONFIG))
    config["isolation"]["baseline"]["key_id_template"] = "key_missing_for_test_s{seed}"
    with pytest.raises(ValueError, match="missing from api_keys"):
        build_argv(config, 42)


def test_repair_dry_run_uses_separate_output_and_redacts_keys(capsys):
    assert run_repair(DEFAULT_CONFIG, 42, dry_run=True) == 0
    metadata = json.loads(capsys.readouterr().out)
    assert metadata["protocol"] == "online-controlled-repair"
    assert metadata["output_dir"].endswith("seed42")
    assert metadata["output_dir"] == metadata["source_output_dir"]
    assert metadata["overwrite_source"] is True
    assert "api_key" not in json.dumps(metadata)


def test_repair_dry_run_can_use_separate_output(capsys):
    assert run_repair(DEFAULT_CONFIG, 42, separate_output=True, dry_run=True) == 0
    metadata = json.loads(capsys.readouterr().out)
    assert metadata["output_dir"].endswith("seed42-repair-round09")
    assert metadata["output_dir"] != metadata["source_output_dir"]
    assert metadata["overwrite_source"] is False
