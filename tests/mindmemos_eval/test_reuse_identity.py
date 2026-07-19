"""Tests for --reuse-identity: skip add and search a prior run's memories."""

from __future__ import annotations

import argparse

import pytest
import yaml
from mindmemos_eval.memory.identity import load_reused_identity
from mindmemos_eval.memory.runner import run_benchmark_matrix

_PRIOR_ENTRY = {
    "key_id": "key_personamem_vanilla_20260101_000000_abcd1234",
    "api_key": "dev-api-key-personamem-vanilla-20260101-000000-abcd1234",
    "project_id": "proj_personamem_vanilla_20260101_000000_abcd1234",
    "memory_algorithm": "vanilla",
    "enabled": True,
    "scopes": ["memory:read", "memory:write"],
    "project_override_config": {"algo_config": {"search": {"vanilla": {"recall_size": 200}}}},
}


def _write_api_keys(path, entries):
    path.write_text(yaml.safe_dump({"api_keys": entries}), encoding="utf-8")


# --------------------------------------------------------------------------- #
# load_reused_identity (pure function)
# --------------------------------------------------------------------------- #


def test_load_reused_identity_recovers_prior_project(tmp_path):
    path = tmp_path / "api_keys.yaml"
    _write_api_keys(path, [_PRIOR_ENTRY])

    ident = load_reused_identity(path, "personamem", "vanilla", profile="vanilla")

    assert ident.project_id == "proj_personamem_vanilla_20260101_000000_abcd1234"
    assert ident.api_key == "dev-api-key-personamem-vanilla-20260101-000000-abcd1234"
    assert ident.key_id == "key_personamem_vanilla_20260101_000000_abcd1234"
    # run_id is the suffix after the proj_<benchmark>_<algo>_ prefix, so memories stay traceable.
    assert ident.run_id == "20260101_000000_abcd1234"
    # The add-time server override (recall_size) rides along untouched.
    assert ident.project_override_config["algo_config"]["search"]["vanilla"]["recall_size"] == 200


def test_load_reused_identity_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="run an add pass first"):
        load_reused_identity(tmp_path / "absent.yaml", "personamem", "vanilla")


def test_load_reused_identity_no_matching_entry_raises(tmp_path):
    path = tmp_path / "api_keys.yaml"
    _write_api_keys(path, [{**_PRIOR_ENTRY, "key_id": "key_locomo_vanilla_x"}])

    with pytest.raises(ValueError, match="no api_keys entry for benchmark 'personamem'"):
        load_reused_identity(path, "personamem", "vanilla")


def test_load_reused_identity_rejects_ambiguous_entries(tmp_path):
    path = tmp_path / "api_keys.yaml"
    _write_api_keys(
        path,
        [
            _PRIOR_ENTRY,
            {**_PRIOR_ENTRY, "key_id": "key_personamem_vanilla_20260202_000000_ffff0000"},
        ],
    )

    with pytest.raises(ValueError, match="multiple api_keys entries"):
        load_reused_identity(path, "personamem", "vanilla")


# --------------------------------------------------------------------------- #
# runner reuse path
# --------------------------------------------------------------------------- #


class _FakeAdapter:
    name = "personamem"

    def __init__(self):
        self.seen_add = None

    async def run(self, *, memory, answer_llm, judge_llm, ctx, bench_config, args):
        self.seen_add = args.add
        return {"metrics": {"overall_accuracy": 1.0}}


def _base_config(tmp_path):
    path = tmp_path / "matrix.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "runner": {"base_url": "http://127.0.0.1:8000"},
                "benchmarks": {"personamem": {"dataset": "/tmp/fake.csv", "default_algorithm": "vanilla"}},
                "algorithm_profiles": {"vanilla": {"memory_algorithm": "vanilla", "search_params": {"top_k": 20}}},
            }
        ),
        encoding="utf-8",
    )
    return path


def _args(config_path, api_key_path, manifest_path, *, reuse, add=None):
    return argparse.Namespace(
        benchmark_config=str(config_path),
        benchmark_list="personamem",
        manifest_output=str(manifest_path),
        api_key_output=str(api_key_path),
        algorithm="vanilla",
        reuse_identity=reuse,
        add=add,
    )


@pytest.mark.asyncio
async def test_reuse_identity_rejects_explicit_add(tmp_path):
    config_path = _base_config(tmp_path)
    api_key_path = tmp_path / "api_keys.yaml"
    _write_api_keys(api_key_path, [_PRIOR_ENTRY])

    with pytest.raises(ValueError, match="cannot be combined"):
        await run_benchmark_matrix(
            _args(config_path, api_key_path, tmp_path / "manifest.jsonl", reuse=True, add=True),
            adapters={"personamem": _FakeAdapter()},
            memory_client_factory=lambda identity: None,
            answer_llm_factory=lambda: object(),
            judge_llm_factory=lambda: object(),
        )


@pytest.mark.asyncio
async def test_reuse_path_searches_prior_project_without_rewriting_keys(tmp_path):
    config_path = _base_config(tmp_path)
    api_key_path = tmp_path / "api_keys.yaml"
    _write_api_keys(api_key_path, [_PRIOR_ENTRY])
    manifest_path = tmp_path / "manifest.jsonl"

    captured = {}

    async def memory_client_factory(identity):
        captured["project_id"] = identity.project_id
        captured["api_key"] = identity.api_key
        return object(), None

    adapter = _FakeAdapter()
    await run_benchmark_matrix(
        _args(config_path, api_key_path, manifest_path, reuse=True),
        adapters={"personamem": adapter},
        memory_client_factory=memory_client_factory,
        answer_llm_factory=lambda: object(),
        judge_llm_factory=lambda: object(),
    )

    # The memory client connected to the PRIOR run's project (so search hits its memories).
    assert captured["project_id"] == "proj_personamem_vanilla_20260101_000000_abcd1234"
    # add stage is disabled by default when reusing an identity.
    assert adapter.seen_add is False
    # The server's active api-key file was left untouched (not clobbered with a new identity).
    assert yaml.safe_load(api_key_path.read_text(encoding="utf-8"))["api_keys"] == [_PRIOR_ENTRY]


@pytest.mark.asyncio
async def test_non_reuse_path_mints_new_project_and_writes_keys(tmp_path):
    config_path = _base_config(tmp_path)
    api_key_path = tmp_path / "api_keys.yaml"
    _write_api_keys(api_key_path, [_PRIOR_ENTRY])
    manifest_path = tmp_path / "manifest.jsonl"

    captured = {}

    async def memory_client_factory(identity):
        captured["project_id"] = identity.project_id
        return object(), None

    await run_benchmark_matrix(
        _args(config_path, api_key_path, manifest_path, reuse=False),
        adapters={"personamem": _FakeAdapter()},
        memory_client_factory=memory_client_factory,
        answer_llm_factory=lambda: object(),
        judge_llm_factory=lambda: object(),
    )

    # A fresh identity was minted (distinct from the prior project) ...
    assert captured["project_id"] != "proj_personamem_vanilla_20260101_000000_abcd1234"
    assert captured["project_id"].startswith("proj_personamem_vanilla_")
    # ... and written to the api-key file, replacing the prior entry.
    entries = yaml.safe_load(api_key_path.read_text(encoding="utf-8"))["api_keys"]
    assert len(entries) == 1
    assert entries[0]["project_id"] == captured["project_id"]


@pytest.mark.asyncio
async def test_reuse_identity_as_searches_aliased_project(tmp_path):
    # A subset benchmark reuses the full personamem run's project via
    # reuse_identity_as, while still running under its own name/adapter.
    config_path = tmp_path / "matrix.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "runner": {"base_url": "http://127.0.0.1:8000"},
                "benchmarks": {
                    "personamem_subset": {
                        "dataset": "/tmp/subset.csv",
                        "default_algorithm": "vanilla",
                        "reuse_identity_as": "personamem",
                    }
                },
                "algorithm_profiles": {"vanilla": {"memory_algorithm": "vanilla", "search_params": {"top_k": 20}}},
            }
        ),
        encoding="utf-8",
    )
    api_key_path = tmp_path / "api_keys.yaml"
    _write_api_keys(api_key_path, [_PRIOR_ENTRY])

    captured = {}

    async def memory_client_factory(identity):
        captured["project_id"] = identity.project_id
        captured["benchmark"] = identity.benchmark
        return object(), None

    args = argparse.Namespace(
        benchmark_config=str(config_path),
        benchmark_list="personamem_subset",
        manifest_output=str(tmp_path / "manifest.jsonl"),
        api_key_output=str(api_key_path),
        algorithm="vanilla",
        reuse_identity=True,
        add=None,
    )
    adapter = _FakeAdapter()
    await run_benchmark_matrix(
        args,
        adapters={"personamem_subset": adapter},
        memory_client_factory=memory_client_factory,
        answer_llm_factory=lambda: object(),
        judge_llm_factory=lambda: object(),
    )

    # Project comes from the aliased full run; the run stays under the subset name.
    assert captured["project_id"] == "proj_personamem_vanilla_20260101_000000_abcd1234"
    assert captured["benchmark"] == "personamem_subset"
    assert adapter.seen_add is False
    # api-key file is not rewritten.
    assert yaml.safe_load(api_key_path.read_text(encoding="utf-8"))["api_keys"] == [_PRIOR_ENTRY]
