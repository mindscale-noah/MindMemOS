from __future__ import annotations

from types import SimpleNamespace

import pytest
from mindmemos_eval.memory.runner import run_benchmark_matrix


class _NoopAdapter:
    name = "locomo"

    async def run(self, **_kwargs):
        return {"ok": True}


def _write_config(path):
    path.write_text(
        """
benchmarks:
  locomo:
    dataset: data/locomo.json
    memory_algorithm: vanilla
""".lstrip(),
        encoding="utf-8",
    )


def _write_api_keys(path):
    path.write_text(
        """
api_keys:
  - key_id: key-1
    api_key: dev-api-key
    project_id: proj-1
    memory_algorithm: vanilla
    enabled: true
""".lstrip(),
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_reuse_api_key_rejects_multiple_benchmarks(tmp_path):
    config_path = tmp_path / "memory_eval.yaml"
    config_path.write_text(
        """
benchmarks:
  locomo:
    dataset: data/locomo.json
    memory_algorithm: vanilla
  longmemeval:
    dataset: data/longmemeval.json
    memory_algorithm: vanilla
""".lstrip(),
        encoding="utf-8",
    )
    api_key_path = tmp_path / "api_keys.yaml"
    api_key_path.write_text(
        """
api_keys:
  - key_id: key-1
    api_key: dev-api-key
    project_id: proj-1
    memory_algorithm: vanilla
    enabled: true
""".lstrip(),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        benchmark_config=str(config_path),
        benchmark_list="locomo,longmemeval",
        manifest_output=str(tmp_path / "manifest.jsonl"),
        api_key_output=str(tmp_path / "generated_api_keys.yaml"),
        reuse_api_key=str(api_key_path),
    )

    with pytest.raises(ValueError, match="--reuse-api-key can only be used with exactly one benchmark"):
        await run_benchmark_matrix(
            args,
            adapters={"locomo": object(), "longmemeval": object()},
        )


@pytest.mark.asyncio
async def test_reuse_api_key_logs_with_standard_logger_kwargs(tmp_path, monkeypatch):
    config_path = tmp_path / "memory_eval.yaml"
    _write_config(config_path)
    api_key_path = tmp_path / "api_keys.yaml"
    _write_api_keys(api_key_path)
    args = SimpleNamespace(
        benchmark_config=str(config_path),
        benchmark_list="locomo",
        manifest_output=str(tmp_path / "manifest.jsonl"),
        api_key_output=str(tmp_path / "generated_api_keys.yaml"),
        reuse_api_key=str(api_key_path),
    )

    from mindmemos_eval.memory import runner

    monkeypatch.setattr(runner.logger, "level", runner.logging.INFO)

    async def memory_client_factory(_identity):
        return object(), None

    manifests = await run_benchmark_matrix(
        args,
        adapters={"locomo": _NoopAdapter()},
        memory_client_factory=memory_client_factory,
        answer_llm_factory=lambda: object(),
        judge_llm_factory=lambda: object(),
    )

    assert manifests[0].eval_result == {"ok": True}
    assert manifests[0].api_key_file == str(api_key_path)
