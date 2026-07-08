from __future__ import annotations

from types import SimpleNamespace

import pytest
from mindmemos_eval.memory.runner import run_benchmark_matrix


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
