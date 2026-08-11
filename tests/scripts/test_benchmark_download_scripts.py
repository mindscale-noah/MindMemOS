from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = REPO_ROOT / "scripts" / "benchmark_download"
SCRIPT_PATH = SCRIPT_DIR / "download_memory_benchmarks.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("download_memory_benchmarks", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SCRIPT = _load_script()


def test_benchmark_downloads_use_one_python_entrypoint() -> None:
    assert list(SCRIPT_DIR.glob("*.sh")) == []
    assert {path.name for path in SCRIPT_DIR.glob("download_*.py")} == {
        "download_memory_benchmarks.py",
        "download_mindmemos_skill_datasets.py",
    }


def test_default_cli_downloads_all_memory_benchmarks_to_repo_dataset_root() -> None:
    args = SCRIPT.parse_args([])

    assert args.benchmarks == []
    assert args.data_root == REPO_ROOT / "datasets"
    assert SCRIPT._BENCHMARK_NAMES == ("locomo", "longmemeval", "memoryagentbench", "personamem")


def test_memory_eval_configs_use_downloader_default_paths() -> None:
    memory_config = yaml.safe_load(
        (REPO_ROOT / "config/mindmemos_eval/memory_evaluation_locomo.example.yaml").read_text(encoding="utf-8")
    )
    dreaming_config = yaml.safe_load(
        (REPO_ROOT / "config/mindmemos_eval/dreaming_evaluation_mab.example.yaml").read_text(encoding="utf-8")
    )

    benchmarks = memory_config["benchmarks"]
    assert benchmarks["locomo"]["dataset"] == "datasets/locomo/locomo10.json"
    assert benchmarks["longmemeval"]["dataset"] == "datasets/longmemeval/longmemeval_s_cleaned.json"
    assert benchmarks["personamem"]["dataset"] == "datasets/personamem/questions_32k.csv"
    assert benchmarks["personamem"]["context_dataset"] == "datasets/personamem/shared_contexts_32k.jsonl"
    assert (
        dreaming_config["benchmarks"]["memoryagentbench"]["dataset"]
        == "datasets/memoryagentbench/conflict_resolution.jsonl"
    )


def test_download_file_uses_python_http_and_replaces_output_atomically(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "nested" / "dataset.json"
    response = io.BytesIO(b'{"ok": true}\n')
    requests: list[Any] = []

    def fake_urlopen(request: Any) -> io.BytesIO:
        requests.append(request)
        return response

    monkeypatch.setattr(SCRIPT.urllib.request, "urlopen", fake_urlopen)

    result = SCRIPT.download_file("https://example.test/dataset.json", output)

    assert result == output.resolve()
    assert output.read_bytes() == b'{"ok": true}\n'
    assert requests[0].full_url == "https://example.test/dataset.json"
    assert requests[0].get_header("User-agent") == SCRIPT.USER_AGENT


def test_atomic_output_preserves_existing_file_on_failure(tmp_path: Path) -> None:
    output = tmp_path / "dataset.json"
    output.write_text("old", encoding="utf-8")

    with pytest.raises(RuntimeError, match="failed"):
        with SCRIPT.atomic_output_path(output) as temporary_path:
            temporary_path.write_text("partial", encoding="utf-8")
            raise RuntimeError("failed")

    assert output.read_text(encoding="utf-8") == "old"
    assert list(tmp_path.glob(".*.tmp")) == []


@pytest.mark.parametrize(
    ("downloader_name", "url_name", "relative_output"),
    [
        ("download_locomo", "LOCOMO_URL", Path("locomo/locomo10.json")),
        ("download_longmemeval", "LONGMEMEVAL_URL", Path("longmemeval/longmemeval_s_cleaned.json")),
    ],
)
def test_single_file_benchmarks_keep_existing_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    downloader_name: str,
    url_name: str,
    relative_output: Path,
) -> None:
    calls: list[tuple[str, Path]] = []

    def fake_download_file(url: str, destination: Path) -> Path:
        calls.append((url, destination))
        return destination.resolve()

    monkeypatch.setattr(SCRIPT, "download_file", fake_download_file)

    output = getattr(SCRIPT, downloader_name)(tmp_path)

    assert output == (tmp_path / relative_output).resolve()
    assert calls == [(getattr(SCRIPT, url_name), tmp_path / relative_output)]


def test_memoryagentbench_serializes_expected_split_as_jsonl(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    class FakeDataset:
        def to_json(self, path: str, **kwargs: Any) -> None:
            calls.append((path, kwargs))
            Path(path).write_text('{"id": 1}\n', encoding="utf-8")

    monkeypatch.setattr(SCRIPT, "_load_memoryagentbench", FakeDataset)

    output = SCRIPT.download_memoryagentbench(tmp_path)

    assert output == (tmp_path / "memoryagentbench" / "conflict_resolution.jsonl").resolve()
    assert output.read_text(encoding="utf-8") == '{"id": 1}\n'
    assert calls[0][1] == {"orient": "records", "lines": True, "force_ascii": False}


def test_personamem_downloads_both_payloads(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[tuple[str, Path]] = []

    def fake_download_file(url: str, destination: Path) -> Path:
        calls.append((url, destination))
        return destination

    monkeypatch.setattr(SCRIPT, "download_file", fake_download_file)

    assert SCRIPT.download_personamem(tmp_path) == (tmp_path / "personamem").resolve()
    assert calls == [
        (SCRIPT.PERSONAMEM_QUESTIONS_URL, tmp_path / "personamem" / "questions_32k.csv"),
        (SCRIPT.PERSONAMEM_CONTEXTS_URL, tmp_path / "personamem" / "shared_contexts_32k.jsonl"),
    ]


def test_main_downloads_selected_benchmarks_once(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[tuple[str, Path]] = []

    for name in SCRIPT._BENCHMARK_NAMES:
        monkeypatch.setattr(SCRIPT, f"download_{name}", lambda root, name=name: calls.append((name, root)))

    assert SCRIPT.main(["locomo", "personamem", "locomo", "--data-root", str(tmp_path)]) == 0
    assert calls == [("locomo", tmp_path.resolve()), ("personamem", tmp_path.resolve())]
