from __future__ import annotations

import importlib.util
import io
import json
import sys
import tarfile
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "benchmark_download" / "download_mindmemos_skill_datasets.py"
SPEC = importlib.util.spec_from_file_location("download_mindmemos_skill_datasets", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
SCRIPT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SCRIPT
SPEC.loader.exec_module(SCRIPT)


def test_default_cli_downloads_all_datasets_to_repo_data_root() -> None:
    args = SCRIPT.parse_args([])

    assert args.datasets == []
    assert args.data_root == REPO_ROOT / "data" / "mindmemos_skill"
    assert SCRIPT._DATASET_NAMES == ("alfworld", "livemath", "spreadsheetbench")


def test_safe_extract_tar_extracts_regular_files(tmp_path: Path) -> None:
    archive_path = tmp_path / "dataset.tar.gz"
    payload = b"[]\n"
    with tarfile.open(archive_path, mode="w:gz") as archive:
        member = tarfile.TarInfo("spreadsheetbench_verified_400/dataset.json")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))

    destination = tmp_path / "output"
    SCRIPT._safe_extract_tar(archive_path, destination)

    assert (destination / "spreadsheetbench_verified_400" / "dataset.json").read_bytes() == payload


def test_safe_extract_tar_rejects_path_traversal(tmp_path: Path) -> None:
    archive_path = tmp_path / "unsafe.tar.gz"
    payload = b"escape"
    with tarfile.open(archive_path, mode="w:gz") as archive:
        member = tarfile.TarInfo("../escape.txt")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))

    with pytest.raises(RuntimeError, match="unsafe tar member"):
        SCRIPT._safe_extract_tar(archive_path, tmp_path / "output")
    assert not (tmp_path / "escape.txt").exists()


def test_alfworld_download_uses_platform_resolved_console_script(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[list[str], bool]] = []

    def fake_run(command: list[str], *, check: bool) -> None:
        calls.append((command, check))
        target = Path(command[command.index("--data-dir") + 1])
        for relative in SCRIPT._ALFWORLD_EXPECTED_PATHS:
            path = target / relative
            if path.suffix:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture", encoding="utf-8")
            else:
                path.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(SCRIPT.shutil, "which", lambda _: "C:/venv/Scripts/alfworld-download.exe")
    monkeypatch.setattr(SCRIPT.subprocess, "run", fake_run)

    target = SCRIPT.download_alfworld(tmp_path, force=True)

    assert target == tmp_path / "alfworld"
    assert calls == [
        (
            [
                "C:/venv/Scripts/alfworld-download.exe",
                "--data-dir",
                str(target),
                "--force",
                "--force-download",
            ],
            True,
        )
    ]


def test_alfworld_download_reports_missing_optional_extra(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(SCRIPT.shutil, "which", lambda _: None)

    with pytest.raises(RuntimeError, match="dataset-download extra"):
        SCRIPT.download_alfworld(tmp_path)


def test_livemath_download_materializes_expected_layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    snapshot = tmp_path / "snapshot"
    for relative in SCRIPT.LIVEMATH_SOURCE_FILES:
        source = snapshot / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(f"{relative}\n", encoding="utf-8")

    def fake_snapshot_download(**kwargs: object) -> str:
        assert kwargs["repo_id"] == SCRIPT.LIVEMATH_REPO_ID
        assert kwargs["revision"] == SCRIPT.LIVEMATH_REVISION
        return str(snapshot)

    monkeypatch.setattr(SCRIPT, "_huggingface_hub", lambda: (None, fake_snapshot_download))
    target = SCRIPT.download_livemath(tmp_path / "payloads")

    assert target == tmp_path / "payloads" / "livemath" / "raw"
    assert all((target / relative).is_file() for relative in SCRIPT.LIVEMATH_SOURCE_FILES)


def test_spreadsheetbench_download_materializes_expected_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive_path = tmp_path / SCRIPT.SPREADSHEETBENCH_ARCHIVE
    payload = b"[]\n"
    with tarfile.open(archive_path, mode="w:gz") as archive:
        member = tarfile.TarInfo("spreadsheetbench_verified_400/dataset.json")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))

    def fake_hf_hub_download(**kwargs: object) -> str:
        assert kwargs["repo_id"] == SCRIPT.SPREADSHEETBENCH_REPO_ID
        assert kwargs["revision"] == SCRIPT.SPREADSHEETBENCH_REVISION
        return str(archive_path)

    monkeypatch.setattr(SCRIPT, "_huggingface_hub", lambda: (fake_hf_hub_download, None))
    target = SCRIPT.download_spreadsheetbench(tmp_path / "payloads")

    assert target == tmp_path / "payloads" / "spreadsheetbench"
    assert (target / "spreadsheetbench_verified_400" / "dataset.json").read_bytes() == payload


@pytest.mark.parametrize(
    ("dataset", "counts"),
    [
        ("alfworld", {"train": 39, "val": 18, "test": 134}),
        ("livemath", {"train": 35, "val": 18, "test": 124}),
        ("spreadsheetbench", {"train": 80, "val": 40, "test": 280}),
    ],
)
def test_packaged_split_manifests_are_complete_and_disjoint(dataset: str, counts: dict[str, int]) -> None:
    root = REPO_ROOT / "resources" / "mindmemos_skill" / "datasets" / dataset
    manifest = json.loads((root / "split_manifest.json").read_text(encoding="utf-8"))
    seen: set[str] = set()

    assert manifest["counts"] == counts
    for split, expected_count in counts.items():
        items = json.loads((root / "splits" / split / "items.json").read_text(encoding="utf-8"))
        ids = {str(item["id"]) for item in items}
        assert len(items) == expected_count
        assert len(ids) == expected_count
        assert seen.isdisjoint(ids)
        seen.update(ids)


def test_mindmemos_skill_configs_only_reference_local_resources() -> None:
    config_root = REPO_ROOT / "config" / "mindmemos_skill"
    expected_data_roots = {
        "alfworld": "data/mindmemos_skill/alfworld",
        "livemath": "data/mindmemos_skill/livemath/raw",
        "spreadsheetbench": "data/mindmemos_skill/spreadsheetbench",
    }
    for config_path in config_root.glob("*/*/*.yaml"):
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        dataset = payload.get("parameters", {}).get("dataset", {})
        data_root = dataset.get("data_root")
        if data_root:
            assert not Path(data_root).is_absolute(), config_path
            assert ".." not in Path(data_root).parts, config_path
            assert data_root == expected_data_roots[payload["environment"]], config_path
        for key in ("split_dir", "initial_skill", "skill"):
            value = dataset.get(key)
            if value:
                assert value.startswith("resources/mindmemos_skill/"), config_path
                assert (REPO_ROOT / value).exists(), (config_path, value)


def test_download_revisions_match_packaged_manifests() -> None:
    resource_root = REPO_ROOT / "resources" / "mindmemos_skill" / "datasets"
    livemath = json.loads((resource_root / "livemath" / "split_manifest.json").read_text(encoding="utf-8"))
    spreadsheetbench = json.loads(
        (resource_root / "spreadsheetbench" / "split_manifest.json").read_text(encoding="utf-8")
    )

    assert livemath["source_revision"] == SCRIPT.LIVEMATH_REVISION
    assert spreadsheetbench["source_revision"] == SCRIPT.SPREADSHEETBENCH_REVISION
