"""Check official dataset acquisition without network access."""

import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("download_datasets", ROOT / "scripts/download_datasets.py")
DOWNLOAD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DOWNLOAD)


def test_released_splits_are_disjoint():
    """Keep committed experiment splits independent of dataset acquisition."""
    paths = list((ROOT / "resources/mindmemos_skill").glob("*/*/splits.json"))
    paths.extend((ROOT / "resources/skill_evaluation").glob("*/data_splits/splits.json"))
    for path in paths:
        splits = json.loads(path.read_text())
        assert set(splits) == {"train", "val", "test"}
        identifiers = [identifier for values in splits.values() for identifier in values]
        assert all(isinstance(identifier, str) and identifier for identifier in identifiers)
        assert len(identifiers) == len(set(identifiers)), path


def test_verification_rejects_changed_missing_and_extra_files(tmp_path):
    """Reject corruption and extra files, not just mismatching file counts."""
    source = tmp_path / "items.json"
    source.write_text("original")
    meta = {"revision": "fixed"}
    DOWNLOAD._write(tmp_path / DOWNLOAD.RECEIPT, {"source": meta, "checksums": DOWNLOAD._checksums(tmp_path)})
    DOWNLOAD._verify(meta, tmp_path)
    source.write_text("changed")
    with pytest.raises(ValueError, match="Payload mismatch"):
        DOWNLOAD._verify(meta, tmp_path)
    source.unlink()
    with pytest.raises(ValueError, match="Payload mismatch"):
        DOWNLOAD._verify(meta, tmp_path)
    source.write_text("original")
    (tmp_path / "unexpected").touch()
    with pytest.raises(ValueError, match="Payload mismatch"):
        DOWNLOAD._verify(meta, tmp_path)
    with pytest.raises(ValueError, match="Source version changed"):
        DOWNLOAD._verify({"revision": "new"}, tmp_path)


def test_manual_official_archive_is_published_without_split(tmp_path, monkeypatch):
    """Preserve upstream paths and avoid reading algorithm resources."""
    archive = tmp_path / "official.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("train/questions.json", "[]")
    data_root = tmp_path / "data"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "download_datasets.py",
            "download",
            "--benchmark",
            "skill",
            "--dataset",
            "docvqa",
            "--data-root",
            str(data_root),
            "--source-archive",
            str(archive),
            "--offline",
        ],
    )
    DOWNLOAD.main()
    destination = data_root / "docvqa"
    assert (destination / "train/questions.json").read_text() == "[]"
    assert not list(destination.rglob("splits.json"))
    receipt = DOWNLOAD._read(destination / DOWNLOAD.RECEIPT)
    assert receipt["archives"] == {archive.name: DOWNLOAD._hash(archive)}
    DOWNLOAD.main()


def test_failed_download_does_not_publish_partial_payload(tmp_path, monkeypatch):
    """A network failure must leave no dataset target or staging files."""

    def fail(meta, destination, offline, source_archives):
        (destination / "partial").touch()
        raise OSError("interrupted")

    monkeypatch.setattr(DOWNLOAD, "_materialize", fail)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "download_datasets.py",
            "download",
            "--benchmark",
            "skill",
            "--dataset",
            "hitab",
            "--data-root",
            str(tmp_path),
        ],
    )
    with pytest.raises(OSError, match="interrupted"):
        DOWNLOAD.main()
    assert not list(tmp_path.iterdir())


def test_zip_rejects_path_traversal(tmp_path):
    """Do not unpack a downloaded file outside the destination."""
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("../escaped", "bad")
    with pytest.raises(ValueError, match="Unsafe archive path"):
        DOWNLOAD._extract(archive, tmp_path / "payload")
    assert not (tmp_path / "escaped").exists()


@pytest.mark.parametrize(
    "group,included,excluded", [("memory", "locomo", "spreadsheetbench"), ("skill", "spreadsheetbench", "locomo")]
)
def test_list_filters_benchmark_group(group, included, excluded, monkeypatch, capsys):
    """List only datasets belonging to the requested group without downloading."""
    monkeypatch.setattr(sys, "argv", ["download_datasets.py", "list", "--benchmark", group])
    DOWNLOAD.main()
    output = capsys.readouterr().out
    assert included + ":" in output
    assert excluded + ":" not in output


def test_download_rejects_dataset_from_another_group(monkeypatch):
    """Reject a mismatched group before creating directories or fetching data."""
    monkeypatch.setattr(
        sys, "argv", ["download_datasets.py", "download", "--benchmark", "memory", "--dataset", "spreadsheetbench"]
    )
    with pytest.raises(SystemExit) as error:
        DOWNLOAD.main()
    assert error.value.code == 2


def test_memory_download_uses_official_file_and_default_root(tmp_path, monkeypatch):
    """Acquire a memory dataset and retain its source receipt under the memory root."""
    urls = []

    def fetch(url, destination):
        urls.append(url)
        destination.write_text('[{"qa": []}]')

    monkeypatch.setattr(DOWNLOAD, "ROOT", tmp_path)
    monkeypatch.setattr(DOWNLOAD, "_fetch", fetch)
    monkeypatch.setattr(
        sys, "argv", ["download_datasets.py", "download", "--benchmark", "memory", "--dataset", "locomo"]
    )
    DOWNLOAD.main()
    destination = tmp_path / "data/mindmemos_memory/locomo"
    assert json.loads((destination / "locomo10.json").read_text()) == [{"qa": []}]
    meta = DOWNLOAD._read(DOWNLOAD.SOURCES)["memory"]["locomo"]
    assert urls == [meta["files"]["locomo10.json"]]
    DOWNLOAD._verify(meta, destination)


def test_memory_huggingface_preserves_files_and_revision(tmp_path, monkeypatch):
    """Pin upstream HF versions and preserve official dataset files in offline mode."""
    from types import SimpleNamespace

    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "questions_32k.csv").write_text("question\nhello\n")
    (cache / "shared_contexts_32k.jsonl").write_text("{}\n")
    calls = []

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        return str(cache)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=snapshot_download))
    destination = tmp_path / "payload"
    destination.mkdir()
    meta = DOWNLOAD._read(DOWNLOAD.SOURCES)["memory"]["personamem"]
    DOWNLOAD._materialize(meta, destination, True, [])
    assert calls == [
        {
            "repo_id": meta["repo"],
            "repo_type": "dataset",
            "revision": meta["revision"],
            "allow_patterns": meta["files"],
            "local_files_only": True,
        }
    ]
    assert (destination / "questions_32k.csv").read_bytes() == (cache / "questions_32k.csv").read_bytes()
    assert (destination / "shared_contexts_32k.jsonl").exists()
