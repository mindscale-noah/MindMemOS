"""Check official dataset acquisition without network access."""

import json
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

from scripts import download_datasets as DOWNLOAD

ROOT = Path(__file__).resolve().parents[2]


def test_released_splits_are_disjoint():
    """Keep committed experiment splits independent of dataset acquisition."""
    paths = list((ROOT / "resources/mindmemos_skill").glob("*/*/splits.json"))
    paths.extend((ROOT / "resources/mindmemos_skill_eval").glob("*/data_splits/splits.json"))
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


def test_failed_download_does_not_publish_partial_payload(tmp_path, monkeypatch, capsys):
    """A network failure must leave no dataset target or staging files."""

    def fail(meta, destination, offline):
        (destination / "partial").touch()
        raise OSError("interrupted")

    monkeypatch.setattr(DOWNLOAD, "_materialize", fail)
    assert DOWNLOAD.main(["download", "skill", "--name", "hitab", "--data-root", str(tmp_path)]) == 1
    assert not list(tmp_path.iterdir())
    assert "interrupted" in capsys.readouterr().err


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
def test_list_filters_benchmark_group(group, included, excluded, capsys):
    """List only datasets belonging to the requested group without downloading."""
    assert DOWNLOAD.main(["list", group]) == 0
    output = capsys.readouterr().out
    assert included + ":" in output
    assert excluded + ":" not in output
    assert "docvqa" not in output


@pytest.mark.parametrize("options", [[], ["--all", "--name", "livemath"], ["--name", "docvqa"], ["--name", "locomo"]])
def test_download_rejects_invalid_selection(options):
    """Reject missing, conflicting, unsupported, or mismatched dataset choices."""
    with pytest.raises(SystemExit) as error:
        DOWNLOAD.main(["download", "skill", *options])
    assert error.value.code == 2


def test_memory_download_uses_official_file_and_default_root(tmp_path, monkeypatch):
    """Acquire a memory dataset and retain its source receipt under the memory root."""
    urls = []

    def fetch(url, destination):
        urls.append(url)
        destination.write_text('[{"qa": []}]')

    monkeypatch.setattr(DOWNLOAD, "ROOT", tmp_path)
    monkeypatch.setattr(DOWNLOAD, "_fetch", fetch)
    assert DOWNLOAD.main(["download", "memory", "--name", "locomo"]) == 0
    destination = tmp_path / "data/mindmemos_memory/locomo"
    assert json.loads((destination / "locomo10.json").read_text()) == [{"qa": []}]
    meta = DOWNLOAD._read(DOWNLOAD.SOURCES)["memory"]["locomo"]
    assert urls == [meta["files"]["locomo10.json"]]
    DOWNLOAD._verify(meta, destination)


def test_memory_huggingface_preserves_files_revision_and_env_token(tmp_path, monkeypatch):
    """Pin source versions and explicitly pass the configured token to Hugging Face."""
    import huggingface_hub

    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "questions_32k.csv").write_text("question\nhello\n")
    (cache / "shared_contexts_32k.jsonl").write_text("{}\n")
    calls = []

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        return str(cache)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    monkeypatch.setenv(DOWNLOAD.HF_TOKEN, "test-token")
    destination = tmp_path / "payload"
    destination.mkdir()
    meta = DOWNLOAD._read(DOWNLOAD.SOURCES)["memory"]["personamem"]
    DOWNLOAD._materialize(meta, destination, True)
    assert calls == [
        {
            "repo_id": meta["repo"],
            "repo_type": "dataset",
            "revision": meta["revision"],
            "allow_patterns": meta["files"],
            "local_files_only": True,
            "token": "test-token",
        }
    ]
    assert (destination / "questions_32k.csv").read_bytes() == (cache / "questions_32k.csv").read_bytes()
    assert (destination / "shared_contexts_32k.jsonl").exists()


def test_batch_continues_after_failure(tmp_path, monkeypatch, capsys):
    """Continue the selected group and return nonzero for incomplete downloads."""
    calls = []

    def download(meta, destination, offline, verify_only):
        calls.append(destination.name)
        if destination.name == "officeqa":
            raise OSError("network unavailable")
        return "Verified"

    monkeypatch.setattr(DOWNLOAD, "_download", download)
    assert DOWNLOAD.main(["download", "skill", "--all", "--data-root", str(tmp_path)]) == 1
    assert calls == list(DOWNLOAD._read(DOWNLOAD.SOURCES)["skill"])
    output = capsys.readouterr()
    assert "network unavailable" in output.err
    assert "searchqa" in output.out and "Failed" in output.out


def test_searchqa_download_uses_pinned_hf_parquet(tmp_path, monkeypatch):
    """Download native source files without split-specific conversion or Drive access."""
    import huggingface_hub

    cache = tmp_path / "cache/data"
    cache.mkdir(parents=True)
    (cache / "train.parquet").write_bytes(b"source parquet")
    calls = []

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        return str(cache.parent)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    root = tmp_path / "download"
    args = ["download", "skill", "--name", "searchqa", "--data-root", str(root)]
    assert DOWNLOAD.main(args) == 0
    assert (root / "searchqa/data/train.parquet").read_bytes() == b"source parquet"
    assert calls[0]["repo_id"] == "lucadiliello/searchqa"
    assert calls[0]["revision"] == "c1a979068ba118d85467179b704031d113d689cc"
    assert calls[0]["allow_patterns"] == ["data/*.parquet"]
    assert DOWNLOAD.main([*args, "--verify-only"]) == 0
    assert len(calls) == 1


@pytest.mark.skipif(shutil.which("git") is None, reason="Git is required to acquire AgentSkillOS")
def test_agentskillos_download_uses_pinned_git_tree(tmp_path, monkeypatch):
    """Publish the benchmark and seeds at the pinned commit and verify them offline."""
    repository = tmp_path / "upstream"
    repository.mkdir()

    def git(*args):
        return subprocess.check_output(
            [
                "git",
                "-c",
                "user.name=Dataset Test",
                "-c",
                "user.email=test@example.com",
                "-c",
                "commit.gpgsign=false",
                *args,
            ],
            cwd=repository,
            text=True,
        ).strip()

    git("init")
    task = Path("benchmark/AgentSkillOS_bench/tasks/task.json")
    seed = Path("data/skill_seeds/example/SKILL.md")
    excluded = Path("data/unrelated/input.txt")
    for path in (task, seed, excluded):
        (repository / path).parent.mkdir(parents=True, exist_ok=True)
        (repository / path).write_text("pinned")
    git("add", ".")
    git("commit", "-m", "Pin the release")
    revision = git("rev-parse", "HEAD")
    (repository / task).write_text("new")
    git("commit", "-am", "Advance upstream")

    sources = DOWNLOAD._read(DOWNLOAD.SOURCES)
    sources["skill"]["agentskillos"].update(url=repository.as_uri(), revision=revision)
    manifest = tmp_path / "sources.json"
    DOWNLOAD._write(manifest, sources)
    monkeypatch.setattr(DOWNLOAD, "SOURCES", manifest)
    args = ["download", "skill", "--name", "agentskillos", "--data-root", str(tmp_path / "downloads")]
    assert DOWNLOAD.main(args) == 0
    payload = tmp_path / "downloads/agentskillos"
    assert (payload / task).read_text() == (payload / seed).read_text() == "pinned"
    assert not (payload / excluded).exists()
    assert not (payload / ".git").exists()
    assert DOWNLOAD.main([*args, "--offline", "--verify-only"]) == 0


@pytest.mark.parametrize("status", [401, 403, 429])
def test_huggingface_errors_provide_correct_guidance_without_secrets(tmp_path, monkeypatch, capsys, status):
    """Report authentication guidance only for access errors and redact provider text."""
    import huggingface_hub
    import requests
    from huggingface_hub.errors import HfHubHTTPError

    response = requests.Response()
    response.status_code = status

    def fail(**kwargs):
        raise HfHubHTTPError("secret-provider-text", response=response)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fail)
    args = ["download", "skill", "--name", "officeqa", "--data-root", str(tmp_path)]
    assert DOWNLOAD.main(args) == 1
    error = capsys.readouterr().err
    assert ("HF_TOKEN" in error) == (status in (401, 403))
    assert "secret-provider-text" not in error
    assert not list(tmp_path.iterdir())


def test_dotenv_token_loaded_without_overriding_process(tmp_path, monkeypatch):
    """Load repo dotenv credentials while preserving explicit process values."""
    monkeypatch.setattr(DOWNLOAD, "ROOT", tmp_path)
    (tmp_path / ".env").write_text("HF_TOKEN=dotenv-token\n")
    seen = []

    def download(meta, destination, offline, verify_only):
        import os

        seen.append(os.getenv(DOWNLOAD.HF_TOKEN))
        return "Verified"

    monkeypatch.setattr(DOWNLOAD, "_download", download)
    monkeypatch.delenv(DOWNLOAD.HF_TOKEN, raising=False)
    assert DOWNLOAD.main(["download", "skill", "--name", "officeqa"]) == 0
    monkeypatch.setenv(DOWNLOAD.HF_TOKEN, "process-token")
    assert DOWNLOAD.main(["download", "skill", "--name", "officeqa"]) == 0
    assert seen == ["dotenv-token", "process-token"]
