#!/usr/bin/env python3
"""Download large benchmark payloads used by MindMemOS Skill experiments."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "mindmemos_skill"

LIVEMATH_REPO_ID = "LiveMathematicianBench/LiveMathematicianBench"
LIVEMATH_REVISION = "b72450f6ce96c26158d64d945a5d31ef7727be41"
LIVEMATH_SOURCE_FILES = (
    "data/202511/qa_202511_final.json",
    "data/202512/qa_202512_final.json",
    "data/202601/qa_202601_final.json",
    "data/202602/qa_202602_final.json",
)

SPREADSHEETBENCH_REPO_ID = "KAKA22/SpreadsheetBench"
SPREADSHEETBENCH_REVISION = "ab0b742b0fc95b946f212d80ac7771b5531272e4"
SPREADSHEETBENCH_ARCHIVE = "spreadsheetbench_verified_400.tar.gz"

_DATASET_NAMES = ("alfworld", "livemath", "spreadsheetbench")
_ALFWORLD_EXPECTED_PATHS = (
    "json_2.1.1/train",
    "json_2.1.1/valid_seen",
    "json_2.1.1/valid_unseen",
    "logic/alfred.pddl",
    "logic/alfred.twl2",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "datasets",
        nargs="*",
        choices=_DATASET_NAMES,
        help="datasets to download; omit to download all supported datasets",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"payload root (default: {DEFAULT_DATA_ROOT})",
    )
    parser.add_argument("--force", action="store_true", help="refresh cached downloads and overwrite payload files")
    return parser.parse_args(argv)


def _huggingface_hub() -> tuple[Any, Any]:
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "Hugging Face downloads require the dataset-download extra. Run with "
            "`uv run --package mindmemos-skill --extra dataset-download python "
            "scripts/benchmark_download/download_mindmemos_skill_datasets.py`."
        ) from exc
    return hf_hub_download, snapshot_download


def download_alfworld(data_root: Path, *, force: bool = False) -> Path:
    target = data_root / "alfworld"
    if not force and all((target / relative).exists() for relative in _ALFWORLD_EXPECTED_PATHS):
        print(f"ALFWorld is ready: {target}")
        return target
    executable = shutil.which("alfworld-download")
    if executable is None:
        raise RuntimeError(
            "ALFWorld downloads require the dataset-download extra. Run with "
            "`uv run --package mindmemos-skill --extra dataset-download python "
            "scripts/benchmark_download/download_mindmemos_skill_datasets.py alfworld`."
        )

    target.mkdir(parents=True, exist_ok=True)
    command = [executable, "--data-dir", str(target)]
    if force:
        command.extend(("--force", "--force-download"))
    subprocess.run(command, check=True)
    missing = [relative for relative in _ALFWORLD_EXPECTED_PATHS if not (target / relative).exists()]
    if missing:
        raise RuntimeError(f"ALFWorld download is incomplete; missing: {', '.join(missing)}")
    print(f"ALFWorld is ready: {target}")
    return target


def download_livemath(data_root: Path, *, force: bool = False) -> Path:
    target = data_root / "livemath" / "raw"
    expected = [target / relative for relative in LIVEMATH_SOURCE_FILES]
    if not force and all(path.is_file() for path in expected):
        print(f"LiveMath is ready: {target}")
        return target

    _, snapshot_download = _huggingface_hub()
    snapshot = Path(
        snapshot_download(
            repo_id=LIVEMATH_REPO_ID,
            repo_type="dataset",
            revision=LIVEMATH_REVISION,
            allow_patterns=list(LIVEMATH_SOURCE_FILES),
            force_download=force,
        )
    )
    for relative in LIVEMATH_SOURCE_FILES:
        source = snapshot / relative
        if not source.is_file():
            raise FileNotFoundError(f"LiveMath snapshot is missing {relative}")
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    print(f"LiveMath is ready: {target}")
    return target


def download_spreadsheetbench(data_root: Path, *, force: bool = False) -> Path:
    target = data_root / "spreadsheetbench"
    verified = target / "spreadsheetbench_verified_400"
    if not force and (verified / "dataset.json").is_file():
        print(f"SpreadsheetBench is ready: {target}")
        return target

    hf_hub_download, _ = _huggingface_hub()
    archive_path = Path(
        hf_hub_download(
            repo_id=SPREADSHEETBENCH_REPO_ID,
            repo_type="dataset",
            revision=SPREADSHEETBENCH_REVISION,
            filename=SPREADSHEETBENCH_ARCHIVE,
            force_download=force,
        )
    )
    target.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix=".spreadsheetbench-", dir=target))
    try:
        _safe_extract_tar(archive_path, temporary_root)
        extracted = temporary_root / "spreadsheetbench_verified_400"
        if not (extracted / "dataset.json").is_file():
            raise FileNotFoundError(f"{SPREADSHEETBENCH_ARCHIVE} does not contain the verified-400 payload")
        shutil.copytree(extracted, verified, dirs_exist_ok=True)
    finally:
        shutil.rmtree(temporary_root)
    print(f"SpreadsheetBench is ready: {target}")
    return target


def _safe_extract_tar(archive_path: Path, destination: Path) -> None:
    """Extract regular files and directories while rejecting links and path traversal."""
    destination = destination.resolve()
    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member in archive.getmembers():
            target = (destination / member.name).resolve()
            if target != destination and destination not in target.parents:
                raise RuntimeError(f"Refusing to extract unsafe tar member: {member.name}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise RuntimeError(f"Refusing to extract non-regular tar member: {member.name}")
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"Could not read tar member: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_root = args.data_root.expanduser().resolve()
    selected = tuple(dict.fromkeys(args.datasets or _DATASET_NAMES))
    downloaders = {
        "alfworld": download_alfworld,
        "livemath": download_livemath,
        "spreadsheetbench": download_spreadsheetbench,
    }
    for name in selected:
        downloaders[name](data_root, force=args.force)
    print(f"Dataset payload root: {data_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
