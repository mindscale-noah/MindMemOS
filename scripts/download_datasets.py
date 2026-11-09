#!/usr/bin/env python3
"""Download official benchmark data independently of algorithms and experiment splits."""

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from dotenv import load_dotenv

if __package__:
    from .constants import HF_TOKEN
else:
    from constants import HF_TOKEN

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "config/presets/dataset_download_sources.json"
RECEIPT = "download_receipt.json"


def _read(path):
    """Read a UTF-8 JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path, value):
    """Write deterministic UTF-8 JSON."""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _hash(path):
    """Hash a file without loading it into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _checksums(destination):
    """Collect payload hashes excluding the download receipt."""
    return {
        path.relative_to(destination).as_posix(): _hash(path)
        for path in sorted(destination.rglob("*"))
        if path.is_file() and path != destination / RECEIPT
    }


def _fetch(url, destination):
    """Download an official file into the staging directory."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)


def _extract(source, destination):
    """Extract an upstream archive while rejecting paths outside the destination."""
    if zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as archive:
            for member in archive.infolist():
                target = (destination / member.filename).resolve()
                if not target.is_relative_to(destination.resolve()):
                    raise ValueError(f"Unsafe archive path: {member.filename}")
            archive.extractall(destination)
    else:
        with tarfile.open(source) as archive:
            archive.extractall(destination, filter="data")


def _materialize(meta, destination, offline):
    """Acquire official data without selecting or modifying experiment cases."""
    archive_hashes = {}
    with tempfile.TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        if meta["kind"] == "huggingface":
            from huggingface_hub import snapshot_download
            from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

            try:
                source = Path(
                    snapshot_download(
                        repo_id=meta["repo"],
                        repo_type="dataset",
                        revision=meta["revision"],
                        allow_patterns=meta["files"],
                        local_files_only=offline,
                        token=os.getenv(HF_TOKEN) or None,
                    )
                )
            except GatedRepoError as error:
                raise RuntimeError(
                    f"Cannot access gated dataset {meta['repo']}. Set HF_TOKEN in {ROOT / '.env'} "
                    f"and request or accept dataset access at {meta['url']}, then retry."
                ) from error
            except HfHubHTTPError as error:
                status = error.response.status_code if error.response is not None else None
                if status in (401, 403):
                    raise RuntimeError(
                        f"Hugging Face access denied ({status}) for {meta['repo']}. "
                        f"Set or update HF_TOKEN in {ROOT / '.env'} and check dataset permissions "
                        f"at {meta['url']}, then retry the same uv download command."
                    ) from error
                raise RuntimeError(f"Hugging Face request failed for {meta['repo']} (HTTP {status}).") from error
            for pattern in meta["files"]:
                paths = sorted(source.glob(pattern))
                if not paths:
                    raise ValueError(f"No upstream files match {pattern}")
                for path in paths:
                    if path.relative_to(source).as_posix() in meta.get("extract", []):
                        archive_hashes[path.name] = _hash(path)
                        _extract(path, destination)
                    else:
                        target = destination / path.relative_to(source)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(path, target)
        else:
            if offline:
                raise ValueError("Official archive downloads require network access; use --verify-only for local data.")
            for name, url in meta.get("archives", {}).items():
                archive = temporary / name
                _fetch(url, archive)
                archive_hashes[name] = _hash(archive)
                if meta.get("strip_root"):
                    unpacked = temporary / "unpacked"
                    unpacked.mkdir()
                    _extract(archive, unpacked)
                    roots = list(unpacked.iterdir())
                    if len(roots) != 1 or not roots[0].is_dir():
                        raise ValueError("Expected one upstream archive root directory.")
                    shutil.copytree(roots[0], destination, dirs_exist_ok=True)
                else:
                    _extract(archive, destination)
            for name, url in meta.get("files", {}).items():
                _fetch(url, destination / name)
    return archive_hashes


def _verify(meta, destination):
    """Check the recorded source version and exact local file hashes."""
    receipt = _read(destination / RECEIPT)
    if receipt["source"] != meta:
        raise ValueError(f"Source version changed for {destination}; use a new --data-root.")
    actual = _checksums(destination)
    expected = receipt["checksums"]
    if not expected or actual != expected:
        changed = sorted(key for key in actual.keys() | expected.keys() if actual.get(key) != expected.get(key))
        raise ValueError(f"Payload mismatch in {destination}: {changed[:10]}")
    print(f"Verified {destination.name}: {len(actual)} files", flush=True)


def _download(meta, destination, offline, verify_only):
    """Verify an existing payload or publish a complete staged download."""
    if verify_only or destination.exists():
        _verify(meta, destination)
        return "Verified"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".download-") as temporary:
        staged = Path(temporary) / "payload"
        staged.mkdir()
        archives = _materialize(meta, staged, offline)
        checksums = _checksums(staged)
        if not checksums:
            raise ValueError("The official source produced an empty payload.")
        _write(staged / RECEIPT, {"source": meta, "archives": archives, "checksums": checksums})
        _verify(meta, staged)
        staged.rename(destination)
    print(f"Downloaded to {destination}", flush=True)
    return "Downloaded"


def main(argv: list[str] | None = None) -> int:
    """Download or verify official datasets in one benchmark group.

    Args:
        argv: Command arguments, or None to read process arguments.

    Returns:
        Zero when all selected datasets succeed, or one when any fails.
        Invalid arguments exit with code two through argparse.
    """
    sources = _read(SOURCES)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("list", "download"))
    parser.add_argument("group", choices=sorted(sources), help="Benchmark group.")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--name", help="Dataset name within the selected group.")
    selection.add_argument("--all", action="store_true", help="Download every dataset in this group.")
    parser.add_argument("--data-root", type=Path, help="Default: data/mindmemos_<group>.")
    parser.add_argument("--verify-only", action="store_true", help="Verify local payloads without downloading.")
    parser.add_argument("--offline", action="store_true", help="Use cached Hugging Face files or verify local data.")
    args = parser.parse_args(argv)
    datasets = sources[args.group]
    if args.action == "list":
        if args.name or args.all or args.verify_only or args.offline or args.data_root:
            parser.error("list accepts only a benchmark group")
        for name, meta in datasets.items():
            print(f"{name}: {meta.get('download_page', meta['url'])} [{meta['kind']}]")
        return 0
    if not args.name and not args.all:
        parser.error("download requires --name DATASET or --all")
    if args.name and args.name not in datasets:
        parser.error(f"Unknown dataset {args.name!r} for {args.group}; choose from: {', '.join(datasets)}")
    load_dotenv(ROOT / ".env", override=False, interpolate=False)
    data_root = args.data_root or ROOT / f"data/mindmemos_{args.group}"
    selected = datasets if args.all else {args.name: datasets[args.name]}
    outcomes = []
    for name, meta in selected.items():
        print(f"\nDownloading {args.group}/{name}...", flush=True)
        try:
            status = _download(meta, data_root / name, args.offline, args.verify_only)
        except Exception as error:
            status = "Failed"
            print(f"Failed {args.group}/{name}: {error}", file=sys.stderr, flush=True)
        outcomes.append((name, status))
    print("\nDataset                 Result")
    for name, status in outcomes:
        print(f"{name:<23} {status}")
    return 1 if any(status == "Failed" for _, status in outcomes) else 0


if __name__ == "__main__":
    raise SystemExit(main())
