#!/usr/bin/env python3
"""Download official benchmark data independently of algorithms and experiment splits."""

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "config/datasets/sources.json"
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


def _materialize(meta, destination, offline, source_archives):
    """Acquire official data without selecting or modifying experiment cases."""
    archive_hashes = {}
    with tempfile.TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        if meta["kind"] == "huggingface":
            from huggingface_hub import snapshot_download

            source = Path(
                snapshot_download(
                    repo_id=meta["repo"],
                    repo_type="dataset",
                    revision=meta["revision"],
                    allow_patterns=meta["files"],
                    local_files_only=offline,
                )
            )
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
        elif meta["kind"] == "manual":
            if not source_archives:
                raise ValueError(
                    f"Download the official archives from {meta['download_page']} "
                    "(DocVQA requires RRC login), then pass --source-archive for each archive."
                )
            for path in source_archives:
                archive_hashes[path.name] = _hash(path)
                _extract(path, destination)
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


def main():
    """Download a dataset or verify its existing payload."""
    sources = _read(SOURCES)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("list", "download"))
    parser.add_argument("--benchmark", choices=sorted(sources), required=True, help="Benchmark group.")
    parser.add_argument("--dataset", help="Dataset name within the selected benchmark group.")
    parser.add_argument("--data-root", type=Path, help="Default: data/mindmemos_<benchmark>.")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--offline", action="store_true", help="Use cached Hugging Face files or local archives only.")
    parser.add_argument(
        "--source-archive",
        type=Path,
        action="append",
        default=[],
        help="Official SearchQA or DocVQA archive; repeat for multiple archives.",
    )
    args = parser.parse_args()
    datasets = sources[args.benchmark]
    if args.dataset and args.dataset not in datasets:
        parser.error(f"Unknown dataset {args.dataset!r} for {args.benchmark}; choose from: {', '.join(datasets)}")
    if args.action == "list":
        for name, meta in datasets.items():
            print(f"{name}: {meta.get('download_page', meta['url'])} [{meta['kind']}]")
        return
    if not args.dataset:
        parser.error("download requires --dataset; use list to view available datasets.")
    meta = datasets[args.dataset]
    if args.data_root is None:
        args.data_root = ROOT / f"data/mindmemos_{args.benchmark}"
    if args.source_archive and meta["kind"] != "manual":
        parser.error("--source-archive is only supported for SearchQA and DocVQA.")
    destination = args.data_root / args.dataset
    if args.verify_only or destination.exists():
        _verify(meta, destination)
        return
    args.data_root.mkdir(parents=True, exist_ok=True)
    # Stage on the destination filesystem and publish only complete payloads.
    with tempfile.TemporaryDirectory(dir=args.data_root, prefix=".download-") as temporary:
        staged = Path(temporary) / "payload"
        staged.mkdir()
        archives = _materialize(meta, staged, args.offline, args.source_archive)
        checksums = _checksums(staged)
        if not checksums:
            raise ValueError("The official source produced an empty payload.")
        _write(staged / RECEIPT, {"source": meta, "archives": archives, "checksums": checksums})
        _verify(meta, staged)
        staged.rename(destination)
    print(f"Downloaded {args.benchmark}/{args.dataset} to {destination}")


if __name__ == "__main__":
    main()
