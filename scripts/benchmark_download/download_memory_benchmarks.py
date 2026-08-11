#!/usr/bin/env python3
"""Download memory benchmark datasets with a cross-platform Python entrypoint."""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = REPO_ROOT / "datasets"
USER_AGENT = "MindMemOS benchmark downloader"

LOCOMO_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
LONGMEMEVAL_URL = (
    "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json"
)
PERSONAMEM_QUESTIONS_URL = (
    "https://huggingface.co/datasets/bowen-upenn/PersonaMem-v1/resolve/main/questions_32k.csv?download=true"
)
PERSONAMEM_CONTEXTS_URL = (
    "https://huggingface.co/datasets/bowen-upenn/PersonaMem-v1/resolve/main/shared_contexts_32k.jsonl?download=true"
)
MEMORYAGENTBENCH_DATASET_ID = "ai-hyz/MemoryAgentBench"
MEMORYAGENTBENCH_SPLIT = "Conflict_Resolution"
MEMORYAGENTBENCH_REVISION = "main"

_BENCHMARK_NAMES = ("locomo", "longmemeval", "memoryagentbench", "personamem")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "benchmarks",
        nargs="*",
        choices=_BENCHMARK_NAMES,
        help="benchmarks to download; omit to download all supported benchmarks",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"dataset root (default: {DEFAULT_DATA_ROOT})",
    )
    return parser.parse_args(argv)


@contextmanager
def atomic_output_path(destination: Path) -> Iterator[Path]:
    """Yield a temporary sibling path and atomically replace the destination on success."""
    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        yield temporary_path
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def download_file(url: str, destination: Path) -> Path:
    """Download one URL without requiring curl or another external executable."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with atomic_output_path(destination) as temporary_path:
        with urllib.request.urlopen(request) as response, temporary_path.open("wb") as output:
            shutil.copyfileobj(response, output)
    return destination.expanduser().resolve()


def download_locomo(data_root: Path) -> Path:
    output = download_file(LOCOMO_URL, data_root / "locomo" / "locomo10.json")
    print(f"LoCoMo downloaded to {output}")
    return output


def download_longmemeval(data_root: Path) -> Path:
    output = download_file(LONGMEMEVAL_URL, data_root / "longmemeval" / "longmemeval_s_cleaned.json")
    print(f"LongMemEval-S downloaded to {output}")
    return output


def _load_memoryagentbench() -> Any:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "MemoryAgentBench download requires the mindmemos-eval dependencies. Run with "
            "`uv run --package mindmemos-eval python "
            "scripts/benchmark_download/download_memory_benchmarks.py memoryagentbench`."
        ) from exc
    return load_dataset(
        MEMORYAGENTBENCH_DATASET_ID,
        split=MEMORYAGENTBENCH_SPLIT,
        revision=MEMORYAGENTBENCH_REVISION,
    )


def download_memoryagentbench(data_root: Path) -> Path:
    output = (data_root / "memoryagentbench" / "conflict_resolution.jsonl").expanduser().resolve()
    dataset = _load_memoryagentbench()
    with atomic_output_path(output) as temporary_path:
        dataset.to_json(str(temporary_path), orient="records", lines=True, force_ascii=False)
    print(f"MemoryAgentBench Conflict Resolution downloaded to {output}")
    return output


def download_personamem(data_root: Path) -> Path:
    output_dir = (data_root / "personamem").expanduser().resolve()
    download_file(PERSONAMEM_QUESTIONS_URL, output_dir / "questions_32k.csv")
    download_file(PERSONAMEM_CONTEXTS_URL, output_dir / "shared_contexts_32k.jsonl")
    print(f"PersonaMem-32K downloaded to {output_dir}")
    return output_dir


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_root = args.data_root.expanduser().resolve()
    selected = tuple(dict.fromkeys(args.benchmarks or _BENCHMARK_NAMES))
    downloaders = {
        "locomo": download_locomo,
        "longmemeval": download_longmemeval,
        "memoryagentbench": download_memoryagentbench,
        "personamem": download_personamem,
    }
    for name in selected:
        downloaders[name](data_root)
    print(f"Memory benchmark root: {data_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
