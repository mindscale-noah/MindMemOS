"""Resolve official downloads and preserve committed split membership."""

import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as parquet
import yaml
from mindmemos_skill.common.json import read_json, write_json
from mindmemos_skill.envs.data import Data
from mindmemos_skill.skill import Skill
from pydantic import JsonValue

from .config import ROOT, EvalConfig
from .typing import BenchmarkName, DatasetInputs


def _livemath(root: Path, resources: Path, run_dir: Path) -> tuple[Data, dict[str, JsonValue]]:
    """Read monthly records directly from the official download."""
    return Data.build("livemath", {"data_path": root / "data"}), {}


def _alfworld(root: Path, resources: Path, run_dir: Path) -> tuple[Data, dict[str, JsonValue]]:
    """Resolve upstream simulator paths without modifying its downloaded YAML."""
    simulator = yaml.safe_load(
        (root / "configs/base_config.yaml").read_text(encoding="utf-8").replace("$ALFWORLD_DATA", str(root))
    )
    return Data.build(
        "alfworld_bounded_history", {"data_path": resources / "data_splits/games.json", "game_root": root}
    ), {"simulator": simulator}


def _spreadsheetbench(root: Path, resources: Path, run_dir: Path) -> tuple[Data, dict[str, JsonValue]]:
    """Use Verified 400 metadata and private workbook references in place."""
    release = root / "spreadsheetbench_verified_400"
    return Data.build("spreadsheetbench", {"data_path": release / "dataset.json", "data_root": release}), {}


def _officeqa(root: Path, resources: Path, run_dir: Path) -> tuple[Data, dict[str, JsonValue]]:
    """Use the official CSV and nested parsed document corpus."""
    return Data.build("officeqa", {"data_path": root / "officeqa_full.csv", "docs_root": root}), {}


def _searchqa(root: Path, resources: Path, run_dir: Path) -> tuple[Data, dict[str, JsonValue]]:
    """Select pinned HF records using their native keys without changing input text."""
    splits = read_json(resources / "data_splits/splits.json")
    expected_ids = [identifier for ids in splits.values() for identifier in ids]
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("SearchQA splits contain duplicate task IDs")
    wanted = set(expected_ids)
    records = {}
    paths = sorted((root / "data").glob("*.parquet"))
    if not paths:
        raise ValueError(f"No SearchQA parquet files found under {root / 'data'}")
    for path in paths:
        for batch in parquet.ParquetFile(path).iter_batches(columns=["key", "question", "context", "answers"]):
            for row in batch.to_pylist():
                identifier = row["key"]
                if identifier not in wanted:
                    continue
                if identifier in records:
                    raise ValueError(f"Duplicate SearchQA source key: {identifier}")
                records[identifier] = {
                    "id": identifier,
                    "question": row["question"],
                    "context": row["context"],
                    "answers": row["answers"],
                }
    missing = wanted - records.keys()
    if missing:
        raise ValueError(f"SearchQA source is missing {len(missing)} split IDs: {sorted(missing)[:5]}")
    destination = run_dir / "prepared/searchqa/items.json"
    write_json(destination, [records[identifier] for identifier in expected_ids])
    return Data.build("searchqa", {"data_path": destination}), {}


def load_dataset(benchmark: BenchmarkName, data_root: Path, run_dir: Path, config: EvalConfig) -> DatasetInputs:
    """Load, verify, and split one official benchmark without rewriting downloads.

    Args:
        benchmark: One of the five explicitly supported benchmarks.
        data_root: Download parent directory.
        run_dir: Benchmark output directory for necessary prepared data.
        config: Shared target model and native environment overrides.

    Returns:
        Complete ordered train/val/test pools and native environment options.

    Raises:
        ValueError: Receipt, membership, references, or source records are invalid.
        OSError: Required data or resources are missing.
    """
    root = (data_root / benchmark).expanduser().resolve()
    resources = ROOT / f"resources/mindmemos_skill_eval/skill_forest-{benchmark}"
    loaders = {
        "livemath": _livemath,
        "alfworld": _alfworld,
        "spreadsheetbench": _spreadsheetbench,
        "searchqa": _searchqa,
        "officeqa": _officeqa,
    }
    data, prepared = loaders[benchmark](root, resources, run_dir)
    verification = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/download_datasets.py"),
            "download",
            "skill",
            "--name",
            benchmark,
            "--data-root",
            str(data_root.expanduser().resolve()),
            "--verify-only",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if verification.returncode:
        raise ValueError(
            f"Download receipt verification failed: {verification.stdout.strip()} {verification.stderr.strip()}"
        )
    split_file = resources / "data_splits/splits.json"
    splits = data.split(data.load(), split_file=split_file)
    if set(splits) != {"train", "val", "test"} or len(splits["train"]) < 2 or not splits["test"]:
        raise ValueError("Fixed splits require train, val, test and at least two train samples and one test sample")
    initial_skill = resources / "initial_skill"
    Skill.from_path(initial_skill)
    return DatasetInputs(
        splits=splits,
        environment=config.environment(benchmark, prepared),
        initial_skill=initial_skill,
        split_file=split_file,
        source=root,
    )
