"""Local-only task selection and append-only experiment records."""

import hashlib
import json
import random
from pathlib import Path
from typing import Any

from .config import ExperimentConfig
from .typing import SplitManifest, Task


def write_json(path: Path, value: Any) -> None:
    """Atomically replace a local JSON record.

    Args:
        path: Destination owned by this experiment.
        value: JSON-serializable record.
    """
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path: Path, value: Any) -> None:
    """Append one evidence record without replacing prior records.

    Args:
        path: JSONL event log destination.
        value: JSON-serializable event.
    """
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def create_split(config: ExperimentConfig) -> SplitManifest:
    """Select fifty training and fifty test tasks from local JSONL only.

    Args:
        config: Dataset location, selection seed, and fresh output directory.

    Returns:
        Persisted disjoint task selection without gold or test patches.

    Raises:
        ValueError: The data contains duplicate IDs or fewer than 100 tasks.
        FileExistsError: The output directory already exists.
    """
    raw = config.dataset_path.read_bytes()
    tasks = [Task.model_validate(json.loads(line)) for line in raw.splitlines() if line.strip()]
    if len({task.instance_id for task in tasks}) != len(tasks):
        raise ValueError("Dataset contains duplicate instance IDs")
    if len(tasks) < 100:
        raise ValueError("At least 100 local Verified tasks are required")
    tasks.sort(key=lambda task: task.instance_id)
    random.Random(config.seed).shuffle(tasks)
    manifest = SplitManifest(
        dataset_name=config.dataset_name,
        dataset_sha256=hashlib.sha256(raw).hexdigest(),
        seed=config.seed,
        train=tasks[:50],
        test=tasks[50:100],
    )
    config.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(config.output_dir / "split.json", manifest.model_dump(mode="json"))
    write_json(config.output_dir / "config.json", config.model_dump(mode="json"))
    return manifest


def load_split(config: ExperimentConfig) -> SplitManifest:
    """Read a frozen split and reject changes to its experiment configuration.

    Args:
        config: Configuration used for the current command.

    Returns:
        The previously selected task split.
    """
    saved = json.loads((config.output_dir / "config.json").read_text(encoding="utf-8"))
    if saved != config.model_dump(mode="json"):
        raise ValueError("Configuration differs from the frozen experiment; use a fresh output directory")
    return SplitManifest.model_validate_json((config.output_dir / "split.json").read_text(encoding="utf-8"))
