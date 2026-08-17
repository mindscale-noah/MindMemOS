"""Resolve registered Dataset and Env components for experiment adapters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

from mindmemos_skill.datasets import TaskDataset
from mindmemos_skill.registry import ComponentType, create

DEFAULT_DATASET_REFS = {
    "alfworld": "alfworld_path_split",
    "livemath": "livemath_id_split",
    "officeqa": "officeqa_id_split",
    "spreadsheetbench": "spreadsheetbench_id_split",
}


def json_object(value: str) -> dict[str, Any]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("expected a JSON object")
    return parsed


def resolve_dataset_ref(*, benchmark: str, dataset_ref: str | None) -> str:
    resolved = dataset_ref or DEFAULT_DATASET_REFS.get(benchmark)
    if resolved is None:
        raise ValueError(f"benchmark {benchmark!r} has no default dataset; set --dataset-ref")
    return resolved


def resolve_env_ref(*, benchmark: str, env_ref: str | None) -> str:
    return env_ref or ("alfworld_bounded_history" if benchmark == "alfworld" else benchmark)


def build_registered_dataset(
    *,
    benchmark: str,
    dataset_ref: str | None,
    dataset_options: dict[str, Any] | None,
    data_root: Path | None,
    split_dir: Path | None,
    seed: int,
    shuffle_choices: bool,
) -> TaskDataset:
    resolved_ref = resolve_dataset_ref(benchmark=benchmark, dataset_ref=dataset_ref)
    options = dict(dataset_options or {})
    if resolved_ref == "alfworld_path_split":
        options.setdefault("split_dir", split_dir or Path("data/ALFWorld/alfworld_path_split"))
        if data_root is not None:
            options.setdefault("alfworld_data", data_root)
    elif resolved_ref == "livemath_id_split":
        if data_root is None and "data_path" not in options:
            raise ValueError("livemath_id_split requires --data-root or dataset_options.data_path")
        if data_root is not None:
            options.setdefault("data_path", data_root)
        options.setdefault("split_dir", split_dir or Path("data/LiveMath/livemathematicianbench_id_split"))
        options.setdefault("seed", seed)
        options.setdefault("shuffle_choices", shuffle_choices)
    elif resolved_ref == "officeqa_id_split":
        if data_root is None and "data_root" not in options:
            raise ValueError("officeqa_id_split requires --data-root or dataset_options.data_root")
        if data_root is not None:
            options.setdefault("data_root", data_root)
        if split_dir is not None:
            options.setdefault("split_dir", split_dir)
    elif resolved_ref == "spreadsheetbench_id_split":
        if data_root is None and "data_root" not in options:
            raise ValueError("spreadsheetbench_id_split requires --data-root or dataset_options.data_root")
        if data_root is not None:
            options.setdefault("data_root", data_root)
        if split_dir is not None:
            options.setdefault("split_dir", split_dir)
    return cast(TaskDataset, create(type=ComponentType.DATASET, name=resolved_ref, **options))


def build_registered_env_options(
    *,
    benchmark: str,
    env_ref: str | None,
    env_options: dict[str, Any] | None,
    max_turns: int,
    env_seed: int,
    shell_timeout: int,
    use_theorem: bool,
    use_sketch: bool,
    docs_root: Path | None = None,
) -> tuple[str, dict[str, Any]]:
    resolved_ref = resolve_env_ref(benchmark=benchmark, env_ref=env_ref)
    options = dict(env_options or {})
    if resolved_ref in {"alfworld_bounded_history", "livemath", "officeqa", "spreadsheetbench"}:
        options.setdefault("max_turns", max_turns)
    if resolved_ref == "alfworld_bounded_history":
        options.setdefault("seed", env_seed)
    elif resolved_ref == "spreadsheetbench":
        options.setdefault("shell_timeout_seconds", shell_timeout)
    elif resolved_ref == "livemath":
        options.setdefault("use_theorem", use_theorem)
        options.setdefault("use_sketch", use_sketch)
    elif resolved_ref == "officeqa":
        if docs_root is not None:
            options.setdefault("docs_root", str(docs_root))
        if "docs_root" not in options:
            raise ValueError("officeqa requires --docs-root or env_options.docs_root")
    return resolved_ref, options


__all__ = [
    "DEFAULT_DATASET_REFS",
    "build_registered_dataset",
    "build_registered_env_options",
    "json_object",
    "resolve_dataset_ref",
    "resolve_env_ref",
]
