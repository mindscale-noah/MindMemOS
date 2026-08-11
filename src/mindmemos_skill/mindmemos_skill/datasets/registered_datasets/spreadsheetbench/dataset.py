"""SpreadsheetBench Verified-400 task adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ....registry import ComponentType, register
from ....typing import Task
from ...base import TaskDataset

_SPLITS = {"train": "train", "validation": "val", "test": "test"}


@register(type=ComponentType.DATASET, name="spreadsheetbench_id_split")
class SpreadsheetBenchIdSplitDataset(TaskDataset):
    """Load the same stable ID splits used by the source experiment."""

    def __init__(self, *, data_root: str | Path, split_dir: str | Path | None = None) -> None:
        self.data_root = Path(data_root)
        self.verified_root = self.data_root / "spreadsheetbench_verified_400"
        self.split_root = Path(split_dir) if split_dir is not None else self.data_root / "spreadsheetbench_id_split"
        raw = json.loads((self.verified_root / "dataset.json").read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError("SpreadsheetBench dataset.json must contain a list")
        self._records: dict[str, dict[str, Any]] = {
            str(record["id"]): record for record in raw if isinstance(record, dict) and "id" in record
        }

    def split(self, name: str) -> list[Task]:
        try:
            source_split = _SPLITS[name]
        except KeyError as exc:
            raise ValueError(f"Unsupported SpreadsheetBench split: {name!r}") from exc
        raw = json.loads((self.split_root / source_split / "items.json").read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError(f"SpreadsheetBench {source_split} items.json must contain a list")
        tasks: list[Task] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            task_id = str(item["id"])
            record = self._records[task_id]
            tasks.append(
                Task(
                    task_id=task_id,
                    instruction=str(record["instruction"]),
                    tags=[source_split],
                    metadata={
                        "benchmark": "SpreadsheetBench",
                        "source_split": source_split,
                        "src_dir": str(self.verified_root / str(record["spreadsheet_path"])),
                        "answer_position": str(record["answer_position"]),
                        "answer_sheet": record.get("answer_sheet"),
                        "instruction_type": str(record.get("instruction_type") or ""),
                    },
                )
            )
        return tasks


__all__ = ["SpreadsheetBenchIdSplitDataset"]
