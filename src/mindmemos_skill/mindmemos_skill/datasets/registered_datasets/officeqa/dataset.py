"""OfficeQA Full adapter using SkillOpt's stable ID splits."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from ....registry import ComponentType, register
from ....typing import Task
from ...base import TaskDataset

_SPLITS = {"train": "train", "validation": "val", "test": "test"}


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        loaded = None
    if isinstance(loaded, list):
        return [str(item).strip() for item in loaded if str(item).strip()]
    if "\n" in text:
        return [part.strip() for part in text.splitlines() if part.strip()]
    if "," in text and not text.lower().endswith(".txt"):
        return [part.strip() for part in text.split(",") if part.strip()]
    return [text]


def _normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    task_id = str(record.get("uid") or record.get("id") or "").strip()
    answer = str(record.get("ground_truth") or record.get("answer") or "").strip()
    category = str(record.get("category") or record.get("difficulty") or "officeqa").strip() or "officeqa"
    return {
        "id": task_id,
        "question": str(record.get("question") or "").strip(),
        "ground_truth": answer,
        "answers": [answer] if answer else [],
        "category": category,
        "source_files": _as_list(record.get("source_files")),
        "source_docs": _as_list(record.get("source_docs")),
    }


@register(type=ComponentType.DATASET, name="officeqa_id_split")
class OfficeQAIdSplitDataset(TaskDataset):
    """Join SkillOpt's OfficeQA split manifests to the gated full payload."""

    def __init__(
        self,
        *,
        data_root: str | Path,
        split_dir: str | Path | None = None,
        data_path: str | Path | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.split_root = Path(split_dir) if split_dir is not None else self.data_root / "officeqa_id_split"
        payload_path = Path(data_path) if data_path is not None else self._find_payload()
        self._records = self._load_payload(payload_path)

    def _find_payload(self) -> Path:
        for name in ("officeqa_full.csv", "officeqa_full.json", "officeqa.csv", "dataset.json"):
            candidate = self.data_root / name
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(
            f"OfficeQA payload not found under {self.data_root}; run the dataset downloader with `officeqa`"
        )

    @staticmethod
    def _load_payload(path: Path) -> dict[str, dict[str, Any]]:
        if path.suffix.lower() == ".csv":
            with path.open(encoding="utf-8", newline="") as stream:
                rows: list[dict[str, Any]] = list(csv.DictReader(stream))
        else:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw, list):
                raise ValueError(f"OfficeQA payload must contain a list: {path}")
            rows = [row for row in raw if isinstance(row, dict)]
        records = {}
        for row in rows:
            record = _normalize_record(row)
            if record["id"]:
                records[record["id"]] = record
        if not records:
            raise ValueError(f"No OfficeQA records loaded from {path}")
        return records

    def split(self, name: str) -> list[Task]:
        try:
            source_split = _SPLITS[name]
        except KeyError as exc:
            raise ValueError(f"Unsupported OfficeQA split: {name!r}") from exc
        split_path = self.split_root / source_split / "items.json"
        raw = json.loads(split_path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError(f"{split_path} must contain a list")

        tasks: list[Task] = []
        missing: list[str] = []
        for split_item in raw:
            if not isinstance(split_item, dict):
                continue
            task_id = str(split_item.get("id") or split_item.get("uid") or "").strip()
            payload = self._records.get(task_id)
            if payload is None or not payload["question"] or not payload["ground_truth"]:
                missing.append(task_id)
                continue
            source_files = _as_list(split_item.get("source_files")) or payload["source_files"]
            source_docs = _as_list(split_item.get("source_docs")) or payload["source_docs"]
            category = str(split_item.get("category") or payload["category"])
            tasks.append(
                Task(
                    task_id=task_id,
                    instruction=payload["question"],
                    tags=[source_split, category],
                    metadata={
                        "benchmark": "OfficeQA",
                        "source_split": source_split,
                        "uid": task_id,
                        "category": category,
                        "ground_truth": payload["ground_truth"],
                        "answers": payload["answers"],
                        "source_files": source_files,
                        "source_docs": source_docs,
                    },
                )
            )
        if missing:
            raise ValueError(
                f"Missing {len(missing)} OfficeQA payload item(s) for {source_split}; first missing id: {missing[0]}"
            )
        return tasks


__all__ = ["OfficeQAIdSplitDataset"]
