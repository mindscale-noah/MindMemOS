"""LiveMathematicianBench data adapter compatible with Skill-GRPO splits."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

from ....registry import ComponentType, register
from ....typing import Task
from ...base import TaskDataset

SPLIT_DIR_NAMES = {"train": "train", "validation": "val", "test": "test"}
_CHOICE_LABELS = ("A", "B", "C", "D", "E", "F", "G")


@register(type=ComponentType.DATASET, name="livemath_id_split")
class LiveMathIdSplitDataset(TaskDataset):
    """Load official LiveMath JSON files through stable split IDs."""

    def __init__(
        self,
        *,
        data_path: str | Path,
        split_dir: str | Path = "data/LiveMath/livemathematicianbench_id_split",
        seed: int = 42,
        shuffle_choices: bool = True,
    ) -> None:
        self.data_path = Path(data_path)
        self.split_dir = Path(split_dir)
        self.seed = int(seed)
        self.shuffle_choices = bool(shuffle_choices)
        self._payload_by_id: dict[str, dict[str, Any]] | None = None

    def split(self, name: str) -> list[Task]:
        if name not in SPLIT_DIR_NAMES:
            raise ValueError(f"Unsupported LiveMath split: {name!r}")
        source_split = SPLIT_DIR_NAMES[name]
        payload = self._payload()
        tasks: list[Task] = []
        missing: list[str] = []
        for split_item in self._split_items(source_split):
            item_id = str(split_item.get("id") or "")
            item = payload.get(item_id)
            if item is None:
                missing.append(item_id)
                continue
            tasks.append(self._task_from_item(item, source_split, split_item))
        if missing:
            raise ValueError(
                f"Missing {len(missing)} LiveMath payload item(s) for {source_split}; "
                f"first missing id: {missing[0]}"
            )
        return tasks

    def _payload(self) -> dict[str, dict[str, Any]]:
        if self._payload_by_id is not None:
            return self._payload_by_id
        files = self._monthly_files()
        if not files:
            raise ValueError(
                "LiveMath requires data_path to be a qa_*_final.json file or a "
                "directory containing official monthly qa_*_final.json files."
            )
        payload: dict[str, dict[str, Any]] = {}
        for path in files:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw, list):
                raise ValueError(f"Expected JSON array in {path}, got {type(raw).__name__}")
            for row_index, row in enumerate(raw):
                if not isinstance(row, dict):
                    continue
                item = _normalize_item(row, row_index, path)
                if item["question"] and item["choices"] and item["correct_choice"]["label"]:
                    if item["id"] in payload:
                        raise ValueError(f"Duplicate LiveMath item ID: {item['id']}")
                    payload[item["id"]] = item
        if not payload:
            raise ValueError(f"No valid LiveMath items loaded from {self.data_path}")
        self._payload_by_id = payload
        return payload

    def _monthly_files(self) -> list[Path]:
        if self.data_path.is_file():
            return [self.data_path]
        if self.data_path.is_dir():
            return sorted(self.data_path.rglob("qa_*_final.json"))
        return []

    def _split_items(self, split: str) -> list[dict[str, Any]]:
        path = self.split_dir / split / "items.json"
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError(f"{path} must contain a list of LiveMath split items")
        return [item for item in raw if isinstance(item, dict)]

    def _task_from_item(self, item: dict[str, Any], split: str, split_item: dict[str, Any]) -> Task:
        task_item = self._shuffle_item_choices(item)
        return Task(
            task_id=task_item["id"],
            instruction=_build_user(task_item["question"], task_item["choices"]),
            tags=[split],
            metadata={
                "benchmark": "LiveMathematicianBench",
                "source_split": split,
                "source_file": split_item.get("source_file", ""),
                **task_item,
            },
        )

    def _shuffle_item_choices(self, item: dict[str, Any]) -> dict[str, Any]:
        choices = [dict(choice) for choice in item["choices"]]
        correct_choice = dict(item["correct_choice"])
        if not self.shuffle_choices:
            return {**item, "choices": choices, "correct_choice": correct_choice}

        digest = hashlib.sha256(f"{self.seed}:{item['id']}".encode()).hexdigest()
        random.Random(int(digest[:16], 16)).shuffle(choices)
        original_correct = _normalize_label(correct_choice["label"])
        remapped: list[dict[str, str]] = []
        for index, choice in enumerate(choices):
            label = _CHOICE_LABELS[index]
            remapped.append({"label": label, "text": choice["text"]})
            if _normalize_label(choice["label"]) == original_correct:
                correct_choice = {"label": label, "text": choice["text"]}
        return {**item, "choices": remapped, "correct_choice": correct_choice}


def _normalize_item(item: dict[str, Any], row_index: int, source_path: Path) -> dict[str, Any]:
    mcq = item.get("mcq") if isinstance(item.get("mcq"), dict) else {}
    question = str(mcq.get("question") or item.get("question") or "").strip()
    choices = _coerce_choices(mcq.get("choices") or item.get("choices") or [])
    raw_correct = mcq.get("correct_choice") or item.get("correct_choice") or {}
    if isinstance(raw_correct, dict):
        correct_label = _normalize_label(raw_correct.get("label", ""))
        correct_text = str(raw_correct.get("text") or "").strip()
    else:
        correct_label, correct_text = _normalize_label(raw_correct), ""
    by_label = {_normalize_label(choice["label"]): choice["text"] for choice in choices}
    if correct_label and not correct_text:
        correct_text = by_label.get(correct_label, "")
    if correct_label and correct_text and correct_label not in by_label:
        choices.append({"label": correct_label, "text": correct_text})
    month = str(item.get("month") or "").strip()
    number = item.get("no", row_index + 1)
    return {
        "id": f"{month}:{number}" if month else str(number),
        "month": month,
        "no": number,
        "paper_link": str(item.get("paper_link") or "").strip(),
        "theorem": str(item.get("theorem") or "").strip(),
        "sketch": str(item.get("sketch") or "").strip(),
        "theorem_type": _as_strings(item.get("theorem_type")),
        "question": question,
        "choices": choices,
        "correct_choice": {"label": correct_label, "text": correct_text},
        "payload_source_path": str(source_path),
    }


def _coerce_choices(raw_choices: Any) -> list[dict[str, str]]:
    if isinstance(raw_choices, list):
        choices: list[dict[str, str]] = []
        for index, choice in enumerate(raw_choices):
            if isinstance(choice, dict):
                label = str(choice.get("label") or _CHOICE_LABELS[index]).strip()
                text = str(choice.get("text") or choice.get("content") or "").strip()
            else:
                label, text = _CHOICE_LABELS[index], str(choice).strip()
            if text:
                choices.append({"label": label, "text": text})
        return choices
    if isinstance(raw_choices, dict):
        return [
            {"label": str(label).strip(), "text": str(text).strip()}
            for label, text in sorted(raw_choices.items())
            if str(text).strip()
        ]
    return []


def _as_strings(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if value is not None and str(value).strip() else []


def _normalize_label(value: Any) -> str:
    return str(value).strip().upper().rstrip(".):")


def _build_user(question: str, choices: list[dict[str, str]]) -> str:
    formatted = "\n".join(f"{choice['label']}. {choice['text']}" for choice in choices)
    return f"## Question\n{question}\n\n## Choices\n{formatted}"


__all__ = ["LiveMathIdSplitDataset"]
