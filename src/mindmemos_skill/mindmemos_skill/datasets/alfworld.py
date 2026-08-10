"""ALFWorld gamefile-path split adapter."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ..envs.registered_envs.alfworld import SYSTEM_PROMPT
from ..registry import ComponentType, register
from ..typing import Task
from .base import TaskDataset

SPLIT_DIR_NAMES = {"train": "train", "validation": "val", "test": "test"}
TASK_TYPES = (
    "pick_and_place_simple",
    "pick_and_place",
    "pick_two_obj_and_place",
    "look_at_obj_in_light",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_clean_then_place_in_recep",
)


@register(type=ComponentType.DATASET, name="alfworld_path_split")
class ALFWorldPathSplitDataset(TaskDataset):
    """Load the same gamefile manifests used by Skill-GRPO."""

    def __init__(
        self,
        *,
        split_dir: str | Path = "data/ALFWorld/alfworld_path_split",
        alfworld_data: str | Path | None = None,
    ) -> None:
        self.split_dir = Path(split_dir)
        data_root = alfworld_data or os.getenv("ALFWORLD_DATA")
        self.alfworld_data = Path(data_root).expanduser() if data_root else None

    def split(self, name: str) -> list[Task]:
        if name not in SPLIT_DIR_NAMES:
            raise ValueError(f"Unsupported ALFWorld split: {name!r}")
        source_split = SPLIT_DIR_NAMES[name]
        return [
            self._task_from_item(item, source_split, index)
            for index, item in enumerate(self._load_items(source_split))
        ]

    def _load_items(self, split: str) -> list[dict[str, Any]]:
        path = self.split_dir / split / "items.json"
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, list):
            raise ValueError(f"{path} must contain a list of ALFWorld split items")
        return [item for item in data if isinstance(item, dict)]

    def _task_from_item(self, item: dict[str, Any], split: str, index: int) -> Task:
        gamefile = str(item.get("gamefile") or "").strip()
        if not gamefile:
            raise ValueError(f"ALFWorld split item missing gamefile: {item!r}")
        task_type = str(item.get("task_type") or _task_type(gamefile))
        case_id = str(item.get("id") or f"{split}:{index:04d}")
        return Task(
            task_id=case_id,
            instruction=(
                f"Complete the ALFWorld household task from gamefile "
                f"{gamefile}. Task type: {task_type}."
            ),
            system_prompt=SYSTEM_PROMPT,
            tags=[split],
            metadata={
                "benchmark": "ALFWorld",
                "source_split": split,
                "gamefile": gamefile,
                "resolved_gamefile": self._resolve_gamefile(gamefile),
                "task_type": task_type,
                "instruction_type": task_type,
            },
        )

    def _resolve_gamefile(self, gamefile: str) -> str:
        path = Path(gamefile)
        if path.is_absolute():
            return str(path)
        if self.alfworld_data is not None:
            return str(self.alfworld_data / path)
        return gamefile


def _task_type(gamefile: str) -> str:
    for task_type in TASK_TYPES:
        if task_type in gamefile:
            return task_type
    return "other"


__all__ = ["ALFWorldPathSplitDataset"]
