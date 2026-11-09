"""Mem-Gallery task files: multi-session dialogues with images and OPEN questions.

Layout (HF ``Ethan-Bei/Mem-Gallery``): ``<root>/dialog/<Task>.json`` holds ``character_profile``,
``multi_session_dialogues`` (MemEye's round format, at most one image per round) and
``human-annotated QAs`` (``point`` is the question category, e.g. VS / AR / CD; questions have no
id). Images live under ``<root>/image/<Task>/``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..memeye.dataset import MemEyeTask, Question, _as_list, _image_path, _session


@dataclass(frozen=True)
class GalleryQuestion(Question):
    speaker: str = "user"
    """How the original answer prompt names the user: ``user (<character name>)``."""


def task_names(root: Path) -> list[str]:
    return sorted(path.stem for path in (root / "dialog").glob("*.json"))


def load_task(root: Path, name: str) -> MemEyeTask:
    data = json.loads((root / "dialog" / f"{name}.json").read_text(encoding="utf-8"))
    image_root = root / "image"
    person = (data.get("character_profile") or {}).get("name")
    speaker = f"user ({person})" if person else "user"
    sessions = tuple(_session(raw, image_root) for raw in data.get("multi_session_dialogues", []))
    questions = tuple(
        _question(raw, image_root, f"{name}#{index}", speaker)
        for index, raw in enumerate(data.get("human-annotated QAs", []))
    )
    return MemEyeTask(name=name, sessions=sessions, questions={"open": questions})


def _question(raw: dict[str, Any], image_root: Path, question_id: str, speaker: str) -> GalleryQuestion:
    return GalleryQuestion(
        question_id=question_id,
        question=str(raw.get("question", "")),
        answer=str(raw.get("answer", "")),
        images=tuple(_image_path(image_root, p) for p in _as_list(raw.get("question_image"))),
        point=raw.get("point"),
        clue=tuple(str(x) for x in _as_list(raw.get("clue"))),
        speaker=speaker,
    )
