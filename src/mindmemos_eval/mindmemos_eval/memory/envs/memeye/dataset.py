"""MemEye task files: dialogue sessions with images, and MCQ / OPEN questions.

Layout (HF ``MemEyeBench/MemEye``): ``<root>/dialog/<Task>.json`` (MCQ, options given as four
rotations of the same choices) and ``<root>/dialog/<Task>_Open.json`` (OPEN), images under
``<root>/image/``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MEMEYE_TASKS = (
    "Brand_Memory_Test",
    "Card_Playlog_Test",
    "Cartoon_Entertainment_Companion",
    "Home_Renovation_Interior_Design",
    "Multi-Scene_Visual_Case_Archive_Assistant",
    "Outdoor_Navigation_Route_Memory_Assistant",
    "Personal_Health_Dashboard_Assistant",
    "Social_Chat_Memory_Test",
)
MODES = ("mcq", "open")


@dataclass(frozen=True)
class Round:
    round_id: str
    user: str
    assistant: str
    images: tuple[Path, ...] = ()
    image_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class Session:
    session_id: str
    date: str
    rounds: tuple[Round, ...]

    def timestamp_ms(self, round_index: int) -> int:
        """Session date at midnight UTC, one minute per round to keep the order."""

        day = (
            datetime.fromisoformat(self.date[:10]).replace(tzinfo=UTC)
            if self.date
            else datetime(2024, 1, 1, tzinfo=UTC)
        )
        return int(day.timestamp() * 1000) + round_index * 60_000


@dataclass(frozen=True)
class Question:
    question_id: str
    question: str
    answer: str
    images: tuple[Path, ...] = ()
    rotations: tuple[dict[str, str], ...] = ()
    """MCQ only: each rotation maps letters to options plus ``answer`` (the correct letter)."""
    point: Any = None
    clue: tuple[str, ...] = ()


@dataclass(frozen=True)
class MemEyeTask:
    name: str
    sessions: tuple[Session, ...]
    questions: dict[str, tuple[Question, ...]] = field(default_factory=dict)

    @property
    def rounds(self) -> int:
        return sum(len(session.rounds) for session in self.sessions)


def load_task(root: Path, name: str, modes: tuple[str, ...] = MODES) -> MemEyeTask:
    """Load a task; sessions come from the MCQ file (the OPEN file shares the dialogues)."""

    image_root = root / "image"
    mcq = _read(root / "dialog" / f"{name}.json")
    sessions = tuple(_session(raw, image_root) for raw in mcq.get("multi_session_dialogues", []))
    questions: dict[str, tuple[Question, ...]] = {}
    for mode in modes:
        data = mcq if mode == "mcq" else _read(root / "dialog" / f"{name}_Open.json")
        questions[mode] = tuple(_question(raw, image_root, mode) for raw in data.get("human-annotated QAs", []))
    return MemEyeTask(name=name, sessions=sessions, questions=questions)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _image_path(image_root: Path, raw: str) -> Path:
    relative = str(raw)
    for prefix in ("../image/", "image/", "./"):
        if relative.startswith(prefix):
            relative = relative[len(prefix) :]
    return image_root / relative


def _as_list(value: Any) -> list[Any]:
    if value is None or value == "":
        return []
    return list(value) if isinstance(value, (list, tuple)) else [value]


def _session(raw: dict[str, Any], image_root: Path) -> Session:
    rounds = tuple(
        Round(
            round_id=str(item.get("round", "")),
            user=str(item.get("user") or ""),
            assistant=str(item.get("assistant") or ""),
            images=tuple(_image_path(image_root, p) for p in _as_list(item.get("input_image"))),
            image_ids=tuple(str(x) for x in _as_list(item.get("image_id"))),
        )
        for item in raw.get("dialogues", []) or []
    )
    return Session(session_id=str(raw.get("session_id", "")), date=str(raw.get("date") or ""), rounds=rounds)


def _question(raw: dict[str, Any], image_root: Path, mode: str) -> Question:
    rotations: tuple[dict[str, str], ...] = ()
    options = raw.get("options")
    if mode == "mcq" and isinstance(options, list):
        rotations = tuple({str(k): str(v) for k, v in rotation.items()} for rotation in options)
    elif mode == "mcq" and isinstance(options, dict):
        rotations = ({**{str(k): str(v) for k, v in options.items()}, "answer": str(raw.get("answer", ""))},)
    return Question(
        question_id=str(raw.get("question_id", "")),
        question=str(raw.get("question", "")),
        answer=str(raw.get("answer", "")),
        images=tuple(_image_path(image_root, p) for p in _as_list(raw.get("question_image"))),
        rotations=rotations,
        point=raw.get("point"),
        clue=tuple(str(x) for x in _as_list(raw.get("clue"))),
    )
