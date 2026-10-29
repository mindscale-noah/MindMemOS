"""Mem-Gallery answer protocol on the shared answerer layout (``memeye.answer.build_messages``).

The prompts are the benchmark's own (``prompts/`` = Mem-Gallery ``benchmark/prompt``): its system
prompt, its question template naming the user by the character, and the per-category format
constraints (AR: "Not mentioned.", CD: "Yes."/"No.", VS: reply with image ids). Mem-Gallery asks
for dataset image ids (``D1:IMG_002``), so every returned memory that points at an image the search
also returned is shown with that image's ids, as the benchmark labels its memory images
``image_id: ...``. The ids are the ones the client sent with the image (``caller_image_ids``);
nothing else from the dataset (captions, clues) is shown.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..memeye.answer import format_memory

_PROMPTS = Path(__file__).with_name("prompts")
SYSTEM = (_PROMPTS / "sys_prompt.txt").read_text(encoding="utf-8").strip() + (
    "\n\nThe memory contents are short notes from the conversation, each with the time it happened. "
    "Some of them are images from the conversation; they are attached and labelled with their memory "
    "number and image_id."
)
CONSTRAINTS = {
    point: (_PROMPTS / f"{point.lower()}_prompt.txt").read_text(encoding="utf-8").strip()
    for point in ("AR", "CD", "VS")
}
QUESTION = """Your task is to answer the question about the conversation between {speaker} and assistant in a concise manner with the help of memory content.
Please only provide the content of the answer, without including introductory phrases like 'answer:'.
For questions that require answering a date or time, strictly follow the format and provide a specific date or time whenever possible.
Generate answers primarily concise, yet complete enough to accurately answer the questions.

The current question is as follows:
{question} {constraint}"""
MAX_TOKENS = 256


def question_block(question: str, point: Any, speaker: str) -> str:
    constraint = CONSTRAINTS.get(str(point or "").strip().upper())
    return QUESTION.format(speaker=speaker, question=question, constraint=f"\n\n{constraint}" if constraint else "")


def image_ids(memories: list[Any]) -> dict[str, list[str]]:
    """Blob id -> the caller's image ids, from the image results of one search."""

    ids: dict[str, list[str]] = {}
    for item in memories:
        metadata = item.metadata or {}
        if item.memory_type == "image" and metadata.get("blob_id"):
            known = ids.setdefault(metadata["blob_id"], [])
            known.extend(i for i in metadata.get("caller_image_ids") or [] if i not in known)
    return ids


def item_image_ids(item: Any, ids: dict[str, list[str]]) -> list[str]:
    """Image ids of the images a result points at: its own blob, or the images a fact is anchored to."""

    metadata = item.metadata or {}
    blobs = [metadata["blob_id"]] if metadata.get("blob_id") else list(metadata.get("image_blob_ids") or [])
    out: list[str] = []
    for blob in blobs:
        out.extend(i for i in ids.get(blob, []) if i not in out)
    return out


def answer_options(memories: list[Any]) -> dict[str, Any]:
    """``build_messages`` overrides: Mem-Gallery system prompt and image-id labels."""

    ids = image_ids(memories)

    def suffix(item: Any) -> str:
        found = item_image_ids(item, ids)
        return f" (image_id: {', '.join(found)})" if found else ""

    return {
        "system": SYSTEM,
        "line": lambda index, item: format_memory(index, item) + suffix(item),
        "image_label": lambda index, item: f"Image of memory [{index}]{suffix(item)}: {item.memory}",
        "max_tokens": MAX_TOKENS,
    }
