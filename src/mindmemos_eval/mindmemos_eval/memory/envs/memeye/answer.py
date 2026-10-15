"""One answerer for every memory algorithm: retrieved text and images go to the same VLM.

The answerer sees exactly what ``memory.search`` returned: text memories as a numbered list with
their times, and image results (multimodal search) attached as images labelled with the dialogue they
came from. Question images are attached for every algorithm. Nothing else (dataset captions,
gold clues) is shown.
"""

from __future__ import annotations

import asyncio
import base64
import io
import mimetypes
from pathlib import Path
from typing import Any

MCQ_SYSTEM = """You answer questions about a user's past conversations with an assistant, using only the MEMORIES retrieved for the question.
MEMORIES are short notes from those conversations, each with the time it happened. Some memories are images shared in the conversations; they are attached and labelled with their memory number and the conversation text around them.

Rules:
- Use only the memories and the attached images; do not use outside knowledge.
- For visual details (colors, text, layout, objects, what is shown), look at the attached images.
- If memories conflict, prefer the most recent one unless the question asks about an earlier state.
- Retrieval may be incomplete: when no option is fully supported, choose the one most consistent with the memories. If the memories show something was never mentioned or shown, choose the option that says so.
- Output only the letter of one option. No words, no punctuation."""

OPEN_SYSTEM = """You answer questions about a user's past conversations with an assistant, using only the MEMORIES retrieved for the question.
MEMORIES are short notes from those conversations, each with the time it happened. Some memories are images shared in the conversations; they are attached and labelled with their memory number and the conversation text around them.

Rules:
- Use only the memories and the attached images; do not use outside knowledge.
- For visual details (colors, text, layout, objects, what is shown), look at the attached images.
- If memories conflict, prefer the most recent one unless the question asks about an earlier state.
- Give the shortest answer that fully answers the question: a name, number, color, date or short phrase, never a full sentence (e.g. "White background", not "The background is white."). If it asks for a number, output only the number.
- Never refuse: if the memories are insufficient, give your best concrete guess. No explanation."""


class ImageEncoder:
    """Loads images (from the memory API or the dataset), downscales them once and caches data URIs."""

    def __init__(self, memory: Any, *, max_side: int = 1024) -> None:
        self._memory = memory
        self._max_side = max_side
        self._cache: dict[str, str] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def from_blob(self, blob_id: str) -> str:
        return await self._cached(f"blob:{blob_id}", lambda: self._memory.get_image(blob_id))

    async def from_path(self, path: Path) -> str:
        async def read() -> tuple[bytes, str]:
            return await asyncio.to_thread(path.read_bytes), mimetypes.guess_type(path.name)[0] or "image/png"

        return await self._cached(f"path:{path}", read)

    async def _cached(self, key: str, load) -> str:
        if key in self._cache:
            return self._cache[key]
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            if key not in self._cache:
                data, mime = await load()
                self._cache[key] = await asyncio.to_thread(self._data_uri, data, mime)
        return self._cache[key]

    def _data_uri(self, data: bytes, mime: str) -> str:
        try:
            from PIL import Image

            image = Image.open(io.BytesIO(data))
            if max(image.size) > self._max_side or image.mode not in ("RGB", "L"):
                image = image.convert("RGB")
                image.thumbnail((self._max_side, self._max_side))
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=90)
                data, mime = buffer.getvalue(), "image/jpeg"
        except Exception:  # noqa: BLE001 - Pillow missing or image unreadable: send the original bytes
            pass
        return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def format_memory(index: int, item: Any) -> str:
    when = item.event_time or item.source_timestamp or item.last_update_at or "unknown time"
    return f"[{index}] ({when}) {item.memory}"


async def build_messages(
    *,
    mode: str,
    question_block: str,
    memories: list[Any],
    question_images: list[Path],
    encoder: ImageEncoder,
    max_images: int,
) -> tuple[list[dict[str, Any]], int]:
    """Chat messages for one question; returns them with the number of attached images."""

    lines, image_parts = [], []
    budget = max(0, max_images - len(question_images))
    for index, item in enumerate(memories, start=1):
        lines.append(format_memory(index, item))
        blob_id = (item.metadata or {}).get("blob_id") if item.memory_type == "image" else None
        if blob_id and budget > 0:
            budget -= 1
            image_parts.append({"type": "text", "text": f"Image of memory [{index}]: {item.memory}"})
            image_parts.append({"type": "image_url", "image_url": {"url": await encoder.from_blob(blob_id)}})
    content: list[dict[str, Any]] = [
        {"type": "text", "text": "MEMORIES:\n" + ("\n".join(lines) if lines else "(no memories retrieved)")},
        *image_parts,
    ]
    for path in question_images[:max_images]:
        content.append({"type": "text", "text": "Image attached to the question:"})
        content.append({"type": "image_url", "image_url": {"url": await encoder.from_path(path)}})
    content.append({"type": "text", "text": f"QUESTION:\n{question_block}"})
    system = MCQ_SYSTEM if mode == "mcq" else OPEN_SYSTEM
    attached = sum(1 for part in content if part["type"] == "image_url")
    return [{"role": "system", "content": system}, {"role": "user", "content": content}], attached


def options_block(question: str, rotation: dict[str, str]) -> tuple[str, set[str]]:
    letters = sorted(key for key in rotation if key != "answer")
    lines = [question, "", *(f"{letter}. {rotation[letter]}" for letter in letters), ""]
    lines.append(f"Answer with only the option letter ({', '.join(letters)}).")
    return "\n".join(lines), set(letters)
