"""Lossless STATE-Bench trajectory-to-memory message mapping.

STATE-Bench messages may contain fields such as ``tool_calls`` that are not
part of MindMemOS's public ``DialogueMessage`` schema. The adapter therefore
keeps normal text readable and appends a canonical JSON copy whenever a
message carries anything beyond a plain role/content pair.
"""

from __future__ import annotations

import json
from typing import Any

TRAJECTORY_MAPPING_VERSION = "statebench-lossless-v1"
_RAW_MARKER = "[STATE-BENCH-RAW-MESSAGE]"


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _lossless_content(message: dict[str, Any]) -> str:
    role = message.get("role")
    content = message.get("content")
    is_plain_text = (
        "role" in message
        and isinstance(role, str)
        and isinstance(content, str)
        and bool(content.strip())
        and set(message).issubset({"role", "content"})
    )
    if is_plain_text:
        return content
    raw = _json(message)
    if isinstance(content, str) and content.strip():
        return f"{content}\n\n{_RAW_MARKER}\n{raw}"
    return f"{_RAW_MARKER}\n{raw}"


def to_add_messages(conversation: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Convert STATE-Bench messages without dropping structured fields."""

    messages: list[dict[str, str]] = []
    for message in conversation:
        if not isinstance(message, dict):
            messages.append({"role": "assistant", "content": f"{_RAW_MARKER}\n{_json(message)}"})
            continue
        role = str(message.get("role") or "assistant")
        messages.append({"role": role, "content": _lossless_content(message)})
    return messages


def mapping_stats(
    conversation: list[Any],
    messages: list[dict[str, str]],
) -> dict[str, int | float]:
    """Return observable size metrics for a STATE-Bench add mapping."""

    raw_content_chars = 0
    structured_message_count = 0
    for item in conversation:
        if not isinstance(item, dict):
            structured_message_count += 1
            raw_content_chars += len(_json(item))
            continue
        content = item.get("content")
        if isinstance(content, str):
            raw_content_chars += len(content)
        elif content is not None:
            raw_content_chars += len(_json(content))
        if not (
            isinstance(item.get("role"), str)
            and isinstance(content, str)
            and bool(content.strip())
            and set(item).issubset({"role", "content"})
        ):
            structured_message_count += 1

    mapped_content_chars = sum(len(item.get("content") or "") for item in messages)
    expansion_ratio = mapped_content_chars / raw_content_chars if raw_content_chars else 1.0
    return {
        "raw_message_count": len(conversation),
        "mapped_message_count": len(messages),
        "structured_message_count": structured_message_count,
        "raw_content_chars": raw_content_chars,
        "mapped_content_chars": mapped_content_chars,
        "content_expansion_chars": mapped_content_chars - raw_content_chars,
        "content_expansion_ratio": expansion_ratio,
    }
