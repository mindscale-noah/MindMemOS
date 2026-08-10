"""OpenClaw CLI JSON and native session-transcript conversion helpers."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


def load_cli_result(stdout: str) -> dict[str, Any] | None:
    """Parse OpenClaw JSON output while tolerating diagnostic prefix lines."""

    try:
        value = json.loads(stdout)
    except json.JSONDecodeError:
        value = None
    if isinstance(value, dict):
        return value
    for line in reversed(stdout.splitlines()):
        try:
            value = json.loads(line.strip())
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def extract_final_text(result: Mapping[str, Any]) -> str:
    """Best-effort final response extraction across OpenClaw JSON variants."""

    for key in ("payloads", "result", "reply", "response", "final", "final_text", "text", "content", "message"):
        text = _string_from_value(result.get(key))
        if text:
            return text
    return ""


def extract_session_file(result: Mapping[str, Any]) -> Path | None:
    """Find the native JSONL transcript path nested in CLI metadata."""

    value = _find_key(result, "sessionFile")
    return Path(value).expanduser() if isinstance(value, str) and value else None


def extract_session_id(result: Mapping[str, Any]) -> str | None:
    value = _find_key(result, "sessionId")
    return value if isinstance(value, str) and value else None


def extract_transport(result: Mapping[str, Any]) -> str | None:
    meta = result.get("meta")
    value = meta.get("transport") if isinstance(meta, Mapping) else None
    return value if isinstance(value, str) and value else None


def extract_error(result: Mapping[str, Any]) -> str | None:
    """Return a structured OpenClaw failure even when the CLI exits zero."""

    for candidate in (result, result.get("result")):
        if not isinstance(candidate, Mapping):
            continue
        status = candidate.get("status")
        if isinstance(status, str) and status.lower() in {"error", "failed", "failure"}:
            detail = candidate.get("error") or candidate.get("errorMessage") or candidate.get("message")
            return _string_from_value(detail) or f"OpenClaw returned status={status!r}"
    return None


def read_session_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return events
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            events.append(value)
    return events


def convert_session_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert native OpenClaw messages to the canonical add-compatible shape."""

    messages: list[dict[str, Any]] = []
    for event in events:
        if event.get("type") != "message":
            continue
        native = event.get("message")
        if not isinstance(native, Mapping):
            continue
        role = native.get("role")
        if role == "assistant":
            messages.extend(_assistant_messages(native.get("content")))
        elif role == "toolResult":
            messages.append(
                {
                    "role": "tool",
                    "name": str(native.get("toolName") or ""),
                    "tool_call_id": str(native.get("toolCallId") or ""),
                    "content": _string_from_value(native.get("content")),
                    "is_error": native.get("isError") is True,
                }
            )
        elif role in {"user", "system"}:
            messages.append({"role": role, "content": _string_from_value(native.get("content"))})
    return messages


def count_assistant_turns(events: list[dict[str, Any]]) -> int:
    return sum(
        event.get("type") == "message"
        and isinstance(event.get("message"), Mapping)
        and event["message"].get("role") == "assistant"
        for event in events
    )


def _assistant_messages(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"role": "assistant", "content": content}] if content else []
    if not isinstance(content, list):
        return []
    messages: list[dict[str, Any]] = []
    reasoning: list[str] = []
    for block in content:
        if not isinstance(block, Mapping):
            continue
        block_type = block.get("type")
        if block_type == "thinking":
            text = _string_from_value(block.get("thinking") or block.get("text"))
            if text:
                reasoning.append(text)
        elif block_type == "text":
            text = _string_from_value(block.get("text"))
            if text:
                message: dict[str, Any] = {"role": "assistant", "content": text}
                if reasoning:
                    message["reasoning_content"] = "\n".join(reasoning)
                    reasoning.clear()
                messages.append(message)
        elif block_type == "toolCall":
            name = str(block.get("name") or "")
            arguments = block.get("arguments") or block.get("input") or {}
            if not isinstance(arguments, Mapping):
                arguments = {}
            message = {
                "role": "assistant",
                "content": f"[tool_call] {name}({json.dumps(dict(arguments), ensure_ascii=False, separators=(',', ':'))})",
                "tool_call_id": str(block.get("id") or ""),
            }
            if reasoning:
                message["reasoning_content"] = "\n".join(reasoning)
                reasoning.clear()
            messages.append(message)
    return messages


def _find_key(value: Any, key: str) -> Any:
    if isinstance(value, Mapping):
        if key in value:
            return value[key]
        for nested in value.values():
            found = _find_key(nested, key)
            if found is not None:
                return found
    elif isinstance(value, list):
        for nested in value:
            found = _find_key(nested, key)
            if found is not None:
                return found
    return None


def _string_from_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        for key in ("text", "content", "message", "output"):
            text = _string_from_value(value.get(key))
            if text:
                return text
        return ""
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        parts = [_string_from_value(item) for item in value]
        return "\n".join(part for part in parts if part)
    return ""


__all__ = [
    "convert_session_events",
    "count_assistant_turns",
    "extract_error",
    "extract_final_text",
    "extract_session_file",
    "extract_session_id",
    "extract_transport",
    "load_cli_result",
    "read_session_events",
]
