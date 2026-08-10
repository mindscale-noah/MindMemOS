"""Claude-only message conversion, Skill injection, and usage binding helpers."""

from __future__ import annotations

import dataclasses
import json
from typing import Any

from ...errors import SkillCapabilityUnavailableError


def _collect_text(blocks: list[dict[str, Any]]) -> str:
    return "\n".join(
        block.get("text", "")
        for block in blocks
        if isinstance(block, dict) and block.get("type") == "text"
    )


def _collect_reasoning(blocks: list[dict[str, Any]]) -> str | None:
    parts = [
        block.get("thinking", "")
        for block in blocks
        if isinstance(block, dict) and block.get("type") == "thinking"
    ]
    return "\n".join(parts) if parts else None


def _collect_tool_calls(blocks: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    calls: list[dict[str, Any]] = []
    for block in blocks:
        if not isinstance(block, dict) or block.get("type") != "tool_use":
            continue
        calls.append(
            {
                "id": block.get("id", ""),
                "type": "function",
                "function": {
                    "name": block.get("name", ""),
                    "arguments": json.dumps(block.get("input", {}), ensure_ascii=False),
                },
            }
        )
    return calls or None


def convert_assistant_blocks(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    """Convert Claude assistant blocks into one OpenAI-format trajectory message."""

    message: dict[str, Any] = {"role": "assistant", "content": _collect_text(blocks) or ""}
    reasoning = _collect_reasoning(blocks)
    if reasoning:
        message["reasoning_content"] = reasoning
    tool_calls = _collect_tool_calls(blocks)
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


def convert_user_blocks(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Claude user/tool-result blocks into OpenAI-format messages."""

    messages: list[dict[str, Any]] = []
    text_parts: list[str] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "tool_result":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": block.get("tool_use_id", ""),
                    "content": str(block.get("content", "")),
                }
            )
        elif block.get("type") == "text":
            text_parts.append(block.get("text", ""))
    if text_parts:
        messages.insert(0, {"role": "user", "content": "\n".join(text_parts)})
    return messages


def parse_cli_events(stdout: str) -> list[dict[str, Any]]:
    """Parse Claude CLI stream-json output, ignoring non-JSON lines."""

    events: list[dict[str, Any]] = []
    for raw_line in (stdout or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            events.append(item)
    return events


def extract_cli_session_id(events: list[dict[str, Any]]) -> str | None:
    for event in events:
        session_id = event.get("session_id")
        if isinstance(session_id, str) and session_id.strip():
            return session_id.strip()
    return None


def extract_cli_trajectory_messages(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Claude CLI stream events into canonical trajectory messages."""

    messages: list[dict[str, Any]] = []
    for event in events:
        event_type = event.get("type")
        if event_type == "assistant":
            blocks = event.get("message", {}).get("content", [])
            if isinstance(blocks, list):
                messages.append(convert_assistant_blocks(blocks))
        elif event_type == "user":
            blocks = event.get("message", {}).get("content", [])
            if isinstance(blocks, list):
                messages.extend(convert_user_blocks(blocks))
    return messages


def extract_cli_num_turns(events: list[dict[str, Any]]) -> int:
    for event in reversed(events):
        if event.get("type") == "result":
            turns = event.get("num_turns")
            if isinstance(turns, int) and turns > 0:
                return turns
    return 1


def load_claude_agent_sdk() -> tuple[Any, Any, type[Any], type[Any], type[Any]]:
    """Load the optional Claude Agent SDK dependency on first execution."""

    try:
        from claude_agent_sdk import ClaudeAgentOptions, query
        from claude_agent_sdk.types import AssistantMessage, ResultMessage, UserMessage
    except ModuleNotFoundError as exc:
        missing_module = exc.name or "claude_agent_sdk"
        raise SkillCapabilityUnavailableError(
            "Claude Agent SDK capability is unavailable because the optional dependency "
            f"is not installed correctly (missing module: {missing_module!r}). "
            "Install it with `pip install 'mindmemos-skill[claude-sdk]'`."
        ) from exc
    return ClaudeAgentOptions, query, AssistantMessage, ResultMessage, UserMessage


_SDK_BLOCK_TYPE_MAP = {
    "TextBlock": "text",
    "ThinkingBlock": "thinking",
    "ToolUseBlock": "tool_use",
    "ToolResultBlock": "tool_result",
}


def _sdk_block_to_dict(block: Any) -> dict[str, Any]:
    if dataclasses.is_dataclass(block):
        converted = dataclasses.asdict(block)
        converted.setdefault("type", _SDK_BLOCK_TYPE_MAP.get(type(block).__name__, type(block).__name__))
        return converted
    return {} if not isinstance(block, dict) else block


def convert_sdk_message(
    message: Any,
    *,
    assistant_message_type: type[Any],
    user_message_type: type[Any],
) -> dict[str, Any] | list[dict[str, Any]]:
    """Convert a Claude SDK message into canonical trajectory messages."""

    blocks = [_sdk_block_to_dict(block) for block in (message.content or [])]
    if isinstance(message, assistant_message_type):
        return convert_assistant_blocks(blocks)
    if isinstance(message, user_message_type):
        return convert_user_blocks(blocks)
    return {}


def extract_used_skill_names(messages: list[dict[str, Any]]) -> set[str]:
    """Interpret Claude's native ``Skill`` tool-use messages."""

    used: set[str] = set()
    for message in messages:
        for tool_call in message.get("tool_calls") or []:
            function = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
            if not isinstance(function, dict) or function.get("name") != "Skill":
                continue
            try:
                arguments = json.loads(function.get("arguments", "{}"))
            except (json.JSONDecodeError, TypeError):
                continue
            name = arguments.get("skill")
            if isinstance(name, str) and name:
                used.add(name)
    return used


__all__ = [
    "convert_sdk_message",
    "extract_used_skill_names",
    "extract_cli_num_turns",
    "extract_cli_session_id",
    "extract_cli_trajectory_messages",
    "load_claude_agent_sdk",
    "parse_cli_events",
]
