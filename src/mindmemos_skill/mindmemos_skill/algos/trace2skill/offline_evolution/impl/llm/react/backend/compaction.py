"""Automatic message compaction for context window management."""

from __future__ import annotations

import json
from typing import Any

from mindmemos_skill.algos.trace2skill.offline_evolution.impl.llm.react.protocol import LLMRequest, estimate_token_count
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.llm.react.protocols import APIProtocol
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.llm.react.tools import SandboxedToolRuntime


def _stable_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _tool_result_entries(
    messages: list[dict[str, Any]],
) -> list[tuple[int, dict[str, Any], str]]:
    entries: list[tuple[int, dict[str, Any], str]] = []
    for index, message in enumerate(messages):
        if message.get("role") == "tool" and isinstance(message.get("content"), str):
            entries.append((index, message, "content"))
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result" and isinstance(block.get("content"), str):
                entries.append((index, block, "content"))
    return entries


def _tool_argument_entries(
    messages: list[dict[str, Any]],
) -> list[tuple[int, dict[str, Any], str]]:
    entries: list[tuple[int, dict[str, Any], str]] = []
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        calls = message.get("tool_calls")
        if isinstance(calls, list):
            for call in calls:
                function = call.get("function") if isinstance(call, dict) else None
                if isinstance(function, dict) and isinstance(function.get("arguments"), str):
                    entries.append((index, function, "arguments"))
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use" and isinstance(block.get("input"), dict):
                    entries.append((index, block, "input"))
    return entries


def _full_result_reference(content: str) -> str | None:
    for line in content.splitlines():
        if line.startswith("Full result: "):
            return line.removeprefix("Full result: ").strip() or None
    return None


def compact_tool_result(
    content: str,
    *,
    runtime: SandboxedToolRuntime | None,
    max_chars: int,
) -> str:
    if len(content) <= max_chars or content.startswith("[Tool result compacted automatically:"):
        return content
    reference = _full_result_reference(content)
    if reference is None and runtime is not None:
        reference = runtime.archive_result(content, label="context-compaction")
    prefix = f"[Tool result compacted automatically: original_chars={len(content)}]\n"
    if reference:
        prefix += f"Full result: {reference}\n"
    remaining = max(0, max_chars - len(prefix) - 32)
    head_size = remaining * 2 // 3
    tail_size = remaining - head_size
    body = content[:head_size]
    if tail_size:
        body += "\n... compacted ...\n" + content[-tail_size:]
    return prefix + body


def compact_tool_arguments(value: Any, *, max_chars: int) -> Any:
    if isinstance(value, str):
        if len(value) <= max_chars:
            return value
        return _stable_json(
            {
                "_compacted": True,
                "note": "arguments omitted after completed tool execution",
            }
        )
    if isinstance(value, dict):
        serialized = _stable_json(value)
        if len(serialized) <= max_chars:
            return value
        return {
            "_compacted": True,
            "note": "arguments omitted after completed tool execution",
        }
    return value


def auto_compact_messages(
    protocol: APIProtocol,
    request: LLMRequest,
    messages: list[dict[str, Any]],
    *,
    runtime: SandboxedToolRuntime | None,
    target_tokens: int,
    keep_recent_messages: int,
    compacted_result_chars: int,
) -> tuple[int, int, int]:
    """Shrink completed tool exchanges while preserving protocol pairing."""
    before = estimate_token_count(protocol.context_payload(request, messages))
    estimated = before
    cutoff = max(0, len(messages) - keep_recent_messages)
    result_entries = _tool_result_entries(messages)
    ordered_results = [
        *[entry for entry in result_entries if entry[0] < cutoff],
        *[entry for entry in result_entries if entry[0] >= cutoff],
    ]
    changed = 0
    for _, container, key in ordered_results:
        if estimated <= target_tokens:
            break
        original = container[key]
        compacted = compact_tool_result(
            original,
            runtime=runtime,
            max_chars=compacted_result_chars,
        )
        if compacted == original:
            continue
        container[key] = compacted
        changed += 1
        estimated = estimate_token_count(protocol.context_payload(request, messages))

    argument_entries = _tool_argument_entries(messages)
    ordered_arguments = [
        *[entry for entry in argument_entries if entry[0] < cutoff],
        *[entry for entry in argument_entries if entry[0] >= cutoff],
    ]
    for _, container, key in ordered_arguments:
        if estimated <= target_tokens:
            break
        original = container[key]
        compacted = compact_tool_arguments(
            original,
            max_chars=compacted_result_chars,
        )
        if compacted == original:
            continue
        container[key] = compacted
        changed += 1
        estimated = estimate_token_count(protocol.context_payload(request, messages))
    return changed, before, estimated


__all__ = ["auto_compact_messages"]
