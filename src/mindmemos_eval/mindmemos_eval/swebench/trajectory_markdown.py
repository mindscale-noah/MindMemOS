"""Tool-independent Markdown rendering of an audited chronological trajectory."""

import json
from pathlib import Path
from typing import Any

from .trajectory_view import _middle_truncate


def _text(value: Any) -> str:
    """Preserve strings and serialize structured values without tool-specific rules."""
    if value is None:
        return ""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)


def render_message_content(message: dict[str, Any]) -> str:
    """Render a normalized chat message without interpreting tool names or arguments.

    Args:
        message: A chat message with content, optional tool_calls or tool_call_id.

    Returns:
        Original body and ordered generic tool-call or tool-result text blocks.
    """
    body = _text(message.get("content"))
    if message["role"] == "tool":
        return f"[tool_result]\ntool_call_id: {message.get('tool_call_id', '')}\nresult:\n{body}"
    parts = [body] if body else []
    for call in message.get("tool_calls") or []:
        function = call["function"]
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                pass
        parts.append(
            f"[tool_call]\nid: {call.get('id', '')}\nname: {function['name']}\n"
            f"arguments:\n{_text(arguments)}"
        )
    return "\n\n".join(parts)


def render_parent_markdown(directory: Path, ordered_view: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """Re-render original messages using existing audited order and source indices.

    Args:
        directory: Directory holding complete parent and child trajectory JSON files.
        ordered_view: Chronological view with trajectory_id and source_message_index.

    Returns:
        Markdown input and per-message rendering provenance. Parent messages and
        each child's last round remain complete; earlier child content is capped
        at 600 characters after rendering. No tool-specific ordering is inferred.
    """
    sources = {}
    last_rounds = {}
    for turn in ordered_view["turns"]:
        identifier = turn["trajectory_id"]
        if identifier not in sources:
            source = json.loads((directory / f"{identifier}.json").read_text(encoding="utf-8"))
            sources[identifier] = source
            indices = [i for i, message in enumerate(source["messages"]) if message["role"] == "assistant"]
            last_rounds[identifier] = indices[-1] if indices else len(source["messages"]) - 1
    sections = [f"# 任务\n\n{ordered_view['task']}\n\n# 执行轨迹"]
    records = []
    for turn in ordered_view["turns"]:
        identifier = turn["trajectory_id"]
        index = turn["source_message_index"]
        source = sources[identifier]
        message = source["messages"][index]
        content = render_message_content(message)
        truncate = source["role"] == "child" and index < last_rounds[identifier]
        rendered = _middle_truncate(content) if truncate else content
        sections.append(
            f"## 消息 {turn['message_index']}\nagent: {turn['agent']}\nrole: {message['role']}\n"
            f"content:\n{rendered}"
        )
        records.append({
            "message_index": turn["message_index"], "trajectory_id": identifier,
            "source_message_index": index, "agent": turn["agent"], "role": message["role"],
            "original_chars": len(content), "rendered_chars": len(rendered),
            "truncated": rendered != content, "content": rendered,
        })
    return "\n\n".join(sections) + "\n", records
