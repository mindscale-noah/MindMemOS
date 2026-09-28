"""Chronological parent-memory views derived without modifying original traces."""

import json
from pathlib import Path
from typing import Any

from .typing import Trajectory

PARENT_VIEW_FORMAT = "chronological_parent_v1"
CHILD_INTERMEDIATE_CHAR_LIMIT = 600


def _middle_truncate(text: str) -> str:
    """Keep both ends within the character budget, including the omission marker."""
    limit = CHILD_INTERMEDIATE_CHAR_LIMIT
    if len(text) <= limit:
        return text
    marker = "\n...[middle truncated]...\n"
    head = (limit - len(marker)) // 2
    tail = limit - len(marker) - head
    return text[:head] + marker + text[-tail:]


def build_parent_memory_view(directory: Path) -> dict[str, Any]:
    """Interleave child invocations at their corresponding parent tool returns.

    Args:
        directory: Task directory containing parent.json and child-*.json.

    Returns:
        Task text and sequentially indexed turns. Parent messages and the last
        child assistant round are complete. Earlier child messages have at most
        600 characters each, including the truncation marker.

    Raises:
        ValueError: Saved parent/child linkage is missing or duplicated.
    """
    parent = Trajectory.model_validate_json((directory / "parent.json").read_text(encoding="utf-8"))
    children = {}
    for path in sorted(directory.glob("child-*.json")):
        child = Trajectory.model_validate_json(path.read_text(encoding="utf-8"))
        if child.trajectory_id in children:
            raise ValueError(f"Duplicate child trajectory: {child.trajectory_id}")
        children[child.trajectory_id] = child
    turns = []
    emitted = set()
    tool_names = {}

    def append(trace: Trajectory, index: int, *, truncate: bool, agent: str) -> None:
        message = trace.messages[index]
        original = json.dumps(message, ensure_ascii=False)
        text = _middle_truncate(original) if truncate else original
        turns.append({
            "message_index": len(turns), "agent": agent, "role": message["role"],
            "text": text, "timestamp": None, "trajectory_id": trace.trajectory_id,
            "source_message_index": index, "truncated": text != original,
            "original_chars": len(original),
        })

    for index, message in enumerate(parent.messages):
        for call in message.get("tool_calls", []):
            tool_names[call["id"]] = call["function"]["name"]
        agent = "main"
        if message["role"] == "tool" and tool_names.get(message.get("tool_call_id")) == "delegate":
            response = json.loads(message["content"])
            identifier = response.get("trajectory_id")
            if identifier:
                if identifier not in children or identifier in emitted:
                    raise ValueError(f"Missing or duplicated child return: {identifier}")
                child = children[identifier]
                assistant_indices = [i for i, item in enumerate(child.messages) if item["role"] == "assistant"]
                last_round = assistant_indices[-1] if assistant_indices else len(child.messages) - 1
                for child_index in range(len(child.messages)):
                    append(child, child_index, truncate=child_index < last_round, agent=child.agent)
                emitted.add(identifier)
                agent = child.agent
        append(parent, index, truncate=False, agent=agent)
    if emitted != set(children):
        raise ValueError("Some child trajectories have no matching parent delegation return")
    return {"task": parent.task, "turns": turns}


def parent_view_messages(view: dict[str, Any]) -> list[dict[str, str]]:
    """Adapt the exact review text to the SDK dialogue contract.

    Args:
        view: Result of build_parent_memory_view.

    Returns:
        Messages in the same order, retaining role, agent identity, and text.
    """
    return [{"role": turn["role"], "agent": turn["agent"], "content": turn["text"]}
            for turn in view["turns"]]
