"""Shared Markdown trajectory formatting for forest reflection."""

import json
from collections.abc import Callable

from pydantic import JsonValue

from ....envs.typing import RolloutResult


def format_conversation(events: list[dict[str, JsonValue]]) -> str:
    """Render recorded events without truncating evidence.

    Args:
        events: Ordered messages, action steps, or paired tool calls.

    Returns:
        Analyst-readable conversation in the reference project's format.
    """
    lines = []
    for event in events:
        if event.get("type") == "tool_call":
            lines.extend([f"[action] {event.get('cmd') or ''}", f"[obs]    {event.get('obs') or ''}"])
        elif "action" in event and "env_feedback" in event:
            step = event.get("step", "?")
            if event.get("reasoning"):
                lines.append(f"[step {step} think] {event['reasoning']}")
            lines.extend([f"[step {step} action] {event['action']}", f"[step {step} obs]    {event['env_feedback']}"])
        else:
            role = "verification" if event.get("role") == "system" else event.get("role", "agent")
            lines.append(f"[{role}] {event.get('content') or ''}")
    return "\n".join(lines)


def render_record(
    index: int,
    task_id: str,
    instruction: str,
    task_type: str,
    failure: str,
    steps: int,
    events: list[dict[str, JsonValue]],
    *,
    system: str = "",
    user: str = "",
) -> str:
    """Render one reference-format trajectory with recorded target context.

    Args:
        index: One-based position in the minibatch.
        task_id: Dataset identity.
        instruction: Task description used in the reference header.
        task_type: Released task category, or empty when absent.
        failure: Evaluator or execution failure reason.
        steps: Environment-specific reference turn count.
        events: Ordered analyst conversation.
        system: Actual initial system prompt, when recorded.
        user: Actual initial user prompt, when recorded.

    Returns:
        Complete trajectory section.
    """
    header = f"### Trajectory {index} (id={task_id})\nTask: {instruction}\nTask type: {task_type}\n"
    if failure:
        header += f"Failure reason: {failure}\n"
    header += f"Steps: {steps}\n"
    if system:
        header += f"\n#### Target System Prompt\n{system}\n"
    if user:
        header += f"\n#### Target User Prompt\n{user}\n"
    return header + "\n" + format_conversation(events)


def recorded_context(result: RolloutResult) -> tuple[str, str]:
    """Read actual initial inputs rather than reconstructing current prompts.

    Args:
        result: Completed rollout containing recorded agent inputs.

    Returns:
        Initial system and user messages.

    Raises:
        ValueError: Required input context or assistant output is absent.
    """
    events = result.trajectory
    if (
        len(events) < 3
        or events[0].get("role") != "system"
        or events[1].get("role") != "user"
        or not any(event.get("role") == "assistant" for event in events)
    ):
        raise ValueError("Reflection requires recorded system, user and assistant messages")
    return events[0]["content"], events[1]["content"]


def tool_conversation(
    events: list[dict[str, JsonValue]],
    *,
    assistant_role: str = "assistant",
    command_formatter: Callable[[str, str, str], str] | None = None,
    strict: bool = True,
) -> list[dict[str, JsonValue]]:
    """Pair native calls with observations while preserving call order and arguments.

    Args:
        events: Recorded conversation after the initial system prompt.
        assistant_role: Reference role label for model messages.
        command_formatter: Optional environment-specific command display using name, JSON arguments and observation.
        strict: Reject incomplete native calls; disable for partial generic evidence.

    Returns:
        Conversation with native tool exchanges represented as action/observation pairs.

    Raises:
        ValueError: A native tool observation cannot be matched to its call.
    """
    conversation, pending = [], {}
    functions = {}
    for event in events:
        if event.get("role") == "assistant":
            conversation.append({"role": assistant_role, "content": event.get("content", "")})
            for call in event.get("tool_calls", []):
                function = call["function"]
                item = {
                    "type": "tool_call",
                    "cmd": format_tool_command(function["name"], function["arguments"]),
                    "obs": "",
                }
                identity = call.get("id")
                if identity is None and not strict:
                    conversation.append(item)
                    continue
                if identity in pending:
                    raise ValueError("Duplicate native tool call identity")
                functions[identity] = function
                pending[identity] = item
                conversation.append(item)
        elif event.get("role") == "tool":
            identity = event.get("tool_call_id")
            if identity not in pending:
                if not strict:
                    conversation.append(event)
                    continue
                raise ValueError("Unmatched native tool observation")
            item = pending.pop(identity)
            item["obs"] = event.get("content", "")
            function = functions[identity]
            formatter = command_formatter or format_tool_command
            item["cmd"] = formatter(function["name"], function["arguments"], item["obs"])
        else:
            conversation.append(event)
    if pending and strict:
        raise ValueError("Missing native tool observation")
    return conversation


def format_tool_command(name: str, arguments: str, observation: str = "") -> str:
    """Format arbitrary tool arguments in the document-tool display style.

    Args:
        name: Recorded tool name.
        arguments: Raw JSON arguments, retained verbatim if malformed.
        observation: Recorded response, unused by the generic formatter.

    Returns:
        Readable call with argument names and values.
    """
    try:
        values = json.loads(arguments or "{}")
    except (ValueError, TypeError):
        return f"{name}({arguments})"
    if not isinstance(values, dict):
        return f"{name}({arguments})"
    return f"{name}({', '.join(f'{key}={value!r}' for key, value in values.items())})"
