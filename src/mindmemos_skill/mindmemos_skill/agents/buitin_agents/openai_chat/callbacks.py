"""Callbacks for native tool calls and textual Action/Observation turns."""

from pydantic import JsonValue

from ....common.json import extract_json_object
from ...typing import AgentResult, ToolCall, ToolTurn
from .config import OpenAIChatConfig

FORMAT_ERROR = """Failed to parse your action. Please use the correct format.

To execute a tool, use this EXACT format:

Action:
{
    "name": "<tool_name>",
    "arguments": {"command": "<command_here>"}
}

To complete the task, output exactly:

ACTION: TASK_COMPLETE

Please try again with the correct format."""
"""Released policy feedback, retained verbatim."""


def parse_action(response: str) -> dict | None:
    """Parse an action or return None for the released completion behavior.

    Args:
        response: Complete policy response.

    Returns:
        Tool name and arguments, or None when completion was signaled.

    Raises:
        ValueError: An Action marker exists but its payload is malformed.
    """
    if "ACTION: TASK_COMPLETE" in response or "Action:" not in response:
        return None
    raw = response.split("Action:", 1)[1]
    try:
        payload = extract_json_object(raw, allow_incomplete=True)
    except ValueError as exc:
        raise ValueError("Malformed Action payload") from exc
    if not isinstance(payload.get("name"), str) or not isinstance(payload.get("arguments", {}), dict):
        raise ValueError("Malformed Action payload")
    return {"name": payload["name"], "arguments": payload.get("arguments", {})}


def text_format_error(error: ValueError) -> str:
    """Return the textual protocol correction prompt.

    Args:
        error: Parsing failure that triggered feedback.

    Returns:
        The exact format instructions for another attempt.
    """
    return FORMAT_ERROR


def parse_text_action(result: AgentResult, config: OpenAIChatConfig) -> ToolTurn:
    """Decode one textual action.

    Args:
        result: Raw model response with its original trace.
        config: Model settings shared by parser callbacks.

    Returns:
        Parsed tool calls or a completion turn.

    Raises:
        ValueError: The action payload is malformed.
        TypeError: The model output is not text.
    """
    if not isinstance(result.output, str):
        raise TypeError("Text actions require text model output")
    action = parse_action(result.output)
    return ToolTurn(text=result.output, calls=[ToolCall(**action)] if action is not None else [])


def parse_native_tools(result: AgentResult, config: OpenAIChatConfig) -> ToolTurn:
    """Decode native calls while preserving per-call argument failures.

    Args:
        result: Raw model response with native calls in its last event.
        config: Controls whether invalid arguments produce errors or empty objects.

    Returns:
        Calls in provider order with their IDs and parsing errors.
    """
    calls = []
    for raw in result.trajectory[-1].get("tool_calls", []):
        function = raw["function"]
        error = ""
        try:
            arguments = extract_json_object(function["arguments"] or "{}", repair=False)
        except (ValueError, TypeError) as exc:
            arguments = {}
            if config.invalid_tool_arguments == "error":
                error = f"Invalid tool arguments: {exc}"
        calls.append(ToolCall(name=function["name"], arguments=arguments, call_id=raw["id"], error=error))
    return ToolTurn(text=result.output, calls=calls)


def native_observation(call: ToolCall, content: str) -> dict[str, JsonValue]:
    """Encode an environment result as a native tool message.

    Args:
        call: Original provider invocation, including its ID.
        content: Verbatim environment result.

    Returns:
        A tool-role message matching the invocation.
    """
    return {"role": "tool", "tool_call_id": call.call_id, "name": call.name, "content": content}


def text_observation(call: ToolCall, content: str) -> dict[str, JsonValue]:
    """Encode an environment result as a textual observation.

    Args:
        call: Invocation being answered.
        content: Verbatim environment result.

    Returns:
        A user message with the Observation prefix.
    """
    return {"role": "user", "content": f"Observation: {content}"}
