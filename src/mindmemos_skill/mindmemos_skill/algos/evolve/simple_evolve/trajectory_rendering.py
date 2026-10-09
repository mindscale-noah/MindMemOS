"""Role-aware middle clipping of structured historical execution evidence."""

from pydantic import JsonValue

from .config import SimpleEvolverConfig

_MARKER = "\n[... middle omitted ...]\n"


def _clip(text: str, limit: int) -> str:
    """Retain both ends within the exact character budget."""
    if len(text) <= limit:
        return text
    marker = _MARKER[:limit]
    remaining = limit - len(marker)
    head = (remaining + 1) // 2
    tail = remaining // 2
    return text[:head] + marker + (text[-tail:] if tail else "")


def clip_payload(value: JsonValue, limit: int) -> JsonValue:
    """Bound text leaves without breaking nested message or tool-call structure.

    Args:
        value: JSON evidence, including structured content and tool arguments.
        limit: Total character budget shared by all string values in this payload.

    Returns:
        Detached JSON evidence with long strings clipped in the middle.
    """

    lengths = []

    def collect_lengths(item):
        if isinstance(item, str):
            lengths.append(len(item))
        elif isinstance(item, dict):
            for child in item.values():
                collect_lengths(child)
        elif isinstance(item, list):
            for child in item:
                collect_lengths(child)

    collect_lengths(value)
    # Share the budget among long text leaves without shortening small tool names.
    lower, upper = 0, max(lengths, default=0)
    while lower < upper:
        ceiling = (lower + upper + 1) // 2
        if sum(min(length, ceiling) for length in lengths) <= limit:
            lower = ceiling
        else:
            upper = ceiling - 1

    def trim(item):
        if isinstance(item, str):
            return _clip(item, lower)
        if isinstance(item, dict):
            return {key: trim(child) for key, child in item.items()}
        if isinstance(item, list):
            return [trim(child) for child in item]
        return item

    return trim(value)


def clip_message(message: dict[str, JsonValue], config: SimpleEvolverConfig) -> dict[str, JsonValue]:
    """Clip message payload by role while preserving routing identifiers.

    Args:
        message: Historical user, assistant, tool, or observation record.
        config: Per-role character budgets.

    Returns:
        A detached message preserving identifiers and bounded evidence text.
    """
    role = message.get("role")
    if role in ("user", "system"):
        limit = config.user_message_chars
    elif role == "assistant":
        limit = config.assistant_message_chars
    else:
        limit = config.tool_message_chars
    identifiers = {key: value for key, value in message.items() if key in {"role", "name", "tool_call_id", "id"}}
    payload = {key: value for key, value in message.items() if key not in identifiers}
    return {**identifiers, **clip_payload(payload, limit)}
