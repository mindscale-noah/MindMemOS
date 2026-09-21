"""Reference transcript rendering for actively sampled environment results."""

import json
from typing import Any


def render_trajectory(events: list[dict[str, Any]], max_chars: int) -> str:
    """Render ordered trajectory events into the compact LLM transcript format.

    Args:
        events: Original ordered environment events.
        max_chars: Maximum retained characters per event under the reference truncation rule.
    """

    lines: list[str] = []
    for event in events:
        if "role" in event and "content" in event:
            rendered = f"[{event.get('role', '?')}] {event.get('content', '')}"
        elif "text" in event:
            rendered = f"[text] {event.get('text', '')}"
        elif "url" in event:
            rendered = f"[url] {event.get('url', '')}"
        elif "file_name" in event:
            rendered = f"[file] {event.get('file_name', '')}"
        else:
            rendered = json.dumps(event, ensure_ascii=False, sort_keys=True)
        lines.append(_truncate(rendered, max_chars))
    return "\n".join(lines)


def _truncate(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    head = text[: max_chars // 2]
    tail = text[-max_chars // 2 :]
    return f"{head}\n…[{len(text) - max_chars} chars elided]…\n{tail}"
