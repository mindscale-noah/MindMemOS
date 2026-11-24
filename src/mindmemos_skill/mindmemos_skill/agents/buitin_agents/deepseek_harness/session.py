"""Read the headless run's retained DSH v2 session log.

Supported contracts: installed dsh 0.1.0-rc.6 (v0) and source d347e703 (v2).
The patch selects plaintext JSONL in a fresh per-run root, avoiding old sessions.
"""

from collections.abc import Mapping
from pathlib import Path

from ..cli import parse_json_lines


def read_session_events(root: Path) -> tuple[Path, list[dict]]:
    """Read the single completed main session from a fresh run directory.

    Args:
        root: Per-run directory containing uncompressed v0 or v2 session files.

    Returns:
        Main session path and ordered native events without the header.

    Raises:
        RuntimeError: The session is missing, ambiguous, unsupported, or incomplete.
    """
    sessions: list[tuple[Path, list[dict]]] = []
    for path in sorted(root.rglob("session*.jsonl")):
        rows = parse_json_lines(path.read_text(encoding="utf-8"), cli_name="DeepSeek session")
        header = rows[0]
        if header.get("type") != "session" or header.get("version") not in {0, 2}:
            raise RuntimeError(f"Unsupported DeepSeek session format: {path}")
        if header.get("origin") == "subagent" or header.get("delegationDepth", 0) > 0:
            continue
        sessions.append((path, rows[1:]))
    if len(sessions) != 1:
        raise RuntimeError(f"Expected one DeepSeek headless session, found {len(sessions)} in {root}")
    path, events = sessions[0]
    terminal = next((event for event in reversed(events) if event.get("type") == "turn/end"), None)
    data = terminal.get("data") if terminal is not None else None
    reason = data.get("reason") if isinstance(data, Mapping) else None
    if not isinstance(reason, Mapping) or reason.get("kind") != "completed":
        raise RuntimeError(f"DeepSeek session has no completed terminal turn: {path}")
    return path, events
