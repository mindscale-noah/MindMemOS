"""Dispatch offline stage calls through the existing registered native Agents."""

from pathlib import Path

from .....agents.buitin_agents.claude.config import ClaudeConfig
from .....agents.buitin_agents.codex.config import CodexConfig
from .....agents.typing import AgentResult
from .claude import run_claude
from .codex import run_codex
from .typing import SessionOptions


async def run_agent(
    config: ClaudeConfig | CodexConfig, *, user: str, workspace: Path, session: SessionOptions
) -> AgentResult:
    """Invoke the selected backend with the same offline stage contract.

    Args:
        config: Registered Claude or Codex connection and native settings.
        user: Stage instruction or retry feedback.
        workspace: Algorithm execution root.
        session: File scopes, persistent home, identity and event log.

    Returns:
        Shared output with stage-compatible token usage and native events.
    """
    if isinstance(config, ClaudeConfig):
        return await run_claude(config, user=user, workspace=workspace, session=session)
    return await run_codex(config, user=user, workspace=workspace, session=session)


def event_identity(event: dict[str, object]) -> str | None:
    """Read a native session identity without changing the event payload.

    Args:
        event: Native event fields; session_id or thread_id carries the optional identity.

    Returns:
        Observed session or thread ID, or None when the event has none.
    """
    for field in ("session_id", "thread_id"):
        value = event.get(field)
        if isinstance(value, str) and value:
            return value
    return None


def session_identity(result: AgentResult) -> str | None:
    """Read the most recent native identity from a completed invocation.

    Args:
        result: Shared output retaining the backend's events.

    Returns:
        Last observed session or thread ID, or None when unavailable.
    """
    return next((identity for event in reversed(result.trajectory) if (identity := event_identity(event))), None)
