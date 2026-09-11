"""Shared contracts and lifecycle policies for agent invocations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mindmemos_skill.algos.trace2skill.offline_evolution.impl.config import setting_float, setting_optional_float


@dataclass
class InvokeResult:
    """Result of one local LLM invocation."""

    text: str
    channel: str
    input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None
    raw: Any = None
    permission_denials: list[dict[str, Any]] = field(default_factory=list)
    session_id: str | None = None
    attempts: int = 1
    finish_reason: str | None = None
    provider_request_id: str | None = None
    transcript: list[dict[str, Any]] = field(default_factory=list)


class InvokeError(RuntimeError):
    """Raised when an invocation fails for any reason."""

    def __init__(
        self,
        channel: str,
        message: str,
        *,
        cause: Exception | None = None,
        retryable: bool = True,
        resume_session: bool = False,
        session_id: str | None = None,
        partial_result: InvokeResult | None = None,
    ):
        super().__init__(f"[{channel}] {message}")
        self.channel = channel
        self.cause = cause
        self.retryable = retryable
        self.resume_session = resume_session
        self.session_id = session_id
        self.partial_result = partial_result
        self.attempts: int | None = None


def agent_timeout_sec(backend: str, stage: str | None = None) -> float:
    """Read ``agent.timeout_sec``, honouring ``agent.timeout_overrides.<stage>``.

    Timeout retries resume the session, so the value bounds one slice of
    work, not the whole task; stages whose slices routinely exceed the
    global default declare themselves via ``stage``.
    """
    try:
        if stage:
            override = setting_optional_float(f"agent.timeout_overrides.{stage}")
            if override is not None:
                return override
        return setting_float("agent.timeout_sec")
    except ValueError as exc:
        raise InvokeError(
            backend,
            str(exc),
            cause=exc,
            retryable=False,
        ) from exc


def retry_followup_message(error: InvokeError) -> str:
    """Tell a reusable session why its complete result was rejected."""
    detail = str(error)
    lines = [
        "The previous turn was rejected by the caller and must be corrected.",
        "",
        "Failure reported by the caller:",
        detail,
        "",
        "Continue from the existing conversation context.",
        "Do not repeat file discovery or reads that already succeeded unless "
        "they are necessary to correct the failure.",
    ]
    if "permission denied" in detail.lower():
        lines.append(
            "For file tools, use only exact relative paths returned by Glob; "
            "do not invent, translate, or reconstruct absolute paths."
        )
    lines.append("Correct only what is needed, then return the complete final response.")
    return "\n".join(lines)


def prepare_retry(
    *,
    original_user: str,
    current_kwargs: dict[str, Any],
    result: InvokeResult | None,
    error: InvokeError,
) -> tuple[str, dict[str, Any]]:
    """Resume a retry whenever the preceding attempt exposed a session."""
    if "prompt is too long" in str(error).casefold():
        restarted = dict(current_kwargs)
        restarted.pop("session_id", None)
        return original_user, restarted

    candidates = (
        result.session_id if result is not None else None,
        error.session_id,
        current_kwargs.get("session_id"),
    )
    session_id = next(
        (candidate.strip() for candidate in candidates if isinstance(candidate, str) and candidate.strip()),
        None,
    )
    if session_id is None:
        restarted = dict(current_kwargs)
        restarted.pop("session_id", None)
        return original_user, restarted
    resumed = dict(current_kwargs)
    resumed["session_id"] = session_id
    return retry_followup_message(error), resumed


__all__ = [
    "InvokeError",
    "InvokeResult",
    "agent_timeout_sec",
    "prepare_retry",
    "retry_followup_message",
]
