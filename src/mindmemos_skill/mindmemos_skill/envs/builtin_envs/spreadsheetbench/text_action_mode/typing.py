"""Parsed text-action policy responses."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class PolicyResponseType(StrEnum):
    """Released textual policy response categories."""

    ACTION = "action"
    """Execute the parsed environment tool invocation."""

    TASK_COMPLETE = "task_complete"
    """Check whether the requested output exists before finishing."""

    FORMAT_ERROR = "format_error"
    """Return the exact correction feedback for invalid action syntax."""


@dataclass(frozen=True, slots=True)
class PolicyAction:
    """One parsed tool invocation from the reference text protocol."""

    name: str
    """Requested tool name from the textual action payload."""

    arguments: dict[str, Any]
    """Tool argument mapping parsed from the action JSON."""


@dataclass(frozen=True, slots=True)
class PolicyResponse:
    """Parsed action, completion or format-error result."""

    response_type: PolicyResponseType
    """Whether the response is an action, completion or format error."""

    action: PolicyAction | None = None
    """Parsed tool invocation, or None when no action is available."""

    error_message: str | None = None
    """Correction feedback for a malformed action payload."""
