"""Contracts for a single, stateless compression request."""

from __future__ import annotations

import json
import math
import threading
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit

from ..errors import MindMemOSSDKError

CompressionMode = Literal["incremental", "full"]


def _chat_url(base_url: str) -> str:
    try:
        parsed = urlsplit(base_url.strip())
        valid = (
            parsed.scheme in ("http", "https")
            and parsed.hostname
            and not (parsed.username or parsed.password or parsed.query or parsed.fragment)
        )
        parsed.port  # Validate the port before attempting network I/O.
    except (AttributeError, TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError("base_url must be an HTTP(S) URL without credentials, query or fragment")
    path = parsed.path.rstrip("/") or "/v1"
    if not path.endswith("/chat/completions"):
        path += "/chat/completions"
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


@dataclass(frozen=True, kw_only=True)
class CompressionConfig:
    """Model connection only; thresholds and history belong to the caller.

    Credentials are explicit and independent of MindMemOSClient's long-term
    memory configuration.
    """

    api_key: str = field(repr=False)
    base_url: str = "https://openrouter.ai/api/v1"
    model: str = "qwen/qwen3.5-122b-a10b"
    reasoning: dict[str, Any] | None = field(default_factory=lambda: {"effort": "high"})
    timeout_seconds: float = 240.0
    headers: dict[str, str] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        _chat_url(self.base_url)
        if not isinstance(self.api_key, str) or any(c in self.api_key for c in "\r\n\0"):
            raise ValueError("api_key must be a single-line string")
        if not isinstance(self.model, str) or not self.model or any(c.isspace() for c in self.model):
            raise ValueError("model must be a nonempty API model ID without whitespace")
        try:
            valid_timeout = (
                type(self.timeout_seconds) in (int, float)
                and math.isfinite(self.timeout_seconds)
                and self.timeout_seconds > 0
                and self.timeout_seconds <= threading.TIMEOUT_MAX
            )
        except OverflowError:
            valid_timeout = False
        if not valid_timeout:
            raise ValueError("timeout_seconds must be finite, positive and within the platform timer limit")
        if self.reasoning is not None:
            if not isinstance(self.reasoning, dict) or not self.reasoning:
                raise ValueError("reasoning must be a nonempty JSON object or None")
            try:
                json.dumps(self.reasoning, allow_nan=False)
            except (TypeError, ValueError, RecursionError):
                raise ValueError("reasoning must be a JSON object with finite values") from None
        if not isinstance(self.headers, dict) or any(
            not isinstance(k, str)
            or not isinstance(v, str)
            or not k
            or any(c.isspace() or c == ":" for c in k)
            or any(c in k + v for c in "\r\n\0")
            for k, v in self.headers.items()
        ):
            raise ValueError("headers must contain single-line string names and values")
        try:
            object.__setattr__(self, "reasoning", deepcopy(self.reasoning))
        except RecursionError:
            raise ValueError("reasoning exceeds supported nesting") from None
        object.__setattr__(self, "headers", dict(self.headers))


@dataclass(frozen=True)
class CompressionResult:
    summary: str
    usage: dict[str, Any] | None = None
    latency_seconds: float = 0.0


class CompressionError(MindMemOSSDKError):
    """Safe diagnostic; excludes provider bodies and credentials.

    ``usage`` remains available even when a billed response is rejected.
    Missing usage is unknown, never an inferred zero.
    """

    def __init__(self, message: str, *, code: str = "invalid_result") -> None:
        super().__init__(message)
        self.code = code
        self.usage: dict[str, Any] | None = None
        self.latency_seconds = 0.0
        self.request_dispatched = False
