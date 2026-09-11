"""Client initialization and caching for API protocols."""

from __future__ import annotations

import contextvars
import threading
from typing import Any

from mindmemos_skill.algos.trace2skill.offline_evolution.impl.config import setting_float, setting_optional_str
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.llm.invocation import InvokeError
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.llm.react.backend.config import _nonnegative_setting
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.llm.react.protocols import APIProtocol

_CLIENT_LOCK = threading.Lock()
_CLIENTS: dict[tuple[Any, ...], Any] = {}

# Context-local injected client from the host wrapper. When set, get_client()
# returns this instead of building one from env; this lets the host's DI take
# precedence while env-based construction remains a fallback for standalone use.
_INJECTED_CLIENT: contextvars.ContextVar[Any | None] = contextvars.ContextVar(
    "offline_evolution_injected_client", default=None
)


def current_injected_client() -> Any | None:
    """Return the host-injected client bound in the calling context, or None."""
    return _INJECTED_CLIENT.get()


def get_client(protocol: APIProtocol, *, backend_name: str) -> Any:
    """Return a client for the given protocol, preferring host injection over env.

    The host wrapper binds its own LLMClient via contextvars when it runs the
    vendored pipeline. When that binding is present this function returns it
    directly, letting the host's DI (model endpoint, timeout, retry, audit sink)
    take precedence. When no injected client is bound this function falls back
    to constructing one from env/config, so standalone vendored usage still works.
    """
    injected = current_injected_client()
    if injected is not None:
        return injected

    # Fallback: construct from env/config (original logic unchanged).
    api_key = setting_optional_str("LLM_API_KEY")
    base_url = setting_optional_str("LLM_BASE_URL")
    retries = _nonnegative_setting("llm.transport_max_retries")
    timeout = setting_float("agent.timeout_sec")
    key = (protocol.name, api_key, base_url, retries, timeout)
    with _CLIENT_LOCK:
        client = _CLIENTS.get(key)
        if client is not None:
            return client
        try:
            client = protocol.create_client(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
                max_retries=retries,
            )
        except Exception as exc:
            raise InvokeError(
                backend_name,
                f"{protocol.name} client initialization failed: {type(exc).__name__}: {exc}",
                cause=exc,
                retryable=False,
            ) from exc
        _CLIENTS[key] = client
        return client


__all__ = ["_INJECTED_CLIENT", "current_injected_client", "get_client"]
