"""Configuration loading for LLM backend."""

from __future__ import annotations

import contextvars

from mindmemos_skill.algos.trace2skill.offline_evolution.impl.config import (
    setting_float,
    setting_int,
    setting_optional_str,
    setting_raw,
)
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.llm.react.protocol import LLMGenerationConfig
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.llm.react.protocols import APIProtocol, resolve_protocol

# Context-local injected model name from the host wrapper. When set, model()
# returns this instead of reading from env/config; this lets the host's DI
# take precedence while env-based resolution remains a fallback.
_INJECTED_MODEL: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "offline_evolution_injected_model", default=None
)


def _positive_setting(name: str) -> int:
    value = setting_int(name)
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return value


def _nonnegative_setting(name: str) -> int:
    value = setting_int(name)
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return value


def _ratio_setting(name: str) -> float:
    value = setting_float(name)
    if not 0 < value < 1:
        raise ValueError(f"{name} must be between 0 and 1, got {value}")
    return value


def generation_config() -> LLMGenerationConfig:
    return LLMGenerationConfig(
        temperature=setting_float("llm.temperature"),
        context_window_tokens=_positive_setting("llm.context_window_tokens"),
        max_output_tokens=_positive_setting("llm.max_output_tokens"),
        reasoning_effort=setting_optional_str("llm.reasoning_effort"),
    )


def current_injected_model() -> str | None:
    """Return the host-injected model name bound in the calling context, or None."""
    return _INJECTED_MODEL.get()


def model() -> str:
    """Return the model identifier, preferring host injection over env/config."""
    injected = current_injected_model()
    if injected is not None:
        return injected
    return setting_raw("llm.model")


def protocol() -> APIProtocol:
    return resolve_protocol(setting_raw("llm.protocol"))


def output_token_parameter() -> str:
    value = setting_raw("llm.output_token_parameter")
    if value not in {"max_tokens", "max_completion_tokens"}:
        raise ValueError("llm.output_token_parameter must be max_tokens or max_completion_tokens")
    return value


__all__ = [
    "_INJECTED_MODEL",
    "current_injected_model",
    "generation_config",
    "model",
    "protocol",
    "output_token_parameter",
    "_positive_setting",
    "_nonnegative_setting",
    "_ratio_setting",
]
