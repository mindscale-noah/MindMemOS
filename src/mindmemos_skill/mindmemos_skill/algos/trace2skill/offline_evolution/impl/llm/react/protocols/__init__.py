"""Registry for supported LLM API wire protocols."""

from __future__ import annotations

from typing import Literal

from mindmemos_skill.algos.trace2skill.offline_evolution.impl.llm.react.protocols.anthropic import AnthropicProtocol
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.llm.react.protocols.base import APIProtocol
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.llm.react.protocols.openai import OpenAIProtocol

ProtocolName = Literal["openai", "anthropic"]
_PROTOCOLS: dict[str, APIProtocol] = {
    "openai": OpenAIProtocol(),
    "anthropic": AnthropicProtocol(),
}


def resolve_protocol(name: str) -> APIProtocol:
    normalized = str(name or "").strip().lower()
    try:
        return _PROTOCOLS[normalized]
    except KeyError as exc:
        allowed = ", ".join(sorted(_PROTOCOLS))
        raise ValueError(f"unsupported LLM protocol {name!r}; expected one of: {allowed}") from exc


__all__ = ["APIProtocol", "ProtocolName", "resolve_protocol"]
