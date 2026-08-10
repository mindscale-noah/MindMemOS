"""Callable tools exposed by the OpenAI-compatible ReAct family."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Tool:
    """A runtime callable plus its OpenAI function schema."""

    name: str
    description: str
    func: Callable[..., Any]
    parameters: dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})
    deliver_result_as_user: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("tool name must not be empty")
        if self.parameters.get("type") != "object":
            raise ValueError("tool parameters must be an object JSON schema")

    def to_openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    async def call(self, arguments: dict[str, Any]) -> Any:
        """Run synchronous handlers off-loop while preserving awaitable results."""

        if inspect.iscoroutinefunction(self.func):
            return await self.func(**arguments)
        result = await asyncio.to_thread(self.func, **arguments)
        if inspect.isawaitable(result):
            return await result
        return result


def tool(
    name: str | None = None,
    description: str | None = None,
    parameters: dict[str, Any] | None = None,
) -> Callable[[Callable[..., Any]], Tool]:
    """Decorate a function as an OpenAI-compatible :class:`Tool`."""

    def decorator(func: Callable[..., Any]) -> Tool:
        return Tool(
            name=name or func.__name__,
            description=description or inspect.cleandoc(func.__doc__ or ""),
            func=func,
            parameters=parameters or {"type": "object", "properties": {}},
        )

    return decorator


__all__ = ["Tool", "tool"]
