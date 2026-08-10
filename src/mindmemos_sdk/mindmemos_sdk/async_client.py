"""Asynchronous root SDK facade backed by ``SDKPortalRuntime``."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Self

from .config import CompiledSDKPortalConfigV2, SDKPortalConfigV2
from .connections import AsyncConnection
from .memory import AsyncMemoryClient
from .runtime import SDKPortalRuntime
from .skills import AsyncSkillClient


class AsyncMindMemOSClient:
    """Expose Memory and Skill facades from one portal-owned lifecycle."""

    def __init__(
        self,
        *,
        config: SDKPortalConfigV2 | Mapping[str, Any] | CompiledSDKPortalConfigV2,
        connections: dict[str, AsyncConnection] | None = None,
    ) -> None:
        self._runtime = SDKPortalRuntime(config, connections=connections)

    @property
    def memory(self) -> AsyncMemoryClient:
        return self._runtime.memory

    @property
    def skills(self) -> AsyncSkillClient:
        return self._runtime.skills

    @property
    def runtime(self) -> SDKPortalRuntime:
        return self._runtime

    async def start(self) -> Self:
        await self._runtime.start()
        return self

    async def aclose(self) -> None:
        await self._runtime.aclose()

    async def __aenter__(self) -> Self:
        return await self.start()

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()


__all__ = ["AsyncMindMemOSClient"]
