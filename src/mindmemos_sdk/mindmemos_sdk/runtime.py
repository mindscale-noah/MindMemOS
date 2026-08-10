"""Unified asynchronous SDK portal lifecycle."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Mapping
from typing import Any, Self

from mindmemos_skill import SkillApplication

from .composition import (
    ConnectionPool,
    build_memory_backend,
    build_portal_connections,
    build_skill_remote_port,
)
from .config import CompiledSDKPortalConfigV2, SDKConfigCompilerV2, SDKPortalConfigV2
from .connections import AsyncConnection
from .memory import AsyncMemoryClient
from .memory.core import MemoryDefaults
from .skills import AsyncSkillClient


class SDKPortalRuntime:
    """Own connections, SkillApplication and SDK resource facades in one event loop."""

    def __init__(
        self,
        config: SDKPortalConfigV2 | Mapping[str, Any] | CompiledSDKPortalConfigV2,
        *,
        connections: dict[str, AsyncConnection] | None = None,
        compiler: SDKConfigCompilerV2 | None = None,
    ) -> None:
        self._config = (
            config
            if isinstance(config, CompiledSDKPortalConfigV2)
            else (compiler or SDKConfigCompilerV2()).compile(config)
        )
        available = connections or build_portal_connections(self._config)
        profile = self._config.profile
        required = {profile.memory_connection}
        if profile.skill_connection is not None:
            required.add(profile.skill_connection)
        missing = sorted(required - available.keys())
        if missing:
            raise ValueError(f"missing SDK portal connections: {', '.join(missing)}")
        self._pool = ConnectionPool({name: available[name] for name in required})
        self._application: SkillApplication | None = None
        self._memory: AsyncMemoryClient | None = None
        self._skills: AsyncSkillClient | None = None
        self._owner_loop: asyncio.AbstractEventLoop | None = None
        self._started = False
        self._closed = False

    @property
    def config(self) -> CompiledSDKPortalConfigV2:
        return self._config

    @property
    def application(self) -> SkillApplication:
        self._require_started()
        assert self._application is not None
        return self._application

    @property
    def memory(self) -> AsyncMemoryClient:
        self._require_started()
        assert self._memory is not None
        return self._memory

    @property
    def skills(self) -> AsyncSkillClient:
        self._require_started()
        assert self._skills is not None
        return self._skills

    async def start(self) -> Self:
        if self._closed:
            raise RuntimeError("SDK portal runtime is closed")
        loop = asyncio.get_running_loop()
        if self._started:
            if loop is not self._owner_loop:
                raise RuntimeError("SDK portal runtime cannot be used across event loops")
            return self
        self._owner_loop = loop
        try:
            await self._pool.open()
            profile = self._config.profile
            memory_connection = self._pool.get(profile.memory_connection, capability="memory")
            remote = None
            if profile.skill_connection is not None:
                skill_connection = self._pool.get(profile.skill_connection, capability="skills")
                remote = build_skill_remote_port(skill_connection)
            self._application = await SkillApplication.from_config(
                profile.skill_application,
                remote=remote,
            )
            identity = profile.identity
            defaults = profile.memory_defaults
            self._memory = AsyncMemoryClient(
                build_memory_backend(memory_connection),
                default_user_id=identity.user_id,
                default_app_id=identity.app_id,
                default_agent_id=identity.agent_id,
                default_session_id=identity.session_id,
                memory_defaults=MemoryDefaults(
                    user_id=identity.user_id,
                    app_id=identity.app_id,
                    agent_id=identity.agent_id,
                    session_id=identity.session_id,
                    add_mode=defaults.add_mode,
                    add_default_role=defaults.add_default_role,
                    add_auto_skill_context=defaults.add_auto_skill_context,
                    search_top_k=defaults.search_top_k,
                    search_strategy=defaults.search_strategy,
                    search_rerank=defaults.search_rerank,
                    search_score_threshold=defaults.search_score_threshold,
                    search_filters=defaults.search_filters,
                    get_top_k=defaults.get_top_k,
                    get_filters=defaults.get_filters,
                    feedback_mode=defaults.feedback_mode,
                    dreaming_mode=defaults.dreaming_mode,
                ),
                skill_application=self._application,
            )
            self._skills = AsyncSkillClient(self._application)
        except BaseException:
            with contextlib.suppress(BaseException):
                await self._close_partial()
            self._owner_loop = None
            self._closed = True
            raise
        self._started = True
        return self

    async def aclose(self) -> None:
        if self._closed:
            return
        if self._owner_loop is not None and asyncio.get_running_loop() is not self._owner_loop:
            raise RuntimeError("SDK portal runtime cannot be closed from another event loop")
        self._closed = True
        await self._close_partial()
        self._started = False

    async def _close_partial(self) -> None:
        error: BaseException | None = None
        if self._application is not None:
            try:
                await self._application.close()
            except BaseException as exc:
                error = exc
            self._application = None
        try:
            await self._pool.aclose()
        except BaseException as exc:
            if error is None:
                error = exc
        self._memory = None
        self._skills = None
        if error is not None:
            raise error

    def _require_started(self) -> None:
        if not self._started:
            raise RuntimeError("SDK portal runtime is not started")

    async def __aenter__(self) -> Self:
        return await self.start()

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()


__all__ = ["SDKPortalRuntime"]
