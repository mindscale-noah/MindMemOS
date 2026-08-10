"""Shared MindMemOS backend composition for evaluation runners."""

from __future__ import annotations

from typing import Self

from mindmemos_sdk.config import ConfigManager, DefaultsConfig, HttpConnectionConfig

from mindmemos_sdk import AsyncMindMemOSClient


class MindMemOSBackend:
    """Own one SDK client shared by Memory and Skill evaluation resources."""

    def __init__(self, client: AsyncMindMemOSClient) -> None:
        self.client = client
        self._started = False

    @property
    def memory(self):
        """Return the SDK Memory resource client."""

        return self.client.memory

    @property
    def skills(self):
        """Return the SDK Skill resource client."""

        return self.client.skills

    async def start(self) -> Self:
        """Open the configured HTTP connection once."""

        if not self._started:
            await self.client.start()
            self._started = True
        return self

    async def aclose(self) -> None:
        """Close the shared connection."""

        await self.client.aclose()
        self._started = False

    async def __aenter__(self) -> Self:
        return await self.start()

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()


def build_mindmemos_backend(
    *,
    base_url: str,
    api_key: str | None = None,
    timeout_seconds: float = 30.0,
    max_retries: int = 2,
    user_id: str | None = None,
    app_id: str | None = None,
    agent_id: str | None = None,
    session_id: str | None = None,
    config_manager: ConfigManager | None = None,
) -> MindMemOSBackend:
    """Build one backend over the public MindMemOS HTTP API."""

    if not base_url:
        raise ValueError("base_url is required for an HTTP MindMemOS backend")
    connection = HttpConnectionConfig(
        base_url=base_url,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
    )

    connection_name = "eval"
    config = (config_manager or ConfigManager()).default_portal()
    profile = config.profiles[config.active_profile]
    profile.default_connection = connection_name
    profile.connections = {connection_name: connection}
    profile.identity = DefaultsConfig(
        user_id=user_id,
        app_id=app_id,
        agent_id=agent_id,
        session_id=session_id,
    )
    profile.memory.connection = connection_name
    profile.skill.remote.connection = connection_name
    return MindMemOSBackend(AsyncMindMemOSClient(config=config))


__all__ = [
    "MindMemOSBackend",
    "build_mindmemos_backend",
]
