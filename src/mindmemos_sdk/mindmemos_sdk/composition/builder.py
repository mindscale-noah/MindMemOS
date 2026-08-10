"""Build SDK connections and resource backends from configuration."""

from __future__ import annotations

from mindmemos_skill import SkillRemotePort

from ..config import CompiledSDKPortalConfigV2, DefaultsConfig
from ..connections import AsyncConnection, HttpConnection
from ..memory.backends import AsyncMemoryBackend, HttpMemoryBackend
from ..skills.http_adapter import HttpSkillRemoteAdapter


def build_portal_connections(config: CompiledSDKPortalConfigV2) -> dict[str, AsyncConnection]:
    """Construct the active portal profile's named connections without opening them."""

    return {name: HttpConnection(connection_config) for name, connection_config in config.profile.connections.items()}


def build_memory_backend(
    connection: AsyncConnection,
    *,
    defaults: DefaultsConfig | None = None,
) -> AsyncMemoryBackend:
    if isinstance(connection, HttpConnection):
        return HttpMemoryBackend(connection)
    raise TypeError(f"connection does not provide a Memory backend: {type(connection).__name__}")


def build_skill_remote_port(connection: AsyncConnection) -> SkillRemotePort:
    """Adapt a borrowed SDK connection without transferring lifecycle ownership."""

    if isinstance(connection, HttpConnection):
        return HttpSkillRemoteAdapter(connection)
    raise TypeError(f"connection does not provide a Skill remote port: {type(connection).__name__}")


__all__ = [
    "build_memory_backend",
    "build_portal_connections",
    "build_skill_remote_port",
]
