"""Use the configured Skill service through the public SDK client."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import HTTPException
from mindmemos_sdk.config import ConfigManager
from mindmemos_sdk.skills import AsyncSkillClient
from mindmemos_sdk.transport import AsyncHttpTransport


@asynccontextmanager
async def skill_client(manager: ConfigManager, *, long_running: bool = False) -> AsyncIterator[AsyncSkillClient]:
    """Open a Skill SDK client for the latest saved connection settings.

    Args:
        manager: Owner of the UI process's SDK settings file.
        long_running: Allow at least five minutes for an operation that should not retry automatically.

    Yields:
        A client scoped by the configured Skill credential.

    Raises:
        HTTPException: The Skill service has no usable credential.
    """
    config = manager.load_or_default()
    base_url, api_key = config.resolve_skill_connection()
    if not base_url or not api_key:
        raise HTTPException(503, "Configure the Skill service URL and API key before using Skills")
    async with AsyncHttpTransport(
        base_url=base_url,
        api_key=api_key,
        timeout_seconds=max(config.network.timeout_seconds, 300) if long_running else config.network.timeout_seconds,
        max_retries=0 if long_running else config.network.max_retries,
    ) as transport:
        yield AsyncSkillClient(transport, path_prefix="")
