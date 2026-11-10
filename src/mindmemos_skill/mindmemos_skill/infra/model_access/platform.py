"""Bind authenticated projects to Platform model calls."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from os import getenv
from time import time
from typing import Final

import httpx
import jwt
from pydantic import AnyHttpUrl

from .typing import EmbeddingSpace

PLATFORM_MODEL_SECRET_ENV: Final = "MINDMEMOS_SKILL_MODEL_GATEWAY_SECRET"
PLATFORM_CHAT_MODEL: Final = "platform-chat"
PLATFORM_EMBEDDING_MODEL: Final = "platform-embedding"


class PlatformModelError(RuntimeError):
    """A sanitized Platform failure suitable for a Skill API response."""

    def __init__(self, status: int) -> None:
        """Translate a Platform response to a public Skill status.

        Args:
            status: Upstream status, or 503 when the transport is unreachable.
        """
        self.status_code = status if status in {403, 409, 422, 429} else 503
        messages = {
            403: "The Skill project is not authorized for this model",
            409: "The requested Skill model or embedding space is unavailable",
            422: "The Platform model rejected the Skill request options",
            429: "The Platform model quota or rate limit was exceeded",
            503: "The Platform model gateway is unavailable",
        }
        super().__init__(messages[self.status_code])


@dataclass(frozen=True)
class PlatformModelGateway:
    """Retain deployment-owned connection settings outside pipeline serialization."""

    base_url: str
    """Internal Skill model gateway API root."""

    secret: str = field(repr=False)
    """Dedicated signing key shared with Platform."""

    @classmethod
    def from_origin(cls, origin: AnyHttpUrl) -> "PlatformModelGateway":
        """Load the connection for the configured Platform origin.

        Args:
            origin: Deployment-owned HTTP origin.

        Returns:
            Connection with its signing key read from the service environment.
        """
        secret = getenv(PLATFORM_MODEL_SECRET_ENV, "")
        if len(secret.strip().encode()) < 32 or secret.strip().casefold().startswith("change-me"):
            raise ValueError(f"{PLATFORM_MODEL_SECRET_ENV} requires a dedicated key of at least 32 bytes")
        return cls(f"{str(origin).rstrip('/')}/internal/v1/skill-models", secret.strip())


_active_route: ContextVar[tuple[PlatformModelGateway, str] | None] = ContextVar(
    "skill_platform_model_route", default=None
)


@contextmanager
def bind_platform_models(gateway: PlatformModelGateway | None, project_id: str) -> Iterator[None]:
    """Bind a model route for an already authorized request or worker task.

    Args:
        gateway: Optional deployment-owned connection.
        project_id: Authorized project namespace.
    """
    token = _active_route.set((gateway, project_id) if gateway is not None else None)
    try:
        yield
    finally:
        _active_route.reset(token)


def platform_connection() -> tuple[str, str]:
    """Return the internal API root and a fresh project-scoped bearer token."""
    route = _active_route.get()
    if route is None:
        raise RuntimeError("An authorized Platform model route is required")
    gateway, project_id = route
    now = int(time())
    token = jwt.encode(
        {
            "iss": "mindmemos-skill",
            "sub": "mindmemos-skill",
            "aud": "platform-skill-model",
            "scope": "skill:model",
            "project_id": project_id,
            "iat": now,
            "exp": now + 60,
        },
        gateway.secret,
        algorithm="HS256",
    )
    return gateway.base_url, token


async def fetch_embedding_space() -> EmbeddingSpace:
    """Read the current project's authorized vector space.

    Returns:
        Model identity and dimensions accepted by the Platform gateway.

    Raises:
        PlatformModelError: The gateway rejects the request or returns an invalid profile.
    """
    base_url, token = platform_connection()
    try:
        async with httpx.AsyncClient(trust_env=False, timeout=10) as client:
            response = await client.get(f"{base_url}/embedding-profile", headers={"Authorization": f"Bearer {token}"})
            response.raise_for_status()
        return EmbeddingSpace.model_validate(response.json())
    except httpx.HTTPStatusError as exc:
        raise PlatformModelError(exc.response.status_code) from None
    except (httpx.RequestError, ValueError):
        raise PlatformModelError(503) from None
