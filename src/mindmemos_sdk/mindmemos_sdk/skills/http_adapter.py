"""SDK HTTP projection of the transport-neutral Skill v2 remote port."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from functools import wraps
from typing import Any, ParamSpec, TypeVar
from urllib.parse import quote

from pydantic import ValidationError

from mindmemos_skill import (
    RemoteEvolveRequest,
    RemoteEvolveResult,
    RemotePushRequest,
    RemotePushResult,
    RemoteSyncRequest,
    RemoteSyncResult,
    RemoteTrajectoryListRequest,
    RemoteTrajectoryPage,
    RemoteTrajectoryReportRequest,
    RemoteTrajectoryReportResult,
    RemoteVersionContent,
    RemoteVersionsPage,
    SkillRemoteRequestError,
)

from ..connections import HttpConnection
from ..errors import ApiError, AuthRequiredError, TransportError

_Parameters = ParamSpec("_Parameters")
_ResultT = TypeVar("_ResultT")


class _InvalidRemoteResponseError(ValueError):
    pass


def _translate_remote_errors(
    method: Callable[_Parameters, Awaitable[_ResultT]],
) -> Callable[_Parameters, Awaitable[_ResultT]]:
    @wraps(method)
    async def wrapper(*args: _Parameters.args, **kwargs: _Parameters.kwargs) -> _ResultT:
        try:
            return await method(*args, **kwargs)
        except (TransportError, AuthRequiredError, ApiError, ValidationError, _InvalidRemoteResponseError) as exc:
            raise _translate_remote_error(exc) from exc

    return wrapper


class HttpSkillRemoteAdapter:
    """Borrow one authenticated SDK connection without owning its lifecycle."""

    def __init__(self, connection: HttpConnection) -> None:
        self._connection = connection

    @property
    def _transport(self):
        return self._connection.transport

    @_translate_remote_errors
    async def push_version(self, request: RemotePushRequest) -> RemotePushResult:
        envelope = await self._transport.post_envelope(
            "/v1/skills/register",
            json=request.model_dump(mode="json", exclude={"cloud_skill_id"}),
        )
        return RemotePushResult.model_validate(_payload(envelope.data))

    @_translate_remote_errors
    async def pull_versions(self, cloud_skill_id: str, cursor: str | None = None) -> RemoteVersionsPage:
        envelope = await self._transport.get_envelope(
            f"/v1/skills/{_path(cloud_skill_id)}/versions",
            params={"cursor": cursor} if cursor else None,
        )
        return RemoteVersionsPage.model_validate(_payload(envelope.data))

    @_translate_remote_errors
    async def pull_content(self, cloud_skill_id: str, version_id: str) -> RemoteVersionContent:
        envelope = await self._transport.get_envelope(
            f"/v1/skills/{_path(cloud_skill_id)}/versions/{_path(version_id)}/content"
        )
        return RemoteVersionContent.model_validate(_payload(envelope.data))

    @_translate_remote_errors
    async def sync(self, request: RemoteSyncRequest) -> RemoteSyncResult:
        envelope = await self._transport.post_envelope("/v1/skills/sync", json=request.model_dump(mode="json"))
        return RemoteSyncResult.model_validate(_payload(envelope.data))

    @_translate_remote_errors
    async def report_trajectories(
        self,
        request: RemoteTrajectoryReportRequest,
    ) -> RemoteTrajectoryReportResult:
        envelope = await self._transport.post_envelope(
            "/v1/skills/trajectory/report",
            json=request.model_dump(mode="json"),
        )
        return RemoteTrajectoryReportResult.model_validate(_payload(envelope.data))

    @_translate_remote_errors
    async def list_trajectories(self, request: RemoteTrajectoryListRequest) -> RemoteTrajectoryPage:
        envelope = await self._transport.post_envelope(
            "/v1/skills/trajectory/list",
            json=request.model_dump(mode="json", exclude_none=True),
        )
        return RemoteTrajectoryPage.model_validate(_payload(envelope.data))

    @_translate_remote_errors
    async def evolve(self, request: RemoteEvolveRequest) -> RemoteEvolveResult:
        envelope = await self._transport.post_envelope(
            "/v1/skills/evolve",
            json=request.model_dump(mode="json"),
        )
        return RemoteEvolveResult.model_validate(_payload(envelope.data))

def _path(value: str) -> str:
    return quote(value, safe="")


def _payload(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _InvalidRemoteResponseError("remote response data must be an object")
    return value


def _translate_remote_error(exc: Exception) -> SkillRemoteRequestError:
    if isinstance(exc, TransportError):
        return SkillRemoteRequestError("Skill remote service is unavailable", error_code="remote_unavailable", retryable=True)
    if isinstance(exc, AuthRequiredError):
        return SkillRemoteRequestError(
            "Skill remote authentication is not configured",
            error_code="remote_auth_required",
            retryable=False,
        )
    if isinstance(exc, ApiError):
        status = exc.status_code
        retryable = status == 429 or (status is not None and status >= 500)
        stable_code = {
            401: "remote_unauthorized",
            403: "remote_forbidden",
            404: "remote_not_found",
            409: "remote_conflict",
            429: "remote_rate_limited",
        }.get(status, "remote_server_error" if status is not None and status >= 500 else None)
        return SkillRemoteRequestError(
            "Skill remote request failed",
            error_code=stable_code or exc.code or "remote_request_failed",
            retryable=retryable,
            status_code=status,
            request_id=exc.request_id,
        )
    return SkillRemoteRequestError(
        "Skill remote returned an invalid response",
        error_code="remote_invalid_response",
        retryable=False,
    )


__all__ = ["HttpSkillRemoteAdapter"]
