"""Port implemented by remote Skill transports."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .models import (
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
)


@runtime_checkable
class SkillRemotePort(Protocol):
    async def push_version(self, request: RemotePushRequest) -> RemotePushResult: ...

    async def pull_versions(self, cloud_skill_id: str, cursor: str | None = None) -> RemoteVersionsPage: ...

    async def pull_content(self, cloud_skill_id: str, version_id: str) -> RemoteVersionContent: ...

    async def sync(self, request: RemoteSyncRequest) -> RemoteSyncResult: ...

    async def report_trajectories(
        self,
        request: RemoteTrajectoryReportRequest,
    ) -> RemoteTrajectoryReportResult: ...

    async def list_trajectories(self, request: RemoteTrajectoryListRequest) -> RemoteTrajectoryPage: ...

    async def evolve(self, request: RemoteEvolveRequest) -> RemoteEvolveResult: ...
