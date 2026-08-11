"""Asynchronous public facade over ``mindmemos_skill.SkillApplication``."""

from __future__ import annotations

from collections.abc import Awaitable
from typing import TypeVar

from mindmemos_skill.management import (
    ExportSkillRequest,
    ExportSkillResult,
    ManagedSkill,
    PublishSkillRequest,
    PublishSkillResult,
    PullResult,
    PushResult,
    RegisterSkillRequest,
    RegisterSkillResult,
    SkillDetail,
    SkillDiffResult,
    SkillManagementDetail,
    SkillManagementOverview,
)
from mindmemos_skill.persistence import SkillRecord

from mindmemos_skill import (
    EvolveRunRequest,
    MindMemOSSkillError,
    SkillAlgorithmRunResult,
    SkillApplication,
    Trace2SkillRunRequest,
)

from ..errors import translate_skill_error

_ResultT = TypeVar("_ResultT")


class AsyncSkillClient:
    """Compatibility-stable SDK entry point with no Skill business state of its own."""

    def __init__(self, application: SkillApplication) -> None:
        if not isinstance(application, SkillApplication):
            raise TypeError("application must be a SkillApplication")
        self._application = application

    @property
    def application(self) -> SkillApplication:
        return self._application

    async def register(self, request: RegisterSkillRequest) -> RegisterSkillResult:
        return await self._call(self._application.register(request))

    async def publish(self, request: PublishSkillRequest) -> PublishSkillResult:
        return await self._call(self._application.publish(request))

    async def list_skills(self) -> list[ManagedSkill]:
        return await self._call(self._application.list_skills())

    async def get_management_overview(self) -> SkillManagementOverview:
        return await self._call(self._application.get_management_overview())

    async def get_skill(self, skill_ref: str) -> SkillDetail:
        return await self._call(self._application.get_skill(skill_ref))

    async def get_management_detail(self, skill_ref: str) -> SkillManagementDetail:
        return await self._call(self._application.get_management_detail(skill_ref))

    async def list_versions(self, skill_ref: str) -> list[SkillRecord]:
        return await self._call(self._application.list_versions(skill_ref))

    async def get_version(self, skill_ref: str, version_id: str) -> SkillRecord:
        return await self._call(self._application.get_version(skill_ref, version_id))

    async def export(self, request: ExportSkillRequest) -> ExportSkillResult:
        return await self._call(self._application.export(request))

    async def diff(
        self,
        skill_ref: str,
        *,
        to_version_id: str,
        from_version_id: str | None = None,
    ) -> SkillDiffResult:
        return await self._call(
            self._application.diff(
                skill_ref,
                to_version_id=to_version_id,
                from_version_id=from_version_id,
            )
        )

    async def push(self, skill_ref: str, version_id: str | None = None) -> PushResult:
        return await self._call(self._application.push(skill_ref, version_id))

    async def pull(self, skill_ref: str) -> PullResult:
        return await self._call(self._application.pull(skill_ref))

    async def sync(self, skill_ref: str) -> SkillDetail:
        return await self._call(self._application.sync(skill_ref))

    async def run_trace2skill(self, request: Trace2SkillRunRequest) -> SkillAlgorithmRunResult:
        return await self._call(self._application.run_trace2skill(request))

    async def run_evolve(self, request: EvolveRunRequest) -> SkillAlgorithmRunResult:
        return await self._call(self._application.run_evolve(request))

    @staticmethod
    async def _call(operation: Awaitable[_ResultT]) -> _ResultT:
        try:
            return await operation
        except MindMemOSSkillError as exc:
            raise translate_skill_error(exc) from exc

    async def aclose(self) -> None:
        """Do nothing: ``SDKPortalRuntime`` owns the shared Application lifecycle."""


__all__ = ["AsyncSkillClient"]
