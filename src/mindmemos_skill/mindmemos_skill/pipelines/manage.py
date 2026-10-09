"""Shared contract for Skill Set management pipelines."""

from abc import ABC, abstractmethod
from uuid import UUID

from ..api.typing import OrganizeRequest, OrganizeResult, RecommendRequest, RecommendResult
from ..infra.task_manager.progress import ProgressReporter
from ..registry.base import ConfigT
from .base import Pipeline


class ManagePipeline(Pipeline[ConfigT], ABC):
    """Organize, recommend, and query project-owned Skill Sets."""

    @abstractmethod
    async def resolve_graph(self, project_id: str, reference: str) -> dict:
        """Resolve a project-owned graph family or immutable snapshot.

        Args:
            project_id: Authorized project identity.
            reference: Alias or unique prefixed graph identity.

        Returns:
            Resolved public graph identity and selected snapshot.
        """

    @abstractmethod
    async def organize(self, request: OrganizeRequest, *, reporter: ProgressReporter | None = None) -> OrganizeResult:
        """Publish one Skill Set snapshot.

        Args:
            request: Authorized graph identity and exact Skill versions.
            reporter: Optional progress sink for graph publication.

        Returns:
            Published snapshot identity.
        """

    @abstractmethod
    async def recommend(self, request: RecommendRequest) -> RecommendResult:
        """Recommend versions from one pinned Skill Set snapshot.

        Args:
            request: Authorized graph snapshot and public task.

        Returns:
            Ordered exact Skill version identities.
        """

    @abstractmethod
    async def list_skill_sets(self, project_id: str) -> list[dict]:
        """List project-owned Skill Sets.

        Args:
            project_id: Authorized project identity.

        Returns:
            Skill Set records and publication pointers.
        """

    @abstractmethod
    async def list_skill_set_snapshots(self, project_id: str, skill_set_id: str) -> list[dict]:
        """List immutable snapshots of one Skill Set.

        Args:
            project_id: Authorized project identity.
            skill_set_id: Exact project-local Skill Set identity.

        Returns:
            Snapshot records and revision numbers.
        """

    @abstractmethod
    async def get_skill_set_snapshot(self, project_id: str, skill_set_id: str, snapshot_id: UUID) -> dict:
        """Read one immutable Skill Set snapshot.

        Args:
            project_id: Authorized project identity.
            skill_set_id: Exact project-local Skill Set identity.
            snapshot_id: Immutable snapshot identity.

        Returns:
            Snapshot members and relations.
        """
