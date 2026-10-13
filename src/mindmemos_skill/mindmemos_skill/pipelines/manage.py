"""Shared contract for graph management pipelines."""

from abc import ABC, abstractmethod

from ..api.typing import (
    AddGraphMembersRequest,
    OrganizeRequest,
    OrganizeResult,
    RecommendRequest,
    RecommendResult,
    RemoveGraphMembersRequest,
)
from ..persistence.db_services.graphs.typing import GraphResolution
from ..registry.base import ConfigT
from .base import Pipeline
from .typing import GraphListItem, GraphSnapshotListItem, GraphTopology


class ManagePipeline(Pipeline[ConfigT], ABC):
    """Organize, recommend, and query project-owned graphs."""

    @abstractmethod
    async def resolve_graph(self, reference: str) -> GraphResolution:
        """Resolve a project-owned graph family or immutable snapshot.

        Args:
            reference: Alias or unique prefixed graph identity.

        Returns:
            Resolved public graph identity and selected snapshot.
        """

    @abstractmethod
    async def organize(self, request: OrganizeRequest) -> OrganizeResult:
        """Publish one graph snapshot.

        Args:
            request: Authorized graph identity and exact Skill versions.

        Returns:
            Published snapshot identity.
        """

    @abstractmethod
    async def recommend(self, request: RecommendRequest) -> RecommendResult:
        """Recommend versions from one pinned graph snapshot.

        Args:
            request: Authorized graph snapshot and public task.

        Returns:
            Ordered exact Skill version identities.
        """

    @abstractmethod
    async def list_graph(self) -> list[GraphListItem]:
        """List project-owned graphs.

        Returns:
            Graph records and publication pointers.
        """

    @abstractmethod
    async def list_graph_snapshots(self, graph_id: str) -> list[GraphSnapshotListItem]:
        """List immutable snapshots of one graph.

        Args:
            graph_id: Exact project-local graph identity.

        Returns:
            Snapshot records and immutable identities.
        """

    @abstractmethod
    async def get_graph_snapshot(self, graph_id: str, snapshot_id: str) -> GraphTopology:
        """Read one immutable graph snapshot.

        Args:
            graph_id: Exact project-local graph identity.
            snapshot_id: Immutable snapshot identity.

        Returns:
            Snapshot members and relations.
        """

    async def add_graph_members(self, request: AddGraphMembersRequest) -> OrganizeResult:
        """Add new members through the configured graph algorithm.

        Args:
            request: Authorized graph and member references.

        Returns:
            Published snapshot and final member versions.
        """
        raise NotImplementedError("Graph member addition is not implemented")

    async def remove_graph_members(self, request: RemoveGraphMembersRequest) -> OrganizeResult:
        """Remove members through the configured graph algorithm.

        Args:
            request: Authorized graph and member references.

        Returns:
            Published snapshot and remaining member versions.
        """
        raise NotImplementedError("Graph member removal is not implemented")

    async def preview_graph(self, graph_ref: str) -> GraphTopology:
        """Resolve and read an immutable published topology.

        Args:
            graph_ref: Graph family or snapshot reference.

        Returns:
            Selected topology without construction side effects.
        """
        raise NotImplementedError("Graph preview is not implemented")
