"""Asynchronous client for every public standalone Skill service endpoint."""

from __future__ import annotations

from ..transport import AsyncHttpTransport
from .core import SkillCore, SkillRoute
from .typing import (
    ActivateRequest,
    CommitRequest,
    DeleteRequest,
    DiffRequest,
    EvolveRequest,
    GraphResolveRequest,
    GraphSnapshotRequest,
    GraphSnapshotsRequest,
    HistoryRequest,
    ListRequest,
    OrganizeRequest,
    RecommendRequest,
    RegisterRequest,
    ShowRequest,
    SkillData,
    SkillRequestBody,
    SkillResolveRequest,
    TaskProgressRequest,
    TaskShowRequest,
    TrainingRequest,
    TrajectoryListRequest,
    TrajectoryReportRequest,
    TrajectoryShowRequest,
    TrajectoryStatisticsRequest,
    UpdateInfoRequest,
    VersionResolveRequest,
)


class AsyncSkillClient:
    """Call the Skill service without local registration or upload state."""

    def __init__(self, transport: AsyncHttpTransport) -> None:
        """Use an existing asynchronous HTTP transport.

        Args:
            transport: Transport configured for the Skill service URL and credentials.
        """
        self._transport = transport

    async def _post(self, route: SkillRoute, body: SkillRequestBody) -> SkillData:
        request = SkillCore.prepare(route, body)
        envelope = await self._transport.post_envelope(request.path, json=request.body, params=request.params)
        return envelope.data if envelope.data is not None else {}

    async def register(self, request: RegisterRequest) -> SkillData:
        """Register a family from a complete content and resources.

        Args:
            request: Family name, alias, initial version, and source files.

        Returns:
            New family and version identities.
        """
        return await self._post(SkillRoute.REGISTER, request)

    async def list(self, request: ListRequest) -> SkillData:
        """List families in a cursor page.

        Args:
            request: Optional filters and pagination cursor.

        Returns:
            Items and next cursor from the service.
        """
        return await self._post(SkillRoute.LIST, request)

    async def show(self, request: ShowRequest) -> SkillData:
        """Read a family and a complete selected version.

        Args:
            request: Family identity and optional historical version.

        Returns:
            Family metadata and portable version content.
        """
        return await self._post(SkillRoute.SHOW, request)

    async def commit(self, request: CommitRequest) -> SkillData:
        """Commit a new immutable source version.

        Args:
            request: Family, complete content and resources, version label, and activation choice.

        Returns:
            New version identity and active-version state.
        """
        return await self._post(SkillRoute.COMMIT, request)

    async def history(self, request: HistoryRequest) -> SkillData:
        """Page one family's version history.

        Args:
            request: Family identity and offset pagination.

        Returns:
            Version summaries and pagination metadata.
        """
        return await self._post(SkillRoute.HISTORY, request)

    async def diff(self, request: DiffRequest) -> SkillData:
        """Compare two stored source versions.

        Args:
            request: Source and target family/version selection.

        Returns:
            Per-file and implementation differences.
        """
        return await self._post(SkillRoute.DIFF, request)

    async def activate(self, request: ActivateRequest) -> SkillData:
        """Select an existing version as active.

        Args:
            request: Family and owned version identities.

        Returns:
            Selected active version identity.
        """
        return await self._post(SkillRoute.ACTIVATE, request)

    async def update_info(self, request: UpdateInfoRequest) -> SkillData:
        """Update mutable family fields.

        Args:
            request: Family identity and fields to replace.

        Returns:
            Updated family metadata.
        """
        return await self._post(SkillRoute.UPDATE_INFO, request)

    async def delete(self, request: DeleteRequest) -> SkillData:
        """Delete an unreferenced family.

        Args:
            request: Exact family identity.

        Returns:
            Deletion confirmation from the service.
        """
        return await self._post(SkillRoute.DELETE, request)

    async def version_resolve(self, request: VersionResolveRequest) -> SkillData:
        """Resolve a canonical source hash to an owned version.

        Args:
            request: Hash and optional exact family/version selectors.

        Returns:
            Resolved version information.
        """
        return await self._post(SkillRoute.VERSION_RESOLVE, request)

    async def skill_resolve(self, request: SkillResolveRequest) -> SkillData:
        """Resolve a family alias or immutable Skill version.

        Args:
            request: Skill identity or alias in the configured project.

        Returns:
            Selected family and version information.
        """
        return await self._post(SkillRoute.SKILL_RESOLVE, request)

    async def report_trajectory(self, request: TrajectoryReportRequest) -> SkillData:
        """Report an execution against its loaded Skill version.

        Args:
            request: Actor, messages, version, and execution timing.

        Returns:
            Stored trajectory identity.
        """
        return await self._post(SkillRoute.TRAJECTORY_REPORT, request)

    async def trajectory_statistics(self, request: TrajectoryStatisticsRequest) -> SkillData:
        """Read bounded aggregate counts for one Skill version.

        Args:
            request: Owning family and exact version identity.

        Returns:
            Global totals, thirty calendar buckets and latest evolution time.
        """
        return await self._post(SkillRoute.TRAJECTORY_STATISTICS, request)

    async def show_trajectory(self, request: TrajectoryShowRequest) -> SkillData:
        """Read one stored trajectory.

        Args:
            request: Exact trajectory identity.

        Returns:
            Complete stored execution record.
        """
        return await self._post(SkillRoute.TRAJECTORY_SHOW, request)

    async def list_trajectories(self, request: TrajectoryListRequest) -> SkillData:
        """Page trajectories of one family.

        Args:
            request: Family, optional filters, and cursor.

        Returns:
            Trajectory summaries and next cursor.
        """
        return await self._post(SkillRoute.TRAJECTORY_LIST, request)

    async def organize(self, request: OrganizeRequest) -> SkillData:
        """Request Skill graph organization.

        Args:
            request: Graph alias, message, and optional exact versions.

        Returns:
            Service task data when the pipeline is available.
        """
        return await self._post(SkillRoute.ORGANIZE, request)

    async def graph_resolve(self, request: GraphResolveRequest) -> SkillData:
        """Resolve a graph reference.

        Args:
            request: Graph ID, alias, or prefix.

        Returns:
            Resolved graph information when available.
        """
        return await self._post(SkillRoute.GRAPH_RESOLVE, request)

    async def delete_graph(self, request: GraphResolveRequest) -> SkillData:
        """Soft-delete one project-owned graph.

        Args:
            request: Graph ID, alias, or unique prefix.

        Returns:
            Deletion confirmation.
        """
        return await self._post(SkillRoute.GRAPH_DELETE, request)

    async def list_graphs(self) -> SkillData:
        """List graphs owned by the authenticated project.

        Returns:
            Project-owned graph records.
        """
        return await self._post(SkillRoute.GRAPHS_LIST, SkillRequestBody())

    async def graph_snapshots(self, request: GraphSnapshotsRequest) -> SkillData:
        """List immutable snapshots of one graph.

        Args:
            request: Exact project-owned graph identity.

        Returns:
            Graph snapshot records and publication revisions.
        """
        return await self._post(SkillRoute.GRAPHS_SNAPSHOTS, request)

    async def graph_snapshot(self, request: GraphSnapshotRequest) -> SkillData:
        """Read one graph's Skill nodes and edges.

        Args:
            request: Exact graph and snapshot identities.

        Returns:
            Resolved nodes, edges, and snapshot identity.
        """
        return await self._post(SkillRoute.GRAPHS_SNAPSHOT, request)

    async def recommend(self, request: RecommendRequest) -> SkillData:
        """Request recommendations from a published graph.

        Args:
            request: Graph reference, query, and optional result limit.

        Returns:
            Recommended versions when the pipeline is available.
        """
        return await self._post(SkillRoute.RECOMMEND, request)

    async def evolve(self, request: EvolveRequest) -> SkillData:
        """Request evolution from an exact version.

        Args:
            request: Baseline version identity.

        Returns:
            Accepted task data when the pipeline is available.
        """
        return await self._post(SkillRoute.EVOLVE, request)

    async def show_task(self, request: TaskShowRequest) -> SkillData:
        """Read one asynchronous Skill task.

        Args:
            request: Exact task identity.

        Returns:
            Task state when task storage is available.
        """
        return await self._post(SkillRoute.TASK_SHOW, request)

    async def train(self, request: TrainingRequest) -> SkillData:
        """Submit a background training run.

        Args:
            request: Complete tasks, split, input version, and server run directory.

        Returns:
            Accepted task identity and state.
        """
        return await self._post(SkillRoute.TRAIN, request)

    async def task_progress(self, request: TaskProgressRequest) -> SkillData:
        """Read task logs after a sequence number.

        Args:
            request: Task identity and exclusive log sequence cursor.

        Returns:
            Ordered progress event records, including an empty list when no new events exist.
        """
        return await self._post(SkillRoute.TASK_PROGRESS, request)
