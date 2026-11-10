"""Public central repository operations, independent of transport."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from ..infra.task_manager import TaskSnapshot
from ..logging.typing import TaskLog
from ..persistence import (
    EvaluationRecord,
    SkillRecord,
    SkillVersionRecord,
    SkillVersionSummary,
    TrajectoryRecord,
    TrajectorySummary,
)
from ..persistence.db_services.trajectories.typing import TrajectoryStatistics
from ..registry import Registrable
from ..skill.typing import DiffResult
from .typing import (
    ActiveRequest,
    AddGraphMembersRequest,
    AlgoSkillEvolveResult,
    CommitFromLocalRequest,
    CommitRequest,
    DiffRequest,
    EvaluationQueryRequest,
    EvaluationQueryResult,
    EvolutionRequest,
    EvolutionResult,
    ExportRequest,
    HistoryRequest,
    ListRequest,
    ListTrajectoriesRequest,
    OrganizeRequest,
    OrganizeResult,
    RecommendRequest,
    RecommendResult,
    RegisterFromLocalRequest,
    RegisterRequest,
    RegisterResult,
    RemoveGraphMembersRequest,
    ReportTrajectoryRequest,
    SetMetaRequest,
    ShowRequest,
    ShowTrajectoryRequest,
    SkillAnalyzeRequest,
    SkillDetail,
    SkillPage,
    SkillPageRequest,
    SkillRequest,
    TrainingRequest,
    TrajectoryAnalyzeRequest,
    TrajectoryStatisticsRequest,
    VersionResolution,
    VersionResolveRequest,
)


class MindMemOSSkill(ABC):
    """Define the public operations exposed by an assembled Skill service."""

    @abstractmethod
    async def resolve_graph(self, reference: str, project_id: str | None = None) -> dict:
        """Resolve a graph alias, family ID, or version ID.

        Args:
            reference: Graph identity or alias in the authorized project.
            project_id: Explicit project, or the configured default project.
        """
        raise NotImplementedError

    @abstractmethod
    async def delete_graph(self, reference: str, project_id: str | None = None) -> None:
        """Soft-delete a project-owned graph.

        Args:
            reference: Active graph identity or alias.
            project_id: Authorized project, or the configured default project.
        """
        raise NotImplementedError

    @abstractmethod
    async def resolve_skill_reference(self, reference: str, project_id: str | None = None) -> SkillDetail:
        """Resolve a Skill alias, family ID, or version ID.

        Args:
            reference: Skill identity or alias in the authorized project.
            project_id: Explicit project, or the configured default project.
        """
        raise NotImplementedError

    @abstractmethod
    async def list_graph(self, project_id: str | None = None) -> list[dict]:
        """List authorized Skill Sets and their publication pointers.

        Args:
            project_id: Explicit project, or the configured default project when omitted.

        Returns:
            Project-owned Skill Set records.
        """
        raise NotImplementedError

    @abstractmethod
    async def list_graph_snapshots(self, graph_id: str, project_id: str | None = None) -> list[dict]:
        """List one graph's immutable snapshots and publication revisions.

        Args:
            graph_id: Exact project-local graph identity.
            project_id: Explicit project, or the configured default project when omitted.

        Returns:
            Project-owned snapshots and revision numbers.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_graph_snapshot(self, graph_id: str, snapshot_id: str, project_id: str | None = None) -> dict:
        """Read one immutable graph snapshot.

        Args:
            graph_id: Exact project-local graph identity.
            snapshot_id: Immutable snapshot identity.
            project_id: Explicit project, or the configured default project when omitted.

        Returns:
            Snapshot members and relations for its representation mode.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_skill_version(self, version_id: str, project_id: str | None = None) -> SkillVersionRecord:
        """Read an exact version in the authorized project.

        Args:
            version_id: Exact immutable Skill version identity.
            project_id: Explicit project, or the configured default project when omitted.

        Returns:
            The project-owned version record.
        """
        raise NotImplementedError

    @abstractmethod
    async def list_skill_versions(self, skill_id: str, project_id: str | None = None) -> list[SkillVersionRecord]:
        """Read the complete history of an exact Skill family.

        Args:
            skill_id: Exact project-local Skill family identity.
            project_id: Explicit project, or the configured default project when omitted.

        Returns:
            Project-owned versions in creation order.
        """
        raise NotImplementedError

    async def organize_skills(self, request: OrganizeRequest) -> OrganizeResult:
        """Publish a graph from exact project-owned Skill versions.

        Args:
            request: Authorized graph identity, actor, version scope, and scenes.

        Returns:
            Immutable graph snapshot identity.
        """
        raise NotImplementedError

    async def recommend_skills(self, request: RecommendRequest) -> RecommendResult:
        """Recommend exact versions from one pinned graph snapshot.

        Args:
            request: Authorized graph snapshot and public task.

        Returns:
            Ordered exact Skill version identities.
        """
        raise NotImplementedError

    @abstractmethod
    async def analyze_skill(self, request: SkillAnalyzeRequest) -> EvaluationRecord:
        """Run or reuse static analysis for one exact Skill version.

        Args:
            request: Authorized project and exact immutable version.

        Returns:
            Stored completed or failed evaluation result.

        Raises:
            NotImplementedError: Analysis is not configured.
        """
        raise NotImplementedError

    @abstractmethod
    async def analyze_trajectory(self, request: TrajectoryAnalyzeRequest) -> EvaluationRecord:
        """Run or reuse analysis for one exact trajectory.

        Args:
            request: Authorized project, trajectory, and optional overrides.

        Returns:
            Stored completed or failed evaluation result.

        Raises:
            NotImplementedError: Analysis is not configured.
        """
        raise NotImplementedError

    @abstractmethod
    async def query_evaluations(self, request: EvaluationQueryRequest) -> EvaluationQueryResult:
        """Read completed analysis rows for exact project-owned versions.

        Args:
            request: Authorized project, exact versions, and optional filters.

        Returns:
            Completed stored evaluations in requested version order.

        Raises:
            NotImplementedError: Analysis is not configured.
        """
        raise NotImplementedError

    async def add_graph_members(self, request: AddGraphMembersRequest) -> OrganizeResult:
        """Add new families to an existing graph.

        Args:
            request: Authorized graph reference and nonempty member batch.

        Returns:
            Published graph snapshot and its complete member versions.
        """
        raise NotImplementedError

    async def remove_graph_members(self, request: RemoveGraphMembersRequest) -> OrganizeResult:
        """Remove families while preserving canonical Skills and graph history.

        Args:
            request: Authorized graph reference and nonempty member batch.

        Returns:
            Published graph snapshot and remaining member versions.
        """
        raise NotImplementedError

    async def preview_graph(self, graph_ref: str, project_id: str | None = None) -> dict:
        """Read the complete published topology selected by a graph reference.

        Args:
            graph_ref: Graph family, alias, or immutable snapshot reference.
            project_id: Authorized project or configured default.

        Returns:
            Snapshot identity, Skill/Scene nodes, and their relations.
        """
        raise NotImplementedError

    async def close(self) -> None:
        """Release owned resources; implementations without resources need no cleanup."""

    async def submit_training(self, request: TrainingRequest) -> TaskSnapshot:
        """Submit an asynchronous training task when configured.

        Args:
            request: Validated training input.

        Returns:
            Accepted task state.
        """
        raise NotImplementedError

    async def submit_evolution(
        self,
        project_id: str | None,
        skill_version_ids: list[str],
        *,
        pipeline: str | None = None,
        auto_accept: bool = True,
        editable_skills: list[str] | None = None,
        allow_create_skills: bool = True,
    ) -> TaskSnapshot:
        """Submit evolution from exact stored Skill versions.

        Args:
            project_id: Authorized project or local bound scope.
            skill_version_ids: Ordered list of one or more exact version identities.
            pipeline: Configured evolution pipeline name; omission requires one pipeline.
            auto_accept: Whether to select the evolved version as active.
            editable_skills: Editable input member IDs; None permits all input members.
            allow_create_skills: Whether collection evolution may create independent Skills.

        Returns:
            Accepted task state.
        """
        raise NotImplementedError

    async def task_status(self, project_id: str | None, task_id: str) -> TaskSnapshot | None:
        """Read a project-owned asynchronous task.

        Args:
            project_id: Authorized project or local bound scope.
            task_id: Server-generated task identity.

        Returns:
            State, or None for an unknown task.
        """
        raise NotImplementedError

    async def task_progress(self, project_id: str | None, task_id: str, *, after: int = 0) -> list[TaskLog]:
        """Read task logs used for progress and failure diagnostics.

        Args:
            project_id: Authorized project or local bound scope.
            task_id: Server-generated task identity.
            after: Last observed committed log sequence; gaps are allowed.

        Returns:
            Later task logs, including structured body and traceback.
        """
        raise NotImplementedError

    async def __aenter__(self) -> "MindMemOSSkill":
        """Return the service for scoped asynchronous resource ownership."""
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        """Release owned resources when leaving the asynchronous context.

        Args:
            exc_type: Exception type raised inside the context, if any.
            exc: Exception raised inside the context, if any.
            traceback: Traceback associated with that exception.
        """
        await self.close()

    @property
    @abstractmethod
    def backends(self) -> dict[str, Registrable[Any]]:
        """Return assembled backend instances keyed by configuration name."""
        raise NotImplementedError

    @abstractmethod
    async def initialize_storage(self) -> None:
        """Idempotently initialize configured stores; also called by create_service."""
        raise NotImplementedError

    @abstractmethod
    async def register_skill(self, request: RegisterRequest) -> RegisterResult:
        """Register a new central Skill and initial version.

        Args:
            request: Validated operation inputs and explicit local project scope.
        """
        raise NotImplementedError

    @abstractmethod
    async def register_from_local(self, request: RegisterFromLocalRequest) -> RegisterResult:
        """Register a standard local Skill through the structured write pipeline.

        Args:
            request: Authorized project, local source, and alias.

        Returns:
            Family and initial version identity with both source hashes.
        """
        raise NotImplementedError

    @abstractmethod
    async def commit_from_local(self, request: CommitFromLocalRequest) -> SkillDetail:
        """Commit a standard local Skill through the structured write pipeline.

        Args:
            request: Authorized family, local source, and publication options.

        Returns:
            Family and exact newly persisted version.
        """
        raise NotImplementedError

    @abstractmethod
    async def list_skills(self, request: ListRequest) -> list[SkillRecord]:
        """List central Skills within a project.

        Args:
            request: Validated operation inputs and explicit local project scope.
        """
        raise NotImplementedError

    @abstractmethod
    async def list_skill_page(self, request: SkillPageRequest) -> SkillPage:
        """Filter a stable family page across repository and trajectory evidence.

        Args:
            request: Authorized project, filters, snapshot cutoff, and page position.

        Returns:
            Matching families with a lookahead indicator.
        """
        raise NotImplementedError

    @abstractmethod
    async def report_trajectory(self, request: ReportTrajectoryRequest) -> TrajectoryRecord:
        """Store one Harness execution in the shared trajectory repository.

        Args:
            request: Authorized project and validated execution details.
        """
        raise NotImplementedError

    @abstractmethod
    async def trajectory_statistics(self, request: TrajectoryStatisticsRequest) -> TrajectoryStatistics:
        """Read aggregates without exposing individual trajectories.

        Args:
            request: Authorized project, owning family and exact version.

        Returns:
            Global counts, fixed thirty-day series and latest evolution time.
        """
        raise NotImplementedError

    @abstractmethod
    async def show_trajectory(self, request: ShowTrajectoryRequest) -> TrajectoryRecord:
        """Read one owned local or reported trajectory.

        Args:
            request: Project and exact trajectory identity.
        """
        raise NotImplementedError

    @abstractmethod
    async def list_trajectories(self, request: ListTrajectoriesRequest) -> list[TrajectorySummary]:
        """Read a receipt-ordered page of reported trajectories.

        Args:
            request: Project, exact family, filters, and cursor position.
        """
        raise NotImplementedError

    @abstractmethod
    async def skill_has_reported_trajectory(
        self,
        project_id: str,
        skill_id: str,
        *,
        user_id: str | None,
        agent_id: str | None,
        snapshot_at: datetime,
    ) -> bool:
        """Check whether one family has a report matching the actor filters.

        Args:
            project_id: Authorized project identity.
            skill_id: Exact family identity.
            user_id: Optional business user identity.
            agent_id: Optional Agent identity.
            snapshot_at: First-page receipt cutoff.
        """
        raise NotImplementedError

    @abstractmethod
    async def show_skill(self, request: ShowRequest) -> SkillDetail:
        """Inspect central metadata and an owned version.

        Args:
            request: Validated operation inputs and explicit local project scope.
        """
        raise NotImplementedError

    @abstractmethod
    async def resolve_skill_version(self, request: VersionResolveRequest) -> VersionResolution:
        """Resolve one exact source version within an authorized project.

        Args:
            request: Source hash and optional exact family and version identities.
        """
        raise NotImplementedError

    @abstractmethod
    async def export_skill(self, request: ExportRequest) -> str:
        """Export a complete snapshot into an application directory.

        Args:
            request: Validated operation inputs and explicit local project scope.
        """
        raise NotImplementedError

    @abstractmethod
    async def commit_skill(self, request: CommitRequest) -> SkillDetail:
        """Save and select a child version with conflict detection.

        Args:
            request: Validated operation inputs and explicit local project scope.
        """
        raise NotImplementedError

    @abstractmethod
    async def skill_history(self, request: HistoryRequest) -> list[SkillVersionSummary]:
        """Read immutable version history.

        Args:
            request: Validated operation inputs and explicit local project scope.
        """
        raise NotImplementedError

    @abstractmethod
    async def diff_skill(self, request: DiffRequest) -> DiffResult:
        """Compare two versions of the same Skill.

        Args:
            request: Validated operation inputs and explicit local project scope.
        """
        raise NotImplementedError

    @abstractmethod
    async def active_skill(self, request: ActiveRequest) -> SkillRecord:
        """Select an existing version without modifying external files.

        Args:
            request: Validated operation inputs and explicit local project scope.
        """
        raise NotImplementedError

    @abstractmethod
    async def set_skill_meta(self, request: SetMetaRequest) -> SkillRecord:
        """Change display metadata without editing content.

        Args:
            request: Validated operation inputs and explicit local project scope.
        """
        raise NotImplementedError

    @abstractmethod
    async def delete_skill(self, request: SkillRequest) -> None:
        """Delete an unreferenced Skill and its version history.

        Args:
            request: Validated operation inputs and explicit local project scope.
        """
        raise NotImplementedError

    @abstractmethod
    async def run_evolution(self, request: EvolutionRequest | None = None) -> EvolutionResult:
        """Run the configured evolution pipeline and return its result.

        Args:
            request: Evolution selection; required when a pipeline is configured.

        Returns:
            Completed evolution result.
        """
        raise NotImplementedError

    @abstractmethod
    async def run_training(self, request: TrainingRequest | None = None) -> AlgoSkillEvolveResult:
        """Run the configured training pipeline and return its result.

        Args:
            request: Tasks, split membership, and an exact input Skill version.

        Returns:
            Completed training result.
        """
        raise NotImplementedError
