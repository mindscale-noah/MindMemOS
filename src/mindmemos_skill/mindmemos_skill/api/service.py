"""Transport-independent facade delegating business operations to pipelines."""

from typing import Any, TypeVar

from ..infra.database import Database
from ..persistence import SkillRecord, SkillVersionRecord
from ..pipelines import EvolutionPipeline, TrainingPipeline
from ..pipelines.graphs import GraphQueryPipeline
from ..pipelines.repository import RepositoryPipeline
from ..pipelines.typing import (
    CommitRequest,
    DiffRequest,
    DiffResult,
    ExportRequest,
    HistoryRequest,
    ListRequest,
    ProjectRequest,
    RegisterRequest,
    SetMetaRequest,
    ShowRequest,
    SkillDetail,
    SkillPipelineRequest,
    SkillPipelineResult,
    SkillRequest,
    SwitchRequest,
    TrainingRequest,
)
from ..registry import Registrable
from .base import MindMemOSSkill

RequestT = TypeVar("RequestT", bound=ProjectRequest)


class SkillService(MindMemOSSkill):
    """Expose injected pipelines and own the assembled backend lifecycle.

    Args:
        backends: Already constructed dependencies, ordered by construction.
        repository: Selected repository operations, or None when unconfigured.
        training: Selected training pipeline, or None when unconfigured.
        evolution: Selected evolution pipeline, or None when unconfigured.
        local_project_id: Scope bound only by explicit local assembly.
    """

    def __init__(
        self,
        *,
        backends: dict[str, Registrable[Any]],
        repository: RepositoryPipeline | None = None,
        training: TrainingPipeline | None = None,
        evolution: EvolutionPipeline | None = None,
        local_project_id: str | None = None,
    ) -> None:
        self._backends = backends.copy()
        self._repository = repository
        self._training = training
        self._evolution = evolution
        self.local_project_id = local_project_id

    @property
    def backends(self) -> dict[str, Registrable[Any]]:
        """Return assembled backend instances for diagnostics and lifecycle inspection."""
        return self._backends.copy()

    def close(self) -> None:
        """Close owned database pools in reverse construction order."""
        for backend in reversed(list(self._backends.values())):
            if isinstance(backend, Database):
                backend.close()

    @property
    def repository(self) -> RepositoryPipeline:
        """Return the selected repository pipeline, never a raw database."""
        if self._repository is None:
            raise NotImplementedError("Repository pipeline is not configured")
        return self._repository

    def initialize_storage(self) -> None:
        """Delegate explicit missing-table initialization to the repository pipeline."""
        self.repository.initialize_storage()

    def _project_id(self, requested: str | None) -> str:
        """Require explicit scope unless this service was assembled for local use."""
        if self.local_project_id is not None:
            if requested is not None and requested != self.local_project_id:
                raise ValueError("The requested project differs from the bound local scope")
            return self.local_project_id
        if requested is None:
            raise ValueError("project_id is required outside local service assembly")
        return requested

    def _scoped(self, request: RequestT) -> RequestT:
        """Bind request scope without mutating the caller's model."""
        return request.model_copy(update={"project_id": self._project_id(request.project_id)})

    def graph_queries(self, project_id: str | None = None) -> GraphQueryPipeline:
        """Create a project-scoped graph query pipeline over shared persistence.

        Args:
            project_id: Explicit project, or the bound local scope when omitted.
        """
        return self.repository.graph_queries(self._project_id(project_id))

    def register_skill(self, request: RegisterRequest) -> SkillDetail:
        """Register a complete source under a new central identity.

        Args:
            request: Project, source content, display metadata, and provenance.
        """
        request = self._scoped(RegisterRequest.model_validate(request))
        return self.repository.register_skill(request)

    def list_skills(self, request: ListRequest) -> list[SkillRecord]:
        """List central records in one project.

        Args:
            request: Project and pagination settings.
        """
        request = self._scoped(ListRequest.model_validate(request))
        return self.repository.list_skills(request)

    def show_skill(self, request: ShowRequest) -> SkillDetail:
        """Read central metadata and an owned version.

        Args:
            request: Project, Skill reference, and optional version.
        """
        request = self._scoped(ShowRequest.model_validate(request))
        return self.repository.show_skill(request)

    def export_skill(self, request: ExportRequest) -> str:
        """Export a complete stored snapshot without tracking the directory.

        Args:
            request: Skill/version selection and explicit destination replacement policy.

        Returns:
            Absolute path of the exported application directory.
        """
        request = self._scoped(ExportRequest.model_validate(request))
        return self.repository.export_skill(request)

    def commit_skill(self, request: CommitRequest) -> SkillDetail:
        """Save a new version and select it only if the captured parent is still current.

        Args:
            request: Target Skill, source, optional expected parent, and commit message.
        """
        request = self._scoped(CommitRequest.model_validate(request))
        return self.repository.commit_skill(request)

    def skill_history(self, request: HistoryRequest) -> list[SkillVersionRecord]:
        """Read a page of versions without changing the current selection.

        Args:
            request: Project, Skill reference, and pagination.
        """
        request = self._scoped(HistoryRequest.model_validate(request))
        return self.repository.skill_history(request)

    def diff_skill(self, request: DiffRequest) -> DiffResult:
        """Compare complete snapshots from the same Skill.

        Args:
            request: Skill and version pair; the starting version defaults to current.
        """
        request = self._scoped(DiffRequest.model_validate(request))
        return self.repository.diff_skill(request)

    def switch_skill(self, request: SwitchRequest) -> SkillRecord:
        """Change the current pointer without editing any application directory.

        Args:
            request: Skill and existing owned version to select.
        """
        request = self._scoped(SwitchRequest.model_validate(request))
        return self.repository.switch_skill(request)

    def set_skill_meta(self, request: SetMetaRequest) -> SkillRecord:
        """Update name or alias without creating a version.

        Args:
            request: Skill reference and explicit metadata changes.
        """
        request = self._scoped(SetMetaRequest.model_validate(request))
        return self.repository.set_skill_meta(request)

    def delete_skill(self, request: SkillRequest) -> None:
        """Delete central data only when no retained evidence references it.

        Args:
            request: Project and Skill identity, alias, or unique prefix.
        """
        request = self._scoped(SkillRequest.model_validate(request))
        return self.repository.delete_skill(request)

    def auto_evolve_skill(self, request: SkillPipelineRequest | None = None) -> SkillPipelineResult:
        """Delegate evolution to the selected pipeline.

        Args:
            request: Skill/version selection; required when a pipeline is configured.

        Raises:
            NotImplementedError: No evolution pipeline is configured.
            ValueError: A configured operation received no request.
        """
        if self._evolution is None:
            raise NotImplementedError("Automatic Skill evolution is not implemented")
        if request is None:
            raise ValueError("An evolution request is required")
        request = SkillPipelineRequest.model_validate(request)
        scoped = request.model_copy(update={"project_id": self._project_id(request.project_id)})
        return self._evolution.evolve(scoped)

    def auto_train_skill(self, request: TrainingRequest | None = None) -> SkillPipelineResult:
        """Delegate tasks, splits, and an exact input version to the training pipeline.

        Args:
            request: Training inputs; required when a pipeline is configured.

        Raises:
            NotImplementedError: No training pipeline is configured.
            ValueError: A configured operation received no request.
        """
        if self._training is None:
            raise NotImplementedError("Automatic Skill training is not implemented")
        if request is None:
            raise ValueError("A training request is required")
        request = TrainingRequest.model_validate(request)
        scoped = request.model_copy(update={"project_id": self._project_id(request.project_id)})
        return self._training.train(scoped)
