"""Public central repository operations, independent of transport."""

from abc import ABC, abstractmethod
from typing import Any

from ..persistence import SkillRecord, SkillVersionRecord
from ..pipelines.typing import (
    CommitRequest,
    DiffRequest,
    DiffResult,
    ExportRequest,
    HistoryRequest,
    ListRequest,
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


class MindMemOSSkill(ABC):
    """Define the public operations exposed by an assembled Skill service."""

    async def close(self) -> None:
        """Release owned resources; implementations without resources need no cleanup."""

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
        """Explicitly create missing persistence tables; never migrate existing schemas."""
        raise NotImplementedError

    @abstractmethod
    async def register_skill(self, request: RegisterRequest) -> SkillDetail:
        """Register a new central Skill and initial version.

        Args:
            request: Validated operation inputs and explicit local project scope.
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
    async def show_skill(self, request: ShowRequest) -> SkillDetail:
        """Inspect central metadata and an owned version.

        Args:
            request: Validated operation inputs and explicit local project scope.
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
    async def skill_history(self, request: HistoryRequest) -> list[SkillVersionRecord]:
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
    async def switch_skill(self, request: SwitchRequest) -> SkillRecord:
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
    async def auto_evolve_skill(self, request: SkillPipelineRequest | None = None) -> SkillPipelineResult:
        """Invoke the configured evolution pipeline.

        Args:
            request: Evolution selection; required when a pipeline is configured.
        """
        raise NotImplementedError

    @abstractmethod
    async def auto_train_skill(self, request: TrainingRequest | None = None) -> SkillPipelineResult:
        """Invoke the configured training pipeline.

        Args:
            request: Tasks, split membership, and an exact input Skill version.
        """
        raise NotImplementedError
