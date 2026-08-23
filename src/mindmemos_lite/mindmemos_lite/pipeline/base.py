"""Transport-neutral pipeline contracts.

This module contains only construction and execution interfaces.  Concrete
algorithm implementations live outside this module and are expected to be
registered with :mod:`mindmemos_lite.pipeline.registry`.

Pipeline construction deliberately has no legacy split database dependencies.
Implementations may accept the persistence and orchestration dependencies they need through
``from_config(config, **kwargs)`` while the operation contracts below remain
small and independent of a particular storage backend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Protocol, Self

from ..service.schema import (
    DeleteMemoryRequest,
    EvolveSkillRequest,
    GetMemoryRequest,
    MemoryListResult,
    MemoryMutationResult,
    RequestContext,
    SkillEvolveResult,
    UpdateMemoryRequest,
)
from ..typing import (
    AddPipelineInput,
    AddPipelineSyncResult,
    DreamingPipelineInput,
    DreamingPipelineResult,
    FeedbackPipelineInput,
    FeedbackPipelineResult,
    MemoryRequestContext,
    SearchPipelineInput,
    SearchPipelineResult,
)

if TYPE_CHECKING:
    from ..persistence.memory import MemoryPersistence


class PipelineBase:
    """Common construction contract for every registered pipeline class.

    ``config`` is the effective configuration for the current invocation.  It
    may be the static process config or a project-scoped override.  The base
    implementation keeps simple algorithms compatible with a normal
    constructor; algorithms that need configuration should override this
    classmethod.
    """

    requires_task: ClassVar[bool] = False
    """Whether an add pipeline needs a non-empty ``task`` input (e.g. trajectory_add)."""

    @classmethod
    def from_config(cls, config: Any, **kwargs: Any) -> Self:
        """Construct an implementation from the effective configuration."""

        del config
        return cls(**kwargs)


class MemoryPersistencePipelineMixin(PipelineBase):
    """Provide memory persistence to synchronous algorithm pipelines."""

    def __init__(
        self,
        *,
        persistence: "MemoryPersistence",
    ) -> None:
        if persistence is None:
            raise TypeError("vanilla pipelines require a MemoryPersistence dependency")
        self.persistence = persistence


class AddPipeline(Protocol):
    """Memory ingestion pipeline contract."""

    async def add_sync(
        self,
        inp: AddPipelineInput,
        context: MemoryRequestContext,
    ) -> AddPipelineSyncResult: ...


class SearchPipeline(Protocol):
    """Top-level memory search pipeline contract."""

    async def search(
        self,
        request: SearchPipelineInput,
        context: MemoryRequestContext,
    ) -> SearchPipelineResult:
        """Search memories using the request's selected strategy."""


class GetPipeline(Protocol):
    """Memory read pipeline contract."""

    async def get(
        self,
        request: GetMemoryRequest,
        context: RequestContext,
    ) -> MemoryListResult:
        """Read project-scoped memories."""


class DeletePipeline(Protocol):
    """Memory deletion pipeline contract."""

    async def delete(
        self,
        request: DeleteMemoryRequest,
        context: RequestContext,
    ) -> MemoryMutationResult:
        """Delete or archive the requested memory."""


class UpdatePipeline(Protocol):
    """Memory update pipeline contract."""

    async def update(
        self,
        request: UpdateMemoryRequest,
        context: RequestContext,
    ) -> MemoryMutationResult:
        """Update the requested memory."""


class FeedbackPipeline(Protocol):
    """Feedback pipeline contract, including sync and async entry points."""

    async def feedback(
        self,
        request: FeedbackPipelineInput,
        context: MemoryRequestContext,
    ) -> FeedbackPipelineResult:
        """Dispatch feedback according to ``request.mode``."""

    async def feedback_sync(
        self,
        request: FeedbackPipelineInput,
        context: MemoryRequestContext,
    ) -> FeedbackPipelineResult:
        """Apply feedback immediately."""

    async def feedback_async(
        self,
        request: FeedbackPipelineInput,
        context: MemoryRequestContext,
    ) -> FeedbackPipelineResult:
        """Submit feedback for asynchronous processing."""


class DreamingPipeline(Protocol):
    """Memory consolidation pipeline contract."""

    async def dream(
        self,
        request: DreamingPipelineInput,
        context: MemoryRequestContext,
    ) -> DreamingPipelineResult:
        """Dispatch dreaming according to ``request.mode``."""

    async def dream_sync(
        self,
        request: DreamingPipelineInput,
        context: MemoryRequestContext,
    ) -> DreamingPipelineResult:
        """Run consolidation synchronously."""


class SkillEvolvePipeline(Protocol):
    """Skill self-evolution pipeline contract."""

    async def evolve(
        self,
        context: RequestContext,
        request: EvolveSkillRequest,
    ) -> SkillEvolveResult:
        """Run one evolution pass for a project skill."""


__all__ = [
    "AddPipeline",
    "DeletePipeline",
    "DreamingPipeline",
    "FeedbackPipeline",
    "GetPipeline",
    "PipelineBase",
    "MemoryPersistencePipelineMixin",
    "SearchPipeline",
    "SkillEvolvePipeline",
    "UpdatePipeline",
]
