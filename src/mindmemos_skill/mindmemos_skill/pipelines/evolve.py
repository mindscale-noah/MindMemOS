"""Business contract for Skill evolution from prepared execution evidence."""

from abc import ABC, abstractmethod

from ..api.typing import EvolutionRequest, EvolutionResult, SkillPipelineRequest
from ..registry.base import ConfigT
from .base import Pipeline


class EvolutionPipeline(Pipeline[ConfigT], ABC):
    """Prepare evidence, invoke an evolver, and publish the resulting Skill versions.

    Concrete implementations choose injected algorithms and data services.
    Evidence may come from stored trajectories or caller-scoped environment
    sampling before evolution. The evolver receives materialized historical
    task, feedback, and actual Skill snapshots, never database handles or lazy
    queries. Candidate-dependent sampling loops belong to a training pipeline.
    Publishing implementations pass their fixed identifier through VersionNaming.algorithm;
    persistence allocates the reserved suffix and next patch number.
    """

    @abstractmethod
    async def evolve(self, request: EvolutionRequest) -> EvolutionResult:
        """Evolve prepared Skill inputs and return the pipeline's declared outcome.

        Implementations must preserve project scope and source data, avoid
        holding database transactions during algorithm execution, retain the
        candidate on persistence failure, and reject concurrent version changes.
        They must not mutate the request or return success before persistence.

        Args:
            request: Prepared single-Skill selection or exact version collection.

        Returns:
            Persisted single-Skill or collection version identities.

        Raises:
            ValueError: Inputs or required evolution evidence are invalid.
            SkillConflictError: The expected current version changed.
        """

    async def prepare_request(self, request: EvolutionRequest) -> EvolutionRequest:
        """Validate supported input before task admission or direct execution.

        Args:
            request: Authorized evolution selection.

        Returns:
            The original single-Skill request; collection implementations override this method.
        """
        return SkillPipelineRequest.model_validate(request)
