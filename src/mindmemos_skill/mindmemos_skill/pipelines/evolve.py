"""Business contract for Skill evolution from prepared execution evidence."""

from abc import ABC, abstractmethod

from ..api.typing import AlgoSkillEvolveResult, SkillPipelineRequest
from ..registry.base import ConfigT
from .base import Pipeline


class EvolutionPipeline(Pipeline[ConfigT], ABC):
    """Prepare evidence, invoke an evolver, and persist a selected Skill.

    Concrete implementations choose injected algorithms and data services.
    Evidence may come from stored trajectories or caller-scoped environment
    sampling before evolution. The evolver receives materialized historical
    task, feedback, and actual Skill snapshots, never database handles or lazy
    queries. Candidate-dependent sampling loops belong to a training pipeline.
    Publishing implementations pass their fixed identifier through VersionNaming.algorithm;
    persistence allocates the reserved suffix and next patch number.
    """

    @abstractmethod
    async def evolve(self, request: SkillPipelineRequest) -> AlgoSkillEvolveResult:
        """Evolve one Skill and commit a child of its captured current version.

        Implementations must preserve project scope and source data, avoid
        holding database transactions during algorithm execution, retain the
        candidate on persistence failure, and reject concurrent version changes.
        They must not mutate the request or return success before persistence.

        Args:
            request: Skill selection and optional expected current version.

        Returns:
            Persisted Skill/version identities and the execution identity.

        Raises:
            ValueError: Inputs or required evolution evidence are invalid.
            SkillConflictError: The expected current version changed.
        """
