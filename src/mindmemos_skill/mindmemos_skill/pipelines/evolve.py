"""Business contract for Skill evolution from prepared execution evidence."""

from abc import ABC, abstractmethod

from ..api.typing import EvolutionRequest, EvolutionResult
from ..errors import SkillNotFoundError, SkillVersionNotFoundError
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
            request: Prepared list of exact versions and editing permissions.

        Returns:
            Persisted single-Skill or collection version identities.

        Raises:
            ValueError: Inputs or required evolution evidence are invalid.
            SkillConflictError: The expected current version changed.
        """

    async def prepare_request(self, request: EvolutionRequest) -> EvolutionRequest:
        """Validate input versions and editing permissions before execution.

        Args:
            request: Exact versions and editing permissions in the active project.

        Returns:
            Validated evolution request.

        Raises:
            SkillVersionNotFoundError: A version is not visible in the active project.
            ValueError: A Skill appears more than once or the editable list names a non-input Skill.
        """
        request = EvolutionRequest.model_validate(request)
        if self.persistence is None:
            return request

        try:
            versions = [await self.persistence.skills.get(version_id) for version_id in request.skill_version_ids]
        except SkillNotFoundError as exc:
            raise SkillVersionNotFoundError("Skill version not found in this project") from exc
        owners = {version.skill_id for version in versions}
        if len(owners) != len(versions):
            raise ValueError("Evolution accepts at most one version per Skill family")
        if request.editable_skills is not None and not set(request.editable_skills) <= owners:
            raise ValueError("Editable Skill IDs must belong to the input collection")
        return request
