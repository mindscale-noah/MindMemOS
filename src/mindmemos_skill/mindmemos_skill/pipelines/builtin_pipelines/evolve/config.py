"""Simple evolution pipeline configuration."""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, model_validator

from ...config import PipelineComponentConfig


class TrajCollectConfig(BaseModel):
    """Configure the trajectory candidate pool and sampling for one Skill version."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")
    """Expose field documentation and reject unsupported collection options."""

    sample_size: int = Field(default=40, ge=1, le=100)
    """Number of eligible trajectories sampled without replacement; fail if fewer are available."""

    trajectory_limit: int = Field(default=100, ge=1, le=1000)
    """Maximum recent matching trajectories retained as the candidate pool before sampling."""

    max_page_size: int = Field(default=200, ge=1, le=10_000)
    """Maximum matching rows fetched in one query; each page also respects the remaining pool capacity."""

    seed: int | None = None
    """Seed for reproducible sampling; None uses system randomness. Saved samples are reused on recovery."""

    @model_validator(mode="after")
    def validate_sample_size(self) -> Self:
        """Ensure the candidate pool can contain the requested sample."""
        if self.sample_size > self.trajectory_limit:
            raise ValueError("Require sample_size <= trajectory_limit")
        return self


class SimpleEvolutionPipelineConfig(BaseModel):
    """Bound the evidence pool and configure one built-in evolver."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")
    """Reject unsupported evidence and publication settings."""

    evolver: PipelineComponentConfig
    """Registered SkillEvolver and its validated model options."""

    work_dir: str = Field(min_length=1)
    """Directory where candidates are saved before publication."""

    traj_collect: TrajCollectConfig = Field(default_factory=TrajCollectConfig)
    """Candidate-pool and sampling settings; omitted collector arguments use function defaults."""

    @model_validator(mode="after")
    def validate_evolver(self, info: ValidationInfo) -> Self:
        """Reject a non-evolution algorithm and validate its model settings.

        Args:
            info: Validation context carrying the service-local Agent catalog.

        Returns:
            This pipeline with concrete algorithm-owned options.

        Raises:
            ValueError: The selected algorithm is not an evolver or its options are invalid.
        """
        from ....algos import Algorithm
        from ....algos.evolve.base import SkillEvolver
        from ....persistence.db_services.skills.typing import VersionNaming

        selected = Algorithm.resolve(self.evolver.class_name)
        if not issubclass(selected, SkillEvolver):
            raise ValueError("evolver must select a SkillEvolver")
        VersionNaming(algorithm=self.evolver.class_name.replace("_", "-"))
        self.evolver.config = selected.config_type.model_validate(self.evolver.config, context=info.context)
        return self
