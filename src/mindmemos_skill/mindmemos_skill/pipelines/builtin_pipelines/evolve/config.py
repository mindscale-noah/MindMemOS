"""Simple evolution pipeline configuration."""

from typing import Literal, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from ...config import PipelineComponentConfig


class SimpleEvolutionPipelineConfig(BaseModel):
    """Bound the evidence pool and configure one built-in evolver."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")
    """Reject unsupported evidence and publication settings."""

    evolver: PipelineComponentConfig
    """Registered SkillEvolver and its validated model options."""

    work_dir: str = Field(min_length=1)
    """Directory where candidates are saved before publication."""

    sample_size: int = Field(default=8, ge=1, le=100)
    """Number of distinct eligible attempts sampled without replacement."""

    trajectory_limit: int = Field(default=100, ge=1, le=1000)
    """Maximum matching attempts retained before sampling."""

    scan_limit: int = Field(default=1000, ge=1, le=10_000)
    """Maximum rows inspected while finding attempts for this Skill."""

    created_from: AwareDatetime | None = None
    """Inclusive lower bound on stored trajectory creation time."""

    created_to: AwareDatetime | None = None
    """Inclusive upper bound on stored trajectory creation time."""

    status: str | None = Field(default=None, min_length=1)
    """Optional exact execution status of collected attempts."""

    source: Literal["local", "reported"] | None = None
    """Optional local or Harness-reported attempt origin."""

    max_record_bytes: int = Field(default=262_144, ge=1024, le=10_485_760)
    """Maximum serialized size of one retained historical attempt."""

    max_total_bytes: int = Field(default=8_388_608, ge=1024, le=67_108_864)
    """Maximum serialized size of all attempts retained before sampling."""

    seed: int | None = None
    """Optional reproducible random sample seed; None uses system randomness."""

    @model_validator(mode="after")
    def validate_evolver(self) -> Self:
        """Reject inconsistent bounds or a non-evolution algorithm."""
        from ....algos import Algorithm
        from ....algos.evolve.base import SkillEvolver
        from ....persistence.db_services.skills.typing import VersionNaming

        if self.sample_size > self.trajectory_limit or self.trajectory_limit > self.scan_limit:
            raise ValueError("Require sample_size <= trajectory_limit <= scan_limit")
        if self.created_from is not None and self.created_to is not None and self.created_to < self.created_from:
            raise ValueError("Trajectory creation bounds are reversed")
        selected = Algorithm.resolve(self.evolver.class_name)
        if not issubclass(selected, SkillEvolver):
            raise ValueError("evolver must select a SkillEvolver")
        VersionNaming(algorithm=self.evolver.class_name.replace("_", "-"))
        self.evolver.config = selected.config_type.model_validate(self.evolver.config)
        return self
