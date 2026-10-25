"""Service pipeline configuration belonging only to offline evolution."""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, model_validator

from .....pipelines.config import PipelineComponentConfig


class OfflineTrajectoryConfig(BaseModel):
    """Use all exact-version evidence unless the offline caller sets explicit caps."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")
    """Expose offline-owned collection options and reject unknown fields."""

    trajectory_limit: int | None = Field(default=None, ge=1)
    """Maximum recent records collected per exact version; None reads all available records."""

    max_page_size: int = Field(default=200, ge=1, le=10_000)
    """Maximum records fetched per database query; does not limit the total evidence count."""


class OfflineEvolutionPipelineConfig(BaseModel):
    """Configure offline evolution with optional limits on historical evidence."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")
    """Reject unsupported evidence and publication settings."""

    evolver: PipelineComponentConfig
    """Registered offline collection algorithm and its validated model options."""

    work_dir: str = Field(min_length=1)
    """Directory where candidates are saved before publication."""

    traj_collect: OfflineTrajectoryConfig = Field(default_factory=OfflineTrajectoryConfig)
    """Offline-owned collection settings; defaults use all available exact-version evidence."""

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
        from .....algos import Algorithm
        from .....persistence.db_services.skills.typing import VersionNaming
        from ..evolver import OfflineSkillEvolver

        selected = Algorithm.resolve(self.evolver.class_name)
        if not issubclass(selected, OfflineSkillEvolver):
            raise ValueError("evolver must select OfflineSkillEvolver")
        VersionNaming(algorithm=self.evolver.class_name.replace("_", "-"))
        self.evolver.config = selected.config_type.model_validate(self.evolver.config, context=info.context)
        return self
