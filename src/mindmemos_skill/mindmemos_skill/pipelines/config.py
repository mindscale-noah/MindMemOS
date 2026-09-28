"""Configuration for built-in repository pipelines."""

from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RepositoryPipelineConfig(BaseModel):
    """Repository operations currently need only their injected persistence."""

    model_config = ConfigDict(extra="forbid")
    """Reject unsupported repository options."""


class TrainingComponentConfig(BaseModel):
    """Select a trainer or environment in its owning registry."""

    model_config = ConfigDict(extra="forbid")
    """Reject unsupported assembly fields."""

    class_name: str
    """Explicit implementation name in the field's registry."""

    config: dict[str, Any] = Field(default_factory=dict)
    """Complete options validated before any resource construction."""


class TrainingPipelineConfig(BaseModel):
    """Compose a trainer and environment without algorithm-specific branches."""

    model_config = ConfigDict(extra="forbid")
    """Reject misspelled pipeline settings."""

    trainer: TrainingComponentConfig
    """Optimization implementation and its private configuration."""

    environment: TrainingComponentConfig
    """Execution and scoring implementation supplied to the trainer."""

    evaluation_concurrency: int = Field(default=1, ge=1)
    """Maximum concurrent held-out evaluations after training."""

    @model_validator(mode="after")
    def validate_components(self) -> Self:
        """Validate component families and options without creating resources.

        Returns:
            The validated pipeline configuration.

        Raises:
            ValueError: A component is unknown, invalid, or not a trainer.
        """
        from ..algos import Algorithm
        from ..algos.train.base import SkillTrainer
        from ..envs import Environment
        from ..persistence.db_services.skills.typing import VersionNaming

        VersionNaming(algorithm=self.trainer.class_name.replace("_", "-"))
        trainer_type = Algorithm.resolve(self.trainer.class_name)
        if not issubclass(trainer_type, SkillTrainer):
            raise ValueError("trainer must select a SkillTrainer")
        trainer_type.config_type.model_validate(self.trainer.config)
        Environment.resolve(self.environment.class_name).config_type.model_validate(self.environment.config)
        return self
