"""Training pipeline configuration."""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, model_validator

from ...config import PipelineComponentConfig


class TrainingPipelineConfig(BaseModel):
    """Configure a trainer independently of each run's environment."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")
    """Reject misspelled pipeline settings."""

    trainer: PipelineComponentConfig
    """Optimization implementation and its private configuration."""

    evaluation_concurrency: int = Field(default=1, ge=1)
    """Maximum concurrent held-out evaluations after training."""

    @model_validator(mode="after")
    def validate_components(self, info: ValidationInfo) -> Self:
        """Validate component families and options without creating resources.

        Args:
            info: Validation context carrying the service-local Agent catalog.

        Returns:
            The validated pipeline configuration.

        Raises:
            ValueError: A component is unknown, invalid, or not a trainer.
        """
        from ....algos import Algorithm
        from ....algos.train.base import SkillTrainer
        from ....persistence.db_services.skills.typing import VersionNaming

        VersionNaming(algorithm=self.trainer.class_name.replace("_", "-"))
        trainer_type = Algorithm.resolve(self.trainer.class_name)
        if not issubclass(trainer_type, SkillTrainer):
            raise ValueError("trainer must select a SkillTrainer")
        self.trainer.config = trainer_type.config_type.model_validate(self.trainer.config, context=info.context)
        return self
