"""Validate an environment selection using the implementation-owned configuration."""

from typing import Self

from pydantic import model_validator

from ..config.config import ImplementationConfig


class EnvironmentConfig(ImplementationConfig):
    """Select and validate one run's environment without constructing resources."""

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Retain the selected environment's original configuration model.

        Returns:
            Validated selection ready for Environment.build.

        Raises:
            ValueError: The environment is unknown or its configuration is invalid.
        """
        from . import Environment

        implementation = Environment.resolve(self.class_name)
        self.config = implementation.config_type.model_validate(self.config)
        return self
