"""Shared model connection settings with explicit values taking precedence."""

import os

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

from .constants import AGENT_API_KEY_ENV, AGENT_BASE_URL_ENV


class AgentConnectionConfig(BaseModel):
    """Resolve shared environment defaults when configuration is constructed."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    """Reject unknown fields and omit input values from validation error messages."""

    api_key: SecretStr | None = Field(default=None, min_length=1, validate_default=True, repr=False)
    """Explicit credential; None reads MINDMEMOS_AGENT_API_KEY, with JSON output masked."""

    base_url: str | None = Field(default=None, min_length=1, validate_default=True)
    """Explicit endpoint; None reads MINDMEMOS_AGENT_BASE_URL."""

    @field_validator("api_key", mode="before")
    @classmethod
    def resolve_api_key(cls, value: str | SecretStr | None) -> str | SecretStr | None:
        """Use the shared credential only when no explicit value was supplied.

        Args:
            value: Explicit credential or None to use the environment.

        Returns:
            Supplied value or the nonempty environment default, if present.
        """
        return (os.environ.get(AGENT_API_KEY_ENV) or None) if value is None else value

    @field_validator("base_url", mode="before")
    @classmethod
    def resolve_base_url(cls, value: str | None) -> str | None:
        """Use the shared endpoint only when no explicit value was supplied.

        Args:
            value: Explicit endpoint or None to use the environment.

        Returns:
            Supplied value or the nonempty environment default, if present.
        """
        return (os.environ.get(AGENT_BASE_URL_ENV) or None) if value is None else value
