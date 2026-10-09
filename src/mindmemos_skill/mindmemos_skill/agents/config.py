"""Shared model connection settings supplied explicitly by callers."""

from pydantic import BaseModel, ConfigDict, Field


class AgentConnectionConfig(BaseModel):
    """Keep model credentials and endpoints exclusively in configuration."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid", hide_input_in_errors=True)
    """Reject unknown fields and omit input values from validation error messages."""

    api_key: str | None = Field(default=None, min_length=1)
    """Model credential; required for model calls and preserved in serialized configuration."""

    base_url: str | None = Field(default=None, min_length=1)
    """Model endpoint; required for model calls, with no environment fallback."""

    def require_connection(self) -> None:
        """Require explicit connection settings before invoking a model.

        Raises:
            ValueError: The configuration omits a credential or endpoint.
        """
        if self.api_key is None:
            raise ValueError("Provide api_key in the agent config")
        if self.base_url is None:
            raise ValueError("Provide base_url in the agent config")
