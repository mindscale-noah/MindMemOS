"""Explicit OpenAI-compatible embedding connection and batching settings."""

from pydantic import BaseModel, ConfigDict, Field


class OpenAIEmbeddingConfig(BaseModel):
    """Configure embeddings without selecting an implicit model or endpoint."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid", frozen=True, hide_input_in_errors=True)
    """Reject unknown settings and omit credentials from validation errors."""

    model: str = Field(min_length=1)
    """Provider model or deployment name."""

    base_url: str = Field(min_length=1)
    """Explicit OpenAI-compatible API base URL."""

    api_key: str | None = Field(default=None)
    """Provider credential; None supports unauthenticated local endpoints."""

    dimensions: int = Field(gt=0, strict=True)
    """Expected output dimension used to validate every returned vector."""

    send_dimensions: bool = False
    """Send the dimensions request parameter only when supported by the provider."""

    batch_size: int = Field(default=64, gt=0, strict=True)
    """Maximum number of input texts in each sequential request."""

    timeout_s: float = Field(default=180, gt=0, allow_inf_nan=False)
    """Timeout in seconds for each request attempt."""

    max_retries: int = Field(default=3, ge=0, strict=True)
    """Additional SDK attempts after a transient request failure."""
