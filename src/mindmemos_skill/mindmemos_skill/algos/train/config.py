"""Registered environment adapter selection shared by training algorithms."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class EnvAdapterSelection(BaseModel):
    """Select an adapter from the owning algorithm's registry."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    """Reject misspelled fields and prevent selection reassignment."""

    class_name: str = Field(default="generic", min_length=1, pattern=r"^[a-z][a-z0-9_]*$")
    """Explicit registry name, scoped to one algorithm."""

    config: dict[str, Any] = Field(default_factory=dict)
    """Options validated by the selected adapter's configuration model."""
