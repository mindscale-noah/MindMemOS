"""Transport-independent model response identities."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

Nonempty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
"""Nonblank identity returned by the gateway."""


class EmbeddingSpace(BaseModel):
    """Identify the authorized vector space exposed by the model gateway."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    """Reject undeclared response fields and freeze the resolved identity."""

    space_id: Nonempty
    """Stable identity of the embedding deployment or revision."""

    model: Nonempty
    """Model name associated with the vector space."""

    dimensions: int = Field(gt=0, strict=True)
    """Required length of every vector in the space."""
