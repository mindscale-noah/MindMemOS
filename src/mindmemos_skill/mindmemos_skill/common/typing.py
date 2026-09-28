"""Validated records for shared local checkpoint operations."""

from typing import Literal

from pydantic import BaseModel, JsonValue


class RecoveryRecord(BaseModel):
    """Complete recovery boundary with verified inputs and payload integrity."""

    version: Literal[1] = 1
    """Supported checkpoint protocol version."""

    identity: str
    """Fingerprint of the inputs that produced this boundary."""

    payload: dict[str, JsonValue]
    """Atomically committed state or completed output."""

    digest: str
    """Fingerprint detecting accidental payload modification."""
