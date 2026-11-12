"""Compatibility fields carried by the Memory API."""

from typing import Annotated, Literal

from pydantic import BaseModel, StringConstraints

NonemptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
"""Nonblank text normalized like the legacy Memory request contract."""


class SkillContext(BaseModel):
    """Reference to a legacy Skill source used during one Memory add."""

    name: NonemptyText
    """Name of the local Skill source observed by the old plugin."""

    content_hash: NonemptyText
    """Legacy content fingerprint used by Memory to resolve the source version."""

    base_version_id: str = ""
    """Legacy parent version ID; empty before the first registration."""

    version_label: NonemptyText | None = None
    """Optional version label from the local source."""

    usage: Literal["injected", "modified"] | None = None
    """Whether the source was injected into the conversation or modified."""
