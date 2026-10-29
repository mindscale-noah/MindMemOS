"""Multimodal memory DTOs: stored images and image search hits."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class ImageOccurrence(BaseModel):
    """One appearance of an image in a dialogue (a deduplicated image keeps several)."""

    session_id: str | None = None
    request_id: str | None = None
    agent_id: str | None = None
    role: str | None = None
    message_index: int | None = None
    event_timestamp_ms: int | None = None
    context: str = ""
    """Dialogue text around the image (the turn carrying it and the next one)."""
    caller_image_id: str | None = None
    caption: str | None = None


class ImageRecord(BaseModel):
    """A stored image of one user in one project, deduplicated by content."""

    image_id: str
    """Deterministic point id: uuid5 of (project, user, blob_id)."""
    blob_id: str
    mime_type: str
    size_bytes: int
    account_id: str | None = None
    project_id: str
    api_key_uuid: str | None = None
    user_id: str | None = None
    app_id: str | None = None
    created_at: datetime
    update_at: datetime
    event_time: datetime | None = None
    """Event time of the first occurrence."""
    embed_model: str | None = None
    source_url: str | None = None
    description: str | None = None
    """VLM description of the image including visible text (enhanced mode)."""
    visual_facts: list[str] = Field(default_factory=list)
    """Facts read from the pixels by the VLM (enhanced mode); kept here, never in text memory."""
    occurrences: list[ImageOccurrence] = Field(default_factory=list)
    anchored_memory_ids: list[str] = Field(default_factory=list)
    """Text memories extracted from the turns showing this image (enhanced mode)."""

    @property
    def session_ids(self) -> list[str]:
        return _unique(occ.session_id for occ in self.occurrences)

    @property
    def request_ids(self) -> list[str]:
        return _unique(occ.request_id for occ in self.occurrences)

    @property
    def agent_ids(self) -> list[str]:
        return _unique(occ.agent_id for occ in self.occurrences)


class ImageSearchHit(BaseModel):
    record: ImageRecord
    score: float


def _unique(values) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))
