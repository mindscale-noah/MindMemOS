"""Shared contracts for the offline trace2skill algorithm family."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, JsonValue


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AnnotationMode(StrEnum):
    """How an algorithm consumes trajectory reward annotations."""

    AUTO = "auto"
    REQUIRED = "required"
    IGNORE = "ignore"


class TraceEvidence(_StrictModel):
    """Normalized evidence extracted from one pre-collected trajectory."""

    trajectory_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    transcript: str
    score: float | None = None
    annotation_detail: str | None = None
    annotation_metadata: dict[str, JsonValue] = Field(default_factory=dict)


class EvidenceSelection(_StrictModel):
    """Deterministic evidence selection plus audit information."""

    evidence: list[TraceEvidence]
    duplicate_trajectory_ids: list[str] = Field(default_factory=list)


__all__ = ["AnnotationMode", "EvidenceSelection", "TraceEvidence"]
