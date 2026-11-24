"""Strict models for trajectory judge responses."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SuccessVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    success: bool
    confidence: Literal["low", "medium", "high"]
    reason: str = Field(min_length=1)
    evidence: list[str] = Field(min_length=1)

    @field_validator("evidence")
    @classmethod
    def validate_evidence(cls, value: list[Any]) -> list[str]:
        if any(not isinstance(item, str) or not item.strip() for item in value):
            raise ValueError("evidence must contain non-empty strings")
        return value


class AttributionSegment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    skill: str = Field(min_length=1)
    suspicion: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1)
    evidence: list[str] = Field(default_factory=list)


class PrimaryCause(BaseModel):
    model_config = ConfigDict(extra="forbid")

    skill: str | None = None
    reason: str = Field(min_length=1)
    confidence: Literal["low", "medium", "high"]


class AttributionVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    segments: list[AttributionSegment] = Field(default_factory=list)
    primary_cause: PrimaryCause


__all__ = ["AttributionSegment", "AttributionVerdict", "PrimaryCause", "SuccessVerdict"]
