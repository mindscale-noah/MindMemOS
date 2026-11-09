"""Strict models for static judge responses."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DimensionScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: int = Field(ge=1, le=5)
    reason: str = Field(min_length=1)


class SkillJudgeVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scores: dict[str, DimensionScore]
    overall_summary: str = Field(min_length=1)
    concerns: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)

    @field_validator("scores")
    @classmethod
    def validate_dimensions(cls, value: dict[str, DimensionScore]) -> dict[str, DimensionScore]:
        expected = {"safety", "completeness", "executability", "maintainability"}
        if set(value) != expected:
            raise ValueError(f"scores must contain exactly: {', '.join(sorted(expected))}")
        return value

    @field_validator("concerns", "evidence")
    @classmethod
    def validate_strings(cls, value: list[Any]) -> list[str]:
        if any(not isinstance(item, str) or not item.strip() for item in value):
            raise ValueError("concerns and evidence must contain non-empty strings")
        return value


class WindowFinding(BaseModel):
    """One observation from a single sliding-window scan.

    Window scans collect evidence only (no dimension scoring — a window
    cannot see the whole skill); ``severity`` grades the observation and
    ``location`` pins it to a file byte range for the merge stage.
    """

    model_config = ConfigDict(extra="forbid")

    severity: str
    description: str = Field(min_length=1)
    location: str = Field(min_length=1)

    @field_validator("severity")
    @classmethod
    def normalize_severity(cls, value: str) -> str:
        """Accept common severity aliases models emit (high/medium/low) and
        fold them onto the canonical three levels."""

        aliases = {
            "high": "critical",
            "severe": "critical",
            "critical": "critical",
            "medium": "warning",
            "moderate": "warning",
            "warning": "warning",
            "warn": "warning",
            "low": "info",
            "minor": "info",
            "info": "info",
            "informational": "info",
        }
        normalized = aliases.get(value.strip().lower())
        if normalized is None:
            raise ValueError(f"severity must be one of info/warning/critical (got {value!r})")
        return normalized


class WindowScan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    findings: list[WindowFinding] = Field(default_factory=list)


__all__ = ["DimensionScore", "SkillJudgeVerdict", "WindowFinding", "WindowScan"]
