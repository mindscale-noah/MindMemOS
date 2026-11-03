"""Transport-neutral data contracts for Skill analysis algorithms."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, JsonValue

from ...skill import Skill


class SkillFinding(BaseModel):
    """One actionable observation produced while analyzing a Skill or run."""

    model_config = ConfigDict(extra="forbid")
    """Reject unrecognized finding fields."""

    category: str = Field(min_length=1)
    """Stable machine-readable category such as ``static.quality``."""

    message: str = Field(min_length=1)
    """Human-readable description of the observation."""

    severity: Literal["info", "warning", "error"] = "info"
    """Operator-facing importance."""

    evidence: list[str] = Field(default_factory=list)
    """Concrete source, event, or metric references supporting the finding."""

    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    """Additional structured details specific to the producing analyzer."""


class SkillAnalysisResult(BaseModel):
    """Transport-neutral result returned by every analysis algorithm."""

    model_config = ConfigDict(extra="forbid")
    """Reject backend-specific top-level fields."""

    summary: str
    """Concise overall assessment."""

    findings: list[SkillFinding] = Field(default_factory=list)
    """Actionable observations in analyzer-defined order."""

    metrics: dict[str, JsonValue] = Field(default_factory=dict)
    """Deterministic scores, counts, and usage values."""

    artifacts: dict[str, str] = Field(default_factory=dict)
    """Optional generated text artifacts keyed by relative name."""

    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    """Identity and run provenance for persistence by the caller."""


class SkillIdentity(BaseModel):
    """Stable identity metadata attached to one analyzed Skill version."""

    model_config = ConfigDict(extra="forbid")
    """Reject identity fields that do not belong to the analysis boundary."""

    project_id: str | None = None
    """Owning service project, when the analysis is initiated by a stored version."""

    skill_id: str | None = None
    """Stable Skill family identity, when known."""

    version_id: str | None = None
    """Exact analyzed Skill version, when known."""

    name: str | None = None
    """Skill display name captured with the analyzed content."""

    description: str | None = None
    """Skill description captured with the analyzed content."""

    content_hash: str | None = None
    """Existing hash of the exact SKILL.md content, when supplied by persistence."""

    skill_hash: str | None = None
    """Existing complete source hash, when supplied by persistence."""

    @classmethod
    def from_skill(cls, skill: Skill) -> SkillIdentity:
        """Build identity metadata from an unversioned in-memory Skill.

        Args:
            skill: Skill whose current source is being analyzed.

        Returns:
            Identity containing the Skill's existing source hashes and display fields.
        """

        snapshot = skill.snapshot()
        return cls(
            name=snapshot.name,
            description=snapshot.description,
            content_hash=snapshot.content_hash,
            skill_hash=snapshot.skill_hash,
        )


class SkillJudgeRequest(BaseModel):
    """Input required for static analysis of one Skill source."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    """Allow the domain Skill object without copying its byte resources into JSON."""

    skill: Skill
    """Exact Skill source to inspect."""

    identity: SkillIdentity | None = None
    """Optional stored identity; omitted for in-memory callers without IDs."""

    request_id: str | None = None
    """Optional caller trace identity, independent of model credentials."""


class TrajectorySkillReference(BaseModel):
    """One injected Skill or resolved usage binding inside a trajectory."""

    model_config = ConfigDict(extra="ignore")
    """Ignore unrelated serialized runtime fields while preserving the core identity."""

    name: str = Field(min_length=1)
    """Skill display name used for detection and usage metrics."""

    usage: str | None = None
    """Resolved runtime usage state, such as ``injected``, ``modified``, or ``unused``."""

    skill_id: str | None = None
    """Skill family identity, when present in the stored binding."""

    version_id: str | None = None
    """Exact Skill version identity, when present in the stored binding."""

    content_hash: str | None = None
    """Skill content hash recorded with the binding, when present."""


class TrajectoryAnalysisInput(BaseModel):
    """Analyze-owned projection of one completed execution trajectory."""

    model_config = ConfigDict(extra="forbid")
    """Reject fields that have not been deliberately exposed to the algorithm."""

    trajectory_id: str = Field(min_length=1)
    """Identity of the analyzed physical attempt."""

    status: str = Field(min_length=1)
    """Execution outcome supplied by the trajectory owner."""

    task_instruction: str
    """Task instruction used for success and attribution prompts."""

    events: list[dict[str, JsonValue]] = Field(default_factory=list)
    """Ordered interaction events, including model and tool records."""

    injected_skills: list[TrajectorySkillReference] = Field(default_factory=list)
    """Skills made available to the run."""

    skill_bindings: list[TrajectorySkillReference] = Field(default_factory=list)
    """Resolved Skill/version usage information recorded by the runtime."""

    skill_id: str | None = None
    """Associated Skill family identity, when the trajectory targets one Skill."""

    skill_version_id: str | None = None
    """Associated exact Skill version, when the trajectory targets one version."""

    reward_score: FiniteFloat | None = None
    """Optional verifier score in the inclusive range from 0 to 1."""

    def skill_names(self) -> list[str]:
        """Return the union of injected and bound Skill names in sorted order."""

        return sorted(
            {
                *(reference.name for reference in self.skill_bindings),
                *(reference.name for reference in self.injected_skills),
            },
            key=str.lower,
        )


class TrajectoryJudgeRequest(BaseModel):
    """Input and per-run options for one trajectory analysis."""

    model_config = ConfigDict(extra="forbid")
    """Reject undeclared options so analysis behavior remains reproducible."""

    trajectory: TrajectoryAnalysisInput
    """Completed execution to analyze."""

    attribute: bool | None = None
    """Override configured failure attribution for this request."""

    verifier_score: FiniteFloat | None = None
    """Optional verifier score override; otherwise the trajectory reward is used."""

    request_id: str | None = None
    """Optional caller trace identity, independent of model credentials."""


__all__ = [
    "SkillAnalysisResult",
    "SkillFinding",
    "SkillIdentity",
    "SkillJudgeRequest",
    "TrajectoryAnalysisInput",
    "TrajectoryJudgeRequest",
    "TrajectorySkillReference",
]
