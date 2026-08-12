"""Data models for the feedback-driven self-evolution (``feedback_evo``) mode."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ParameterChange(BaseModel):
    """One parameter modification proposed by the evolution planner."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(
        description="Dotted parameter path, e.g. 'search_config.weights.fact'."
    )
    before: Any = Field(default=None, description="Value before the change.")
    after: Any = Field(description="Value after the change.")
    reason: str | None = Field(default=None, description="Why this parameter changed.")


class EvolutionTrigger(BaseModel):
    """What triggered one evolution version."""

    model_config = ConfigDict(extra="forbid")

    signal_ids: list[str] = Field(
        default_factory=list,
        description="Feedback signal ids that triggered this evolution.",
    )


class EvolutionState(BaseModel):
    """One version of the evolvable configuration for a project."""

    model_config = ConfigDict(extra="forbid")

    project_id: str
    mode: str = "feedback_evo"
    version: int = Field(default=1, ge=1)
    is_current: bool = True
    add_config: dict[str, Any] = Field(default_factory=dict)
    search_config: dict[str, Any] = Field(default_factory=dict)
    trigger: EvolutionTrigger | None = Field(default=None)
    changes: list[ParameterChange] = Field(default_factory=list)
    rollback_version: int | None = Field(default=None)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def state_id(self) -> str:
        """Stable Qdrant point id: ``{project_id}:v{version}``."""

        return f"{self.project_id}:v{self.version}"


class EvolutionResult(BaseModel):
    """Result of applying one evolution round."""

    model_config = ConfigDict(extra="forbid")

    project_id: str
    version: int
    is_rollback: bool = False
    changes: list[ParameterChange] = Field(default_factory=list)


class FeedbackEvoEvent(BaseModel):
    """One feedback event recorded at task end (payload for ``feedback_event_v1``)."""

    model_config = ConfigDict(extra="forbid")

    event_id: str
    account_id: str
    project_id: str
    api_key_uuid: str
    user_id: str | None = None
    session_id: str | None = None
    app_id: str | None = None
    agent_id: str | None = None
    task_id: str | None = Field(
        default=None,
        description="Identifier of the finished task that produced this event.",
    )
    submitted_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    signals: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Detected feedback signals (ImplicitFeedbackSignal-like dicts).",
    )
    actions: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Planned feedback actions (kept for audit; evolution ignores them).",
    )
