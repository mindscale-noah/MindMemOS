"""Fixed model settings for TreeSkill's task-time subtree router."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ....agents.buitin_agents.openai_chat.config import OpenAIChatConfig
from ....agents.config import AgentConfig
from ..config import EnvAdapterSelection
from .typing import AnnotationMode


class TreeSkillConfig(BaseModel):
    """Configure routing without storing optimizable instructions outside source content."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")
    """Reject unknown router settings."""

    env_adapter: EnvAdapterSelection = Field(default_factory=EnvAdapterSelection)
    """Registered adaptation restored with this dynamic Skill."""

    tree_metadata: dict = Field(default_factory=dict)
    """Verified topology and identity ledger for reference routing; contains no instructions."""

    router: Annotated[OpenAIChatConfig, AgentConfig(allowed=("openai_chat",), namespace="treeskill")]
    """Router model settings, including generation limits and provider retries."""


class TreeSkillTrainerConfig(BaseModel):
    """Settings for one active collection, analysis and fusion round."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid", frozen=True)
    """Reject misspelled settings and freeze the experiment protocol."""

    # Training models
    optimizer: Annotated[OpenAIChatConfig, AgentConfig(allowed=("openai_chat",), namespace="treeskill")]
    """Model transport shared by analysis, localization and fusion."""

    failure_optimizer: (
        Annotated[OpenAIChatConfig, AgentConfig(allowed=("openai_chat",), namespace="treeskill")] | None
    ) = None
    """Reference failure-analysis model; None reuses optimizer settings."""

    # Trajectory collection
    samples_per_task: int = Field(default=1, ge=1)
    """Independent training trajectories sampled from each task."""

    rollout_concurrency: int = Field(default=32, ge=1)
    """Maximum simultaneous environment executions."""

    # Evidence selection
    min_trajectories: int = Field(default=1, ge=1)
    """Minimum completed trajectories required to update the Skill."""

    max_trajectories: int = Field(default=1000, ge=1)
    """Maximum evidence count accepted by one training round."""

    annotation_mode: AnnotationMode = AnnotationMode.REQUIRED
    """Whether to include environment reward labels and feedback."""

    require_skill_match: bool = True
    """Verify sampled trajectories belong to the initial Skill."""

    success_score_threshold: float = 1.0
    """Score cutoff separating success and error analyses."""

    # Trajectory analysis
    analysis_concurrency: int = Field(default=32, ge=1)
    """Maximum simultaneous trajectory analyses."""

    analysis_temperature: float = Field(default=1.0, ge=0)
    """Sampling temperature for trajectory analysis."""

    analysis_max_tokens: int = Field(default=16384, ge=1)
    """Completion budget per analysis call."""

    analysis_max_turns: int = Field(default=20, ge=1)
    """Maximum tool-use turns in reference failure analysis."""

    shell_timeout_seconds: int = Field(default=120, ge=1)
    """Reference analysis shell-command timeout in seconds."""

    transcript_max_chars: int = Field(default=20000, ge=1)
    """Reference truncation limit per event, not per trajectory."""

    # Evidence localization
    localization_concurrency: int = Field(default=32, ge=1)
    """Maximum simultaneous evidence localization calls."""

    localization_temperature: float = Field(default=0.0, ge=0)
    """Sampling temperature for localization."""

    localization_max_tokens: int = Field(default=16384, ge=1)
    """Initial completion budget for localization."""

    # Node fusion
    fusion_temperature: float = Field(default=0.0, ge=0)
    """Sampling temperature for node fusion."""

    fusion_max_tokens: int = Field(default=16384, ge=1)
    """Completion budget for each node fusion."""

    # Task-time routing
    router: Annotated[OpenAIChatConfig, AgentConfig(allowed=("openai_chat",), namespace="treeskill")]
    """Task-time routing model for the final Skill."""

    routing_temperature: float = Field(default=0.0, ge=0)
    """Sampling temperature used by the final task-time router."""

    routing_max_tokens: int = Field(default=16384, ge=1)
    """Completion-token budget for one subtree-selection request."""

    # Protocol versions
    algorithm_version: str = Field(default="1", min_length=1)
    """Algorithm behavior revision."""

    prompt_version: str = Field(default="treeskill-v1", min_length=1)
    """Prompt contract revision."""

    @model_validator(mode="after")
    def validate_batch_bounds(self) -> TreeSkillTrainerConfig:
        if self.max_trajectories < self.min_trajectories:
            raise ValueError("max_trajectories must be greater than or equal to min_trajectories")
        return self


class TreeSkillEnvAdapterConfig(BaseModel):
    """Override generic analysis and routing instructions with literal text."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid", frozen=True)
    """Reject unknown settings and avoid run-local mutable configuration."""

    analysis_system: str | None = None
    """Complete analysis system prompt; None uses the bundled protocol."""

    routing_system: str | None = None
    """Complete routing system prompt; None uses the adapter's default protocol."""
