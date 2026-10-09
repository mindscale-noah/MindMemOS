"""Fixed model settings for TreeSkill's task-time subtree router."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ....agents.buitin_agents.openai_chat.config import OpenAIChatConfig
from ..config import EnvAdapterSelection
from .typing import AnnotationMode


class TreeSkillConfig(BaseModel):
    """Configure routing without storing optimizable instructions outside source content."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")
    """Reject unknown router settings."""

    env_adapter: EnvAdapterSelection = Field(default_factory=EnvAdapterSelection)
    """Registered adaptation restored with this dynamic Skill."""

    reference_routing: bool = False
    """Use the pinned upstream routing protocol and stable fusion node IDs."""

    tree_metadata: dict = Field(default_factory=dict)
    """Verified topology and identity ledger for reference routing; contains no instructions."""

    router: OpenAIChatConfig
    """Router model settings, including generation limits and provider retries."""


class TreeSkillTrainerConfig(BaseModel):
    """Settings for one active collection, analysis and fusion round."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid", frozen=True)
    """Reject misspelled settings and freeze the experiment protocol."""

    optimizer: OpenAIChatConfig
    """Model transport shared by analysis, localization and fusion."""

    router: OpenAIChatConfig
    """Task-time routing model for the final Skill."""

    failure_optimizer: OpenAIChatConfig | None = None
    """Reference failure-analysis model; None reuses optimizer settings."""

    samples_per_task: int = Field(default=1, ge=1)
    """Independent training trajectories sampled from each task."""

    rollout_concurrency: int = Field(default=8, ge=1)
    """Maximum simultaneous environment executions."""

    analysis_max_turns: int = Field(default=20, ge=1)
    """Maximum tool-use turns in reference failure analysis."""

    shell_timeout_seconds: int = Field(default=120, ge=1)
    """Reference analysis shell-command timeout in seconds."""

    algorithm_version: str = Field(default="1", min_length=1)
    """Algorithm behavior revision."""

    prompt_version: str = Field(default="treeskill-v1", min_length=1)
    """Prompt contract revision."""

    min_trajectories: int = Field(default=1, ge=1)
    """Minimum completed trajectories required to update the Skill."""

    max_trajectories: int = Field(default=1000, ge=1)
    """Maximum evidence count accepted by one training round."""

    transcript_max_chars: int = Field(default=20000, ge=1)
    """Reference truncation limit per event, not per trajectory."""

    annotation_mode: AnnotationMode = AnnotationMode.REQUIRED
    """Whether to include environment reward labels and feedback."""

    require_skill_match: bool = True
    """Verify sampled trajectories belong to the initial Skill."""

    success_score_threshold: float = 1.0
    """Score cutoff separating success and error analyses."""

    analysis_concurrency: int = Field(default=16, ge=1)
    """Maximum simultaneous trajectory analyses."""

    localization_concurrency: int = Field(default=16, ge=1)
    """Maximum simultaneous evidence localization calls."""

    analysis_temperature: float = Field(default=1.0, ge=0)
    """Sampling temperature for trajectory analysis."""

    localization_temperature: float = Field(default=0.0, ge=0)
    """Sampling temperature for localization."""

    fusion_temperature: float = Field(default=0.0, ge=0)
    """Sampling temperature for node fusion."""

    analysis_max_tokens: int = Field(default=4096, ge=1)
    """Completion budget per analysis call."""

    localization_max_tokens: int = Field(default=2048, ge=1)
    """Initial completion budget for localization."""

    fusion_max_tokens: int = Field(default=4096, ge=1)
    """Completion budget for each node fusion."""

    analysis_task: str = Field(default="treeskill_trajectory_analysis", min_length=1)
    """Audit label for trajectory analysis calls."""

    localization_task: str = Field(default="treeskill_evidence_localization", min_length=1)
    """Audit label for localization calls."""

    fusion_task: str = Field(default="treeskill_node_fusion", min_length=1)
    """Audit label for fusion calls."""

    @model_validator(mode="after")
    def validate_batch_bounds(self) -> TreeSkillTrainerConfig:
        if self.max_trajectories < self.min_trajectories:
            raise ValueError("max_trajectories must be greater than or equal to min_trajectories")
        return self


class TreeSkillExperimentConfig(BaseModel):
    """Standalone script inputs, independent of evaluation services or databases."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")
    """Reject unknown experiment fields."""

    algorithm: TreeSkillTrainerConfig
    """Training and routing model settings."""

    environment: str
    """Environment registry name, or spreadsheetbench_text_action."""

    environment_options: dict = Field(default_factory=dict)
    """Options validated by the selected environment's config model."""

    data: str
    """Data registry name for the materialized task loader."""

    data_options: dict = Field(default_factory=dict)
    """Options validated by the selected data loader."""

    split_file: str
    """JSON mapping train, val and test to disjoint task IDs."""

    initial_skill: str
    """Initial Skill directory or SKILL.md entrypoint."""

    train_limit: int | None = Field(default=None, ge=1)
    """Optional prefix of the ordered training split, without reshuffling."""

    test_limit: int | None = Field(default=None, ge=1)
    """Optional prefix of the ordered held-out split, without reshuffling."""

    strict_reference_skill: bool = False
    """Require exact authorized reference xlsx package hashes before reproduction."""


class TreeSkillEnvAdapterConfig(BaseModel):
    """Override generic analysis and routing instructions with literal text."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid", frozen=True)
    """Reject unknown settings and avoid run-local mutable configuration."""

    analysis_system: str | None = None
    """Complete analysis system prompt; None uses the bundled protocol."""

    routing_system: str | None = None
    """Complete routing system prompt; None uses the adapter's default protocol."""
