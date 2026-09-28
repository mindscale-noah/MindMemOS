"""Explicit controls for paper-defined Skill-Forest training."""

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ....agents import OpenAIChatConfig
from ..config import EnvAdapterSelection


class ForestOptimizerConfig(OpenAIChatConfig):
    """Single-attempt optimizer transport with a six-hour request timeout."""

    timeout_s: float = Field(default=21600, gt=0, allow_inf_nan=False)
    """Seconds allowed for one optimizer request."""

    max_retries: Literal[0] = 0
    """Never automatically repeat a failed provider request."""

    sdk_max_retries: Literal[0] = 0
    """Disable retries inside the provider SDK."""

    parse_max_attempts: Literal[1] = 1
    """Malformed responses fail the proposal without another model call."""

    max_completion_tokens: int | None = Field(default=16384, gt=0)
    """Patch-generation completion budget; caller may override provider limits."""


class SkillForestTrainerConfig(BaseModel):
    """Configure forest growth and final reduction independently of environments."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    """Reject unsupported strategies and freeze the run configuration."""

    optimizer: ForestOptimizerConfig
    """Shared transport for reflection, patch aggregation and Skill synthesis."""

    ensemble_strategy: Literal["select", "hierarchical_merge"] = "select"
    """Return a scored winner or synthesize winners through pairwise merging."""

    num_trees: int = Field(default=7, ge=1)
    """Number of independently partitioned optimizers."""

    train_ratio: float = Field(default=0.5, gt=0, lt=1)
    """Fraction of the supplied training pool used for tree-local reflection."""

    seed: int = 42
    """Base seed for deterministic independent tree partitions."""

    candidates_per_node: int = Field(default=3, ge=1)
    """Independent reflection and patch-generation repeats per expanded parent."""

    max_depth: int = Field(default=3, ge=1)
    """Maximum child depth; the shared initial Skill has depth zero."""

    expansion_budget: int = Field(default=30, ge=1)
    """Global number of parent expansions, including all roots."""

    expansion_concurrency: int = Field(default=40, ge=1)
    """Maximum parent nodes scheduled in each global priority wave."""

    rollout_concurrency: int = Field(default=40, ge=1)
    """Forest-wide cap on physical environment calls across all stages and trees."""

    model_concurrency: int = Field(default=40, ge=1)
    """Forest-wide cap on optimizer calls, including nested patch and Skill merges."""

    root_share_train_trajectory: bool = False
    """Reuse root validation outcomes and collect missing root training tasks once."""

    min_gain: float = Field(default=0.0, ge=0, le=1)
    """Required strict child-minus-parent validation improvement."""

    early_rejection: bool = True
    """Stop validation only when the exact remaining-score bound cannot pass."""

    merge_batch_size: int = Field(default=8, ge=2)
    """Maximum patches combined per level of within-candidate aggregation."""

    reflection_minibatch_size: int = Field(default=8, ge=1)
    """Trajectory count per homogeneous failure or success reflection minibatch."""

    reflection_edit_budget: int = Field(default=4, ge=1)
    """Maximum atomic edits retained per reflection minibatch."""

    ranking_enabled: bool = True
    """Rank merged candidate edits when they exceed the depth-specific budget."""

    ranking_edit_budgets: tuple[Annotated[int, Field(ge=1)], ...] = Field(default=(4, 3, 2), min_length=1)
    """Edit limits by parent depth; deeper parents reuse the last limit."""

    merge_threshold: float = Field(default=0.15, ge=0, le=1)
    """Keep initial merge winners within this absolute score distance of the best."""

    env_adapter: EnvAdapterSelection = Field(default_factory=EnvAdapterSelection)
    """Registered environment adaptation bound when the trainer is constructed."""

    @model_validator(mode="after")
    def validate_root_budget(self) -> Self:
        """Reserve enough global expansions to visit every root."""
        if self.expansion_budget < self.num_trees:
            raise ValueError("expansion_budget must include every root")
        return self


class SkillForestEnvAdapterConfig(BaseModel):
    """Override selected system prompts with literal text; None uses bundled defaults."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    """Keep one adapter's settings independent of run-local state."""

    analyst_error: str | None = None
    """Optional complete system prompt for the analyst_error stage."""

    analyst_success: str | None = None
    """Optional complete system prompt for the analyst_success stage."""

    merge_failure: str | None = None
    """Optional complete system prompt for the merge_failure stage."""

    merge_success: str | None = None
    """Optional complete system prompt for the merge_success stage."""

    merge_final: str | None = None
    """Optional complete system prompt for the merge_final stage."""

    ranking: str | None = None
    """Optional complete system prompt for candidate edit ranking."""

    hierarchical_merge_skills: str | None = None
    """Optional complete system prompt for the hierarchical_merge_skills stage."""
