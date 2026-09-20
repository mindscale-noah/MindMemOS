"""Local LiveMath data and single-rollout configuration."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, StrictInt

from ....agents.buitin_agents.openai_chat.config import OpenAIChatConfig
from ...data.config import SplitConfig


class LiveMathDataConfig(BaseModel):
    """Read official monthly JSON files without importing legacy split layouts."""

    model_config = ConfigDict(extra="forbid")
    """Reject unrecognized data options."""

    data_path: Path
    """One monthly JSON file or a directory containing qa_*_final.json files."""

    choice_seed: StrictInt | None = 42
    """Seed for stable per-task choice shuffling; None explicitly preserves source labels."""


class LiveMathConfig(BaseModel):
    """Execute one tool-free math response and deterministic MCQ evaluation."""

    model_config = ConfigDict(extra="forbid")
    """Reject unsupported execution options."""

    agent: OpenAIChatConfig
    """Model, endpoint, and request limits included in audit identity; credentials are masked."""

    evaluator_version: str = "skillopt-mcq-v1"
    """Evaluation protocol version included in resume identity."""

    prompt_version: str = "skillopt-math-v1"
    """Prompt protocol version included in resume identity."""


class LiveMathRunConfig(BaseModel):
    """Configure the reproducible local single-rollout entry point."""

    model_config = ConfigDict(extra="forbid")
    """Reject unknown entry-point settings."""

    data: LiveMathDataConfig
    """Local raw dataset and option-shuffle settings."""

    environment: LiveMathConfig
    """Model invocation and evaluation settings."""

    split: SplitConfig = Field(
        default_factory=lambda: SplitConfig(ratios={"train": 0.8, "val": 0.1, "test": 0.1}, seed=42)
    )
    """Shared random or ID-file split configuration."""

    split_name: str = "train"
    """Split from which the entry point selects one task."""
