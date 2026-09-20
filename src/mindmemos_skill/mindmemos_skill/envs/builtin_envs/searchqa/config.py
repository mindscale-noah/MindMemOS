"""SearchQA source and rollout settings."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from ....agents.buitin_agents.openai_chat.config import OpenAIChatConfig


class SearchQADataConfig(BaseModel):
    """Load questions and passages from a local JSON or JSONL release."""

    model_config = ConfigDict(extra="forbid")
    """Reject unsupported source settings."""

    data_path: Path
    """Local source file containing question, context, and answers."""


class SearchQAConfig(BaseModel):
    """Configure passage-grounded answering and self-refinement."""

    model_config = ConfigDict(extra="forbid")
    """Reject unsupported execution settings."""

    agent: OpenAIChatConfig
    """Audited model and transport settings."""

    max_turns: int = Field(default=1, ge=1)
    """Maximum model calls, without private reference feedback."""

    max_context_chars: int = Field(default=6000, ge=1)
    """Passage character budget, truncated at DOC boundaries."""

    evaluator_version: str = "skillopt-searchqa-v1"
    """Identity of the exact-match, token F1, and feedback protocol."""
