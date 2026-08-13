"""Configuration for Tree Skill GRPO."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, JsonValue


class TreeSkillGrpoConfig(BaseModel):
    """Application-owned runtime settings for one Tree Skill GRPO run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str = "1"
    env_ref: str = Field(min_length=1)
    agent_ref: str = Field(default="react", min_length=1)
    env_options: dict[str, JsonValue] = Field(default_factory=dict)
    agent_options: dict[str, JsonValue] = Field(default_factory=dict)
    workspace_root: Path | None = None
    checkpoint_dir: Path | None = None
    epochs: int = Field(default=1, ge=1)
    batch_size: int = Field(default=40, ge=1)
    max_task_attempts: int = Field(default=4, ge=1)
    max_concurrency: int = Field(default=32, ge=1)


__all__ = ["TreeSkillGrpoConfig"]
