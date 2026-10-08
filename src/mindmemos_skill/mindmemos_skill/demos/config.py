"""Timing and content settings for offline pipeline demonstrations."""

from pydantic import BaseModel, ConfigDict, Field

from ..pipelines.builtin_pipelines.manage.config import GraphManagePipelineConfig


class DemoAlgorithmConfig(BaseModel):
    """Control visible offline processing time and appended Skill text."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")
    """Reject unsupported options, including model credentials."""

    delay_seconds: float = Field(default=1.0, ge=0, le=30)
    """Pause per training sample, or once during evolution, in seconds."""

    note: str = Field(default="Verify the demo output before returning it.", min_length=1)
    """Instruction appended to a copy of the input Skill."""


class DemoEnvironmentConfig(BaseModel):
    """Configure an environment with deterministic synthetic execution."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")
    """Reject agent, model, and other unsupported settings."""

    delay_seconds: float = Field(default=0.2, ge=0, le=30)
    """Pause for each synthetic rollout, in seconds."""


class DemoGraphConfig(GraphManagePipelineConfig):
    """Retain graph query behavior and add a deterministic publication delay."""

    delay_seconds: float = Field(default=0.2, ge=0, le=30)
    """Pause for each resolved graph member, in seconds."""
