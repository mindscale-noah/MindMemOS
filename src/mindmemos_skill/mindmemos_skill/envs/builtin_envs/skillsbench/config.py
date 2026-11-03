"""Configure downloaded SkillsBench tasks and an independently installed Harbor CLI."""

from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .typing import TaskId


class SkillsBenchDataConfig(BaseModel):
    """Select tasks from the pinned source produced by the shared downloader."""

    model_config = ConfigDict(extra="forbid", use_attribute_docstrings=True)
    """Reject unsupported loader options and expose their documentation."""

    benchmark_root: Path
    """Downloaded v1.0 root containing tasks/ and download_receipt.json."""

    task_ids: tuple[TaskId, ...] | None = None
    """Exact ordered task IDs; None selects all tasks/ entries, excluding tasks-extra/."""

    @field_validator("task_ids")
    @classmethod
    def validate_task_ids(cls, value: tuple[str, ...] | None) -> tuple[str, ...] | None:
        """Reject an empty or duplicate task selection.

        Args:
            value: Exact task IDs or None for all default tasks.

        Returns:
            The unchanged, validated selection.
        """
        if value is not None and (not value or len(value) != len(set(value))):
            raise ValueError("task_ids must be nonempty and unique when supplied")
        return value


class SkillsBenchConfig(BaseModel):
    """Execute each rollout with harbor run and the original official verifier."""

    model_config = ConfigDict(extra="forbid", use_attribute_docstrings=True)
    """Reject unsupported runtime options without changing other environments."""

    harbor_command: list[str] = Field(default_factory=lambda: ["harbor"], min_length=1)
    """Literal CLI argv prefix, including an absolute executable path when needed."""

    agent: Literal["oracle", "claude-code", "codex"] = "oracle"
    """Harbor-native agent; oracle validates the task without a model call."""

    model: str | None = Field(default=None, min_length=1)
    """Explicit Harbor model name; required for a model agent and absent for oracle."""

    agent_version: str | None = Field(default=None, min_length=1)
    """Optional native agent version forwarded through Harbor's version kwarg."""

    agent_env: tuple[str, ...] = ()
    """Names of set process variables forwarded using ${NAME} references, never literal credentials."""

    timeout_s: float = Field(default=7200, gt=0, allow_inf_nan=False)
    """Outer CLI deadline in seconds; original task build, agent and verifier timeouts still apply."""

    @field_validator("harbor_command")
    @classmethod
    def validate_command(cls, value: list[str]) -> list[str]:
        """Require a literal nonempty executable and arguments.

        Args:
            value: Executable and optional literal prefix arguments.

        Returns:
            The unchanged command prefix.
        """
        if any(not item.strip() for item in value):
            raise ValueError("harbor_command entries must not be blank")
        return value

    @field_validator("agent_env")
    @classmethod
    def validate_environment_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Accept variable names rather than inline environment assignments.

        Args:
            value: Child environment variable names selected by the caller.

        Returns:
            The unchanged validated names.
        """
        if len(value) != len(set(value)) or any(not name.isidentifier() or not name.isascii() for name in value):
            raise ValueError("agent_env must contain unique environment variable names")
        return value

    @model_validator(mode="after")
    def validate_agent(self) -> Self:
        """Require an explicit model only when running a model agent."""
        if self.agent == "oracle" and (self.model is not None or self.agent_version is not None):
            raise ValueError("oracle does not accept model or agent_version")
        if self.agent != "oracle" and self.model is None:
            raise ValueError("A model agent requires an explicit model")
        return self
