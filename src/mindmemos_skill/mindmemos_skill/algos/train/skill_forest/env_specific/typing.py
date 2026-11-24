"""Validated optional diagnostics retained by unified environment rewards."""

from pydantic import BaseModel, Field


class ReflectionFeedback(BaseModel):
    """Read environment diagnostics without depending on private reward inputs."""

    fail_reason: str = ""
    """Original evaluator failure explanation, empty after success."""

    n_turns: int | None = Field(default=None, ge=0)
    """Recorded model turn count; None for older results without this diagnostic."""

    task_description: str | None = None
    """Simulator task text recorded at reset, when available."""

    task_type: str = ""
    """Simulator category recorded by the environment, when available."""
