"""Logical task contracts."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, JsonValue


class Task(BaseModel):
    """Logical task identity and input submitted to an Agent backend.

    Runtime-specific context belongs to ``Environment``; rollout and retry
    identity belong to ``Rollout``. Keeping them separate lets algorithms
    group different attempts of the same task without parsing metadata.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    task_id: str = Field(min_length=1)
    """任务标识"""

    instruction: str = Field(min_length=1)
    """任务指令"""

    system_prompt: str | None = None
    """系统提示词"""

    tags: list[str] = Field(default_factory=list)
    """任务标签。"""

    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    """任务元数据"""


__all__ = ["Task"]
