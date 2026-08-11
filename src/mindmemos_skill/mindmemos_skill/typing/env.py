"""Environment and evaluation contracts for algorithm trajectories."""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator


class EnvConfig(BaseModel):
    """Validated construction-time configuration shared by environments."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class Environment(BaseModel):
    """Execution environment attached to one task rollout."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    env_ref: str = Field(default="unknown", min_length=1)
    """注册环境名称；旧轨迹或非注册环境使用 ``unknown``。"""

    running_dir: str | None = None
    """Agent 执行任务时使用的工作目录。"""

    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    """数据集环境、沙箱或工具运行时的可序列化信息。"""


class Reward(BaseModel):
    """Structured evaluation result for one trajectory."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    score: float | None = None
    """奖励分数；尚未评分时为空。"""

    detail: str | None = None
    """测评详情"""

    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    """测评metadata"""

    @field_validator("score")
    @classmethod
    def validate_finite_score(cls, value: float | None) -> float | None:
        if value is None:
            return None
        if not math.isfinite(value):
            raise ValueError("score must be a finite number")
        return value


__all__ = ["EnvConfig", "Environment", "Reward"]
