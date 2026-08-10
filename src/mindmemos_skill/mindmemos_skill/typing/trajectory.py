"""Aggregated trajectory contracts consumed by Skill algorithms."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from ..persistence.enums import RolloutType, TrajectoryStatus
from .agent import AgentProfile
from .env import Environment, Reward
from .skill import Skill, SkillBinding
from .task import Task

if TYPE_CHECKING:
    from ..persistence.models import TrajectoryRecord


class Rollout(BaseModel):
    """Stable rollout identity plus the current retry attempt."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    rollout_id: str = Field(min_length=1)
    """一次计划 rollout 的稳定标识；其所有重试共享该值。"""

    attempt_no: int = Field(default=0, ge=0)
    """当前物理尝试的序号，首次执行为 0。"""

    rollout_type: RolloutType = RolloutType.INFERENCE
    """本次 rollout 的训练、评估、测试或推理用途。"""


class ExecutionInfo(BaseModel):
    """Runtime outcome of one physical trajectory attempt."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    status: TrajectoryStatus = TrajectoryStatus.RUNNING
    """当前 attempt 的执行状态。"""

    started_at: datetime
    """执行开始时间。"""

    finished_at: datetime | None = None
    """执行结束时间；尚未结束时为空。"""

    n_turn: int = Field(default=0, ge=0)
    """Agent 交互轮数。"""

    error_info: str | None = None
    """执行失败、工具异常或环境错误信息。"""

    @property
    def duration_s(self) -> float | None:
        """Duration derived from the two timestamps when the attempt finished."""

        if self.finished_at is None:
            return None
        return (self.finished_at - self.started_at).total_seconds()

    @model_validator(mode="after")
    def validate_timestamps(self) -> ExecutionInfo:
        if self.finished_at is not None and self.finished_at < self.started_at:
            raise ValueError("finished_at must not be earlier than started_at")
        return self


class Trajectory(BaseModel):
    """Algorithm-facing aggregate for one physical Agent execution attempt.

    Persistence stores this aggregate as one flattened ``TrajectoryRecord``:
    task, rollout, environment, Agent, reward and execution fields become
    columns, while events and Skill snapshots remain JSON columns.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    trajectory_id: str = Field(min_length=1)
    """某一次实际执行 attempt 的唯一标识。"""

    task: Task
    """逻辑任务及其输入上下文。"""

    rollout: Rollout
    """计划 rollout 与重试信息。"""

    environment: Environment = Field(default_factory=Environment)
    """任务运行目录和环境元数据。"""

    agent: AgentProfile = Field(default_factory=AgentProfile)
    """执行轨迹的 Agent 类型及可复现配置。"""

    injected_skills: list[Skill] = Field(default_factory=list)
    """执行开始前提供给 Agent 的不可变 Skill 版本快照。"""

    events: list[dict[str, JsonValue]] = Field(default_factory=list)
    """按发生顺序记录的消息、工具调用和内部事件。"""

    skill_bindings: list[SkillBinding] = Field(default_factory=list)
    """本次执行实际使用到的 Skill 版本引用。"""

    reward: Reward = Field(default_factory=Reward)
    """结构化评分；未评估时 ``score`` 为空。"""

    execution: ExecutionInfo
    """该物理 attempt 的状态、时间、轮数和错误。"""

    metadata: dict[str, Any] = Field(default_factory=dict)
    """数据集、采集器和算法附加信息。"""

    def to_record(self) -> TrajectoryRecord:
        """Flatten the aggregate into the canonical persistence row."""

        from ..persistence.models import TrajectoryRecord

        created_at = self.execution.finished_at or self.execution.started_at
        source_payload = {
            "trajectory_id": self.trajectory_id,
            "task_id": self.task.task_id,
            "rollout_id": self.rollout.rollout_id,
            "attempt_no": self.rollout.attempt_no,
            "rollout_type": self.rollout.rollout_type.value,
            "task_instruction": self.task.instruction,
            "task_system_prompt": self.task.system_prompt,
            "task_tags": self.task.tags,
            "task_metadata": self.task.metadata,
            "env_metadata": self.environment.metadata,
            "agent_type": self.agent.agent_type.value,
            "agent_profile": self.agent.model_dump(mode="json", exclude={"agent_type"}, exclude_none=True),
            "status": self.execution.status.value,
            "trajectory": self.events,
            "skill_bindings": [binding.model_dump(mode="json") for binding in self.skill_bindings],
            "reward_score": self.reward.score,
            "reward_detail": self.reward.detail,
            "reward_metadata": self.reward.metadata,
            "started_at": self.execution.started_at.isoformat(),
            "finished_at": self.execution.finished_at.isoformat() if self.execution.finished_at else None,
            "n_turn": self.execution.n_turn,
            "error_info": self.execution.error_info,
            "metadata": self.metadata,
            "source": "skill_runtime",
            "created_at": created_at.isoformat(),
        }
        trajectory_hash = hashlib.sha256(
            json.dumps(source_payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
        ).hexdigest()
        return TrajectoryRecord(
            trajectory_id=self.trajectory_id,
            trajectory_hash=trajectory_hash,
            task_id=self.task.task_id,
            rollout_id=self.rollout.rollout_id,
            attempt_no=self.rollout.attempt_no,
            rollout_type=self.rollout.rollout_type,
            task_instruction=self.task.instruction,
            task_system_prompt=self.task.system_prompt,
            task_tags=self.task.tags,
            task_metadata=self.task.metadata,
            running_dir=self.environment.running_dir,
            env_metadata=self.environment.metadata,
            injected_skills=[skill.model_dump(mode="json") for skill in self.injected_skills],
            agent_type=self.agent.agent_type,
            agent_profile=self.agent.model_dump(
                mode="json",
                exclude={"agent_type"},
                exclude_none=True,
            ),
            status=self.execution.status,
            trajectory=self.events,
            skill_bindings=[binding.model_dump(mode="json") for binding in self.skill_bindings],
            reward_score=self.reward.score,
            reward_detail=self.reward.detail,
            reward_metadata=self.reward.metadata,
            started_at=self.execution.started_at,
            finished_at=self.execution.finished_at,
            n_turn=self.execution.n_turn,
            error_info=self.execution.error_info,
            metadata=self.metadata,
            source="skill_runtime",
            created_at=created_at,
        )

    @classmethod
    def from_record(cls, record: TrajectoryRecord) -> Trajectory:
        """Rebuild the business aggregate from a validated persistence row."""

        return cls(
            trajectory_id=record.trajectory_id,
            task=Task(
                task_id=record.task_id,
                instruction=record.task_instruction,
                system_prompt=record.task_system_prompt,
                tags=record.task_tags,
                metadata=record.task_metadata,
            ),
            rollout=Rollout(
                rollout_id=record.rollout_id,
                attempt_no=record.attempt_no,
                rollout_type=record.rollout_type,
            ),
            environment=Environment(running_dir=record.running_dir, metadata=record.env_metadata),
            agent=AgentProfile.from_serialized(
                record.agent_profile,
                agent_type=record.agent_type,
            ),
            injected_skills=[Skill.model_validate(skill) for skill in record.injected_skills],
            events=record.trajectory,
            skill_bindings=[SkillBinding.model_validate(binding) for binding in record.skill_bindings],
            reward=Reward(
                score=record.reward_score,
                detail=record.reward_detail,
                metadata=record.reward_metadata,
            ),
            execution=ExecutionInfo(
                status=record.status,
                started_at=record.started_at,
                finished_at=record.finished_at,
                n_turn=record.n_turn,
                error_info=record.error_info,
            ),
            metadata=record.metadata,
        )


__all__ = [
    "ExecutionInfo",
    "Rollout",
    "RolloutType",
    "Trajectory",
    "TrajectoryStatus",
]
