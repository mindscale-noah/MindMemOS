"""Algorithm-facing data contracts."""

from .agent import AgentProfile, AgentType, SkillInjectionMode
from .algorithm import AlgorithmIdentity, AlgorithmLog, AlgorithmStep
from .env import EnvConfig, Environment, Reward
from .execution import AgentExecutionRequest
from .operations import (
    SkillAnalysisRequest,
    SkillAnalysisResult,
    SkillFinding,
    SkillOptimizationRequest,
    SkillOptimizationResult,
)
from .skill import (
    Skill,
    SkillBinding,
    SkillUsageType,
    SkillVersionOrigin,
    SkillVersionStatus,
    compute_skill_content_hash,
    normalize_skill_text,
    serialize_skill_files,
)
from .task import Task
from .trajectory import ExecutionInfo, Rollout, RolloutType, Trajectory, TrajectoryStatus

__all__ = [
    "AgentExecutionRequest",
    "AgentProfile",
    "AgentType",
    "AlgorithmIdentity",
    "AlgorithmLog",
    "AlgorithmStep",
    "Environment",
    "ExecutionInfo",
    "EnvConfig",
    "Reward",
    "Rollout",
    "RolloutType",
    "Skill",
    "SkillAnalysisRequest",
    "SkillAnalysisResult",
    "SkillBinding",
    "SkillFinding",
    "SkillInjectionMode",
    "SkillOptimizationRequest",
    "SkillOptimizationResult",
    "SkillUsageType",
    "SkillVersionOrigin",
    "SkillVersionStatus",
    "Task",
    "Trajectory",
    "TrajectoryStatus",
    "compute_skill_content_hash",
    "normalize_skill_text",
    "serialize_skill_files",
]
