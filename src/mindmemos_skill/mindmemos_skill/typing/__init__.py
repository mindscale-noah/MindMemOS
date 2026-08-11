"""Algorithm-facing data contracts."""

from .agent import AgentProfile, AgentType, SkillInjectionMode
from .algorithm import AlgorithmIdentity, AlgorithmLog, AlgorithmStep
from .env import EnvConfig, Environment, Reward
from .execution import AgentExecutionRequest
from .operations import (
    EvolveInput,
    EvolveOutput,
    SkillAnalysisRequest,
    SkillAnalysisResult,
    SkillCandidate,
    SkillFinding,
    Trace2SkillInput,
    Trace2SkillOutput,
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
    "EvolveInput",
    "EvolveOutput",
    "Rollout",
    "RolloutType",
    "Skill",
    "SkillAnalysisRequest",
    "SkillAnalysisResult",
    "SkillBinding",
    "SkillCandidate",
    "SkillFinding",
    "SkillInjectionMode",
    "SkillUsageType",
    "SkillVersionOrigin",
    "SkillVersionStatus",
    "Task",
    "Trajectory",
    "TrajectoryStatus",
    "Trace2SkillInput",
    "Trace2SkillOutput",
    "compute_skill_content_hash",
    "normalize_skill_text",
    "serialize_skill_files",
]
