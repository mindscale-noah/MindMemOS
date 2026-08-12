"""Canonical enum values stored by the Skill persistence schema."""

from enum import StrEnum


class AgentType(StrEnum):
    """Agent implementation that produced a trajectory."""

    CLAUDE = "claude"
    CLAUDE_SDK = "claude_sdk"
    REACT = "react"
    CODEX = "codex"
    OPENCLAW = "openclaw"
    OPENCODE = "opencode"
    GEMINI_CLI = "gemini_cli"
    CUSTOM = "custom"
    UNKNOWN = "unknown"


class SkillInjectionMode(StrEnum):
    """Runtime mechanism used to expose persisted Skills to an Agent."""

    TOOL = "tool"
    SYSTEM_PROMPT = "system_prompt"
    TREE_ROUTED_SYSTEM_PROMPT = "tree_routed_system_prompt"
    FILESYSTEM = "filesystem"


class RolloutType(StrEnum):
    """Business purpose of one planned rollout."""

    TRAIN = "train"
    EVALUATE = "evaluate"
    TEST = "test"
    INFERENCE = "inference"


class SkillVersionOrigin(StrEnum):
    """Source that created one immutable Skill version."""

    LOCAL = "local"
    CLOUD = "cloud"
    EVOLUTION = "evolution"
    MERGE = "merge"


class SkillVersionStatus(StrEnum):
    """Lifecycle state of one immutable Skill version."""

    DRAFT = "draft"
    REJECTED = "rejected"
    PUBLISHED = "published"
    ARCHIVED = "archived"


class TrajectoryStatus(StrEnum):
    """Lifecycle state of one physical rollout attempt."""

    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


__all__ = [
    "AgentType",
    "RolloutType",
    "SkillInjectionMode",
    "SkillVersionOrigin",
    "SkillVersionStatus",
    "TrajectoryStatus",
]
