"""Canonical string values used by the Skill application layer."""

from enum import StrEnum


class SkillApplicationCapability(StrEnum):
    """Operation exposed by :class:`SkillApplication`."""

    ANALYZE = "analyze"
    DIFF = "diff"
    EVOLVE = "evolve"
    EXECUTE = "execute"
    EXPORT = "export"
    LIST = "list"
    OPTIMIZE = "optimize"
    PUBLISH = "publish"
    PULL = "pull"
    PUSH = "push"
    REGISTER = "register"
    SHOW = "show"
    SYNC = "sync"
    UNREGISTER = "unregister"


class AlgorithmResultStatus(StrEnum):
    """Status recorded for an application-managed algorithm result."""

    SUCCEEDED = "succeeded"


__all__ = ["AlgorithmResultStatus", "SkillApplicationCapability"]
