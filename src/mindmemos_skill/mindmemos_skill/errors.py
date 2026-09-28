"""Business errors for the MindMemOS Skill package."""


class MindMemosSkillError(Exception):
    """Base class for Skill business errors, excluding generic Python errors."""


class SkillNotFoundError(MindMemosSkillError):
    """The requested Skill or owned version does not exist."""


class SkillConflictError(MindMemosSkillError):
    """A reference is ambiguous, an alias conflicts, or the current version moved."""


class SkillInUseError(SkillConflictError):
    """Stored evidence or another object still references this Skill."""


class UnsupportedPipelineError(MindMemosSkillError):
    """The selected pipeline cannot run in this deployment."""
