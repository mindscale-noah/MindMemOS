"""Business errors for the MindMemOS Skill package."""


class MindMemosSkillError(Exception):
    """Base class for Skill business errors, excluding generic Python errors."""


class SkillNotFoundError(MindMemosSkillError):
    """The requested Skill or owned version does not exist."""


class SkillVersionNotFoundError(SkillNotFoundError):
    """The requested exact Skill version does not exist in this project."""


class TrajectoryNotFoundError(SkillNotFoundError):
    """The requested exact trajectory does not exist in this project."""


class SkillConflictError(MindMemosSkillError):
    """A reference is ambiguous, an alias conflicts, or the current version moved."""


class SkillAliasConflictError(SkillConflictError):
    """A requested alias already identifies a different Skill."""


class SkillAmbiguousVersionError(SkillConflictError):
    """A source content hash identifies more than one Skill version."""


class SkillInUseError(SkillConflictError):
    """Stored evidence or another object still references this Skill."""


class GraphNotFoundError(MindMemosSkillError):
    """The requested graph or snapshot does not exist in the selected project."""


class GraphConflictError(MindMemosSkillError):
    """A graph reference has multiple matches within the selected project."""


class UnsupportedPipelineError(MindMemosSkillError):
    """The selected pipeline cannot run in this deployment."""


class ProjectTaskBusyError(MindMemosSkillError):
    """A project has reached its limit for queued and running algorithm tasks."""

    def __init__(self, task_id: str, kind: str, status: str):
        """Identify the existing task within the caller's authorized project.

        Args:
            task_id: Identity of the task holding the slot.
            kind: Trusted Skill task name.
            status: Current queued or running state.
        """
        self.task_id = task_id
        self.kind = kind
        self.status = status
        service = kind.removeprefix("skill.").replace("_", " ").capitalize()
        super().__init__(f"{service} already has a {status} task in this project (task ID: {task_id}).")
