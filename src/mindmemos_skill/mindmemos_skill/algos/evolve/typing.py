"""In-memory historical execution evidence for skill evolution."""

from pydantic import BaseModel, ConfigDict, JsonValue, SerializeAsAny

from ...envs.typing import Reward, Task
from ...skill import Skill


class SkillTrajectory(BaseModel):
    """Carry a historical task, its actual skill snapshots, and observed feedback.

    The caller supplies detached snapshots, not live database records or current
    replacements for historical skills. This is an in-memory algorithm input,
    not a database record or a JSON serialization schema for Skill objects.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    """Reject unknown fields and accept the project's runtime Skill objects."""

    task: SerializeAsAny[Task]
    """Task that produced this historical execution, including subtype fields."""

    skills: dict[str, Skill]
    """IDs and actual skill snapshots used together; empty for a baseline run."""

    trajectory: list[dict[str, JsonValue]]
    """Ordered interaction records, including agent messages and tool calls."""

    reward: SerializeAsAny[Reward] | None = None
    """Existing evaluation feedback; None when the execution was not scored."""
