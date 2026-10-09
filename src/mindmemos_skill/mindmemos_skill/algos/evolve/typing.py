"""In-memory historical execution evidence for skill evolution."""

from pydantic import BaseModel, ConfigDict, Field, JsonValue, SerializeAsAny

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

    trajectory_id: str | None = None
    """Stored execution identity used to trace patch evidence; None uses a local index."""

    task: SerializeAsAny[Task]
    """Task that produced this historical execution, including subtype fields."""

    task_system_prompt: str | None = None
    """System instruction active during the historical execution, if stored."""

    task_metadata: dict[str, JsonValue] = Field(default_factory=dict)
    """Stored task context needed to interpret the execution evidence."""

    skills: dict[str, Skill]
    """IDs and actual skill snapshots used together; empty for a baseline run."""

    trajectory: list[dict[str, JsonValue]]
    """Ordered interaction records, including agent messages and tool calls."""

    reward: SerializeAsAny[Reward] | None = None
    """Existing evaluation feedback; None when the execution was not scored."""

    status: str | None = None
    """Recorded execution outcome, or None when the source omitted it."""

    observed_score: float | None = None
    """Stored environment score, without inferring success from its scale."""

    feedback_detail: str | None = None
    """Stored evaluation explanation, or None when no explanation was recorded."""
