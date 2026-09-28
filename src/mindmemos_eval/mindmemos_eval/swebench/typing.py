"""Persisted task, split, trajectory, and tool argument contracts."""

from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator

AgentRole = Literal["parent", "child"]


class Task(BaseModel):
    """Allowlisted task input, deliberately excluding reference and test patches."""

    model_config = ConfigDict(extra="ignore")
    """Drop upstream grading-only fields before creating agent inputs."""

    instance_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    """Official instance identifier, safe to use as a directory name."""

    repo: str = Field(min_length=1)
    """Repository owner/name used for task context."""

    base_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    """Exact repository revision that the local image must contain."""

    problem_statement: str = Field(min_length=1)
    """Issue description supplied to both agent roles."""

    @field_validator("instance_id")
    @classmethod
    def check_instance_id(cls, value: str) -> str:
        """Reject special relative directory components."""
        if value in {".", ".."}:
            raise ValueError("Invalid instance ID")
        return value


class SplitManifest(BaseModel):
    """Frozen selection with dataset provenance and disjoint task IDs."""

    dataset_name: str
    """Claimed identity of the locally supplied dataset."""

    dataset_sha256: str
    """Digest of the source JSONL bytes used for this split."""

    seed: int
    """Random seed used for selection."""

    train: list[Task] = Field(min_length=50, max_length=50)
    """Tasks whose one-shot traces may be added to memory."""

    test: list[Task] = Field(min_length=50, max_length=50)
    """Tasks whose traces must never be added to experiment memory."""

    @model_validator(mode="after")
    def validate_disjoint(self) -> Self:
        """Reject overlap and duplicates across all selected tasks."""
        ids = [task.instance_id for task in self.train + self.test]
        if len(set(ids)) != 100:
            raise ValueError("The 50/50 split must contain 100 unique tasks")
        return self


class TodoItem(BaseModel):
    """One explicit planning item owned by the parent."""

    model_config = ConfigDict(extra="forbid")
    """Accept only the documented todo fields."""

    id: str = Field(min_length=1)
    """Stable item identifier referenced in delegation calls."""

    task: str = Field(min_length=1)
    """Work to be performed by a child agent."""

    status: Literal["pending", "in_progress", "completed"]
    """Parent-reported progress, not an official benchmark verdict."""


class TodoArgs(BaseModel):
    """Arguments for replacing the parent's todo list."""

    model_config = ConfigDict(extra="forbid")
    """Reject undeclared tool arguments."""

    items: list[TodoItem] = Field(min_length=1)
    """Complete current plan, replacing the previous list."""

    @field_validator("items")
    @classmethod
    def unique_ids(cls, items: list[TodoItem]) -> list[TodoItem]:
        """Require an unambiguous ID for each planned task."""
        if len({item.id for item in items}) != len(items):
            raise ValueError("Todo IDs must be unique")
        return items


class DelegateArgs(BaseModel):
    """Arguments for invoking a fresh coding child."""

    model_config = ConfigDict(extra="forbid")
    """Reject undeclared delegation arguments."""

    todo_id: str = Field(min_length=1)
    """Existing todo item assigned to this child invocation."""

    task: str = Field(min_length=1)
    """Self-contained instruction including any relevant findings and constraints."""


class ShellArgs(BaseModel):
    """Arguments for running commands exclusively inside the task container."""

    model_config = ConfigDict(extra="forbid")
    """Reject undeclared shell arguments."""

    command: str = Field(min_length=1)
    """Shell program used to inspect, edit, or test the repository."""


class Trajectory(BaseModel):
    """One agent invocation with native tool-call messages preserved."""

    trajectory_id: str
    """Local invocation identifier, unique within its benchmark task."""

    role: AgentRole
    """Parent planner or child executor."""

    task: str
    """Instruction supplied to this agent invocation."""

    metadata: dict[str, str]
    """Extraction mode, instance identity, and parent-child linkage."""

    messages: list[dict[str, Any]] = Field(default_factory=list)
    """Original messages, including arguments and outputs but excluding search injection."""

    finished: bool = False
    """Whether the agent returned a terminal answer within its turn limit."""

    error: str | None = None
    """Fatal provider or infrastructure failure, if any."""

    @computed_field
    @property
    def agent(self) -> str:
        """Return the persisted identity used in memory extraction.

        Returns:
            Main-agent identity or the numbered coding-agent identity within this task.
        """
        return "main" if self.role == "parent" else f"coding-agent-{self.trajectory_id.removeprefix('child-')}"
