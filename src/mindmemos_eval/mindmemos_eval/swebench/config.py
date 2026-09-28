"""Validated, credential-free experiment settings."""

from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class AgentConfig(BaseModel):
    """Model and bounded execution settings for one agent role."""

    model_config = ConfigDict(extra="forbid")
    """Reject misspelled settings."""

    model: str = Field(min_length=1)
    """Model identifier accepted by the configured endpoint."""

    max_turns: int = Field(default=30, ge=1)
    """Maximum model calls in one invocation of this role."""

    max_tokens: int | None = Field(default=4096, ge=1)
    """Maximum generated tokens per model call; None omits the client-side limit."""

    thinking: Literal["disabled", "enabled"] | None = None
    """Explicit provider thinking switch; None preserves the provider default."""

    temperature: float = Field(default=0.0, ge=0, le=2)
    """Sampling temperature for tool calls and responses."""

    timeout_seconds: float = Field(default=180, gt=0)
    """Timeout for a model request; failed requests are not retried."""


class MemoryConfig(BaseModel):
    """Independent project scopes for planning and execution memories."""

    model_config = ConfigDict(extra="forbid")
    """Reject unrecognized memory settings."""

    plan_project_id: str = Field(min_length=1)
    """Fresh project bound to the planning memory API key on the server."""

    experience_project_id: str = Field(min_length=1)
    """Fresh project bound to the execution memory API key on the server."""

    user_id: str = "swebench-verified"
    """Actor identity sent on memory requests; not a project isolation boundary."""

    task_top_k: int = Field(default=5, ge=1, le=10)
    """Number of matched tasks, each accompanied by its own experience list."""

    experiences_per_task: int = Field(default=100, ge=1, le=100)
    """Separate per-task experience cap, passed as API top_k."""

    timeout_seconds: float = Field(default=600, gt=0)
    """Timeout for synchronous extraction or search."""

    @model_validator(mode="after")
    def check_projects(self) -> Self:
        """Require distinct project namespaces for the two memory roles."""
        if self.plan_project_id == self.experience_project_id:
            raise ValueError("Planning and execution memory require distinct projects")
        return self


class DockerConfig(BaseModel):
    """Limits for existing local task images; never pull or build implicitly."""

    model_config = ConfigDict(extra="forbid")
    """Reject unrecognized container settings."""

    pull_missing: bool = False
    """Download missing task images explicitly when enabled."""

    remove_downloaded: bool = False
    """Remove only images downloaded by this container lifecycle after use."""

    restore_base_commit: bool = False
    """Reset tracked files to the task commit inside the disposable container."""

    platform: str = "linux/amd64"
    """Platform used for pulling and running prepared task images."""

    workdir: str = "/testbed"
    """Repository path inside every prepared task image."""

    command_prefix: str = ""
    """Optional trusted shell setup, such as activating the image's testbed environment."""

    memory_limit: str = "8g"
    """Docker memory limit for one task container."""

    cpus: float = Field(default=2, gt=0)
    """CPU quota for one task container."""

    command_timeout_seconds: int = Field(default=120, ge=1)
    """Hard timeout enforced inside the container for child shell commands."""

    output_limit_chars: int = Field(default=24000, ge=1000)
    """Maximum tool output exposed to the child per command."""


class ExperimentConfig(BaseModel):
    """Local dataset, disjoint splits, and one-attempt hierarchical rollouts."""

    model_config = ConfigDict(extra="forbid")
    """Fail on unknown experiment fields."""

    dataset_name: Literal["SWE-bench/SWE-bench_Verified"] = "SWE-bench/SWE-bench_Verified"
    """Expected upstream dataset identity; local files must be exported from it."""

    dataset_path: Path
    """Local JSONL export; no Hugging Face download occurs in this runner."""

    image_map_path: Path
    """Local JSON object mapping instance IDs to prepared Docker image names."""

    output_dir: Path
    """Exclusive experiment directory containing splits, traces, and predictions."""

    seed: int = 42
    """Seed used to shuffle sorted task IDs before choosing the two splits."""

    train_size: Literal[50] = 50
    """Exactly fifty memory-construction tasks."""

    test_size: Literal[50] = 50
    """Exactly fifty disjoint held-out tasks."""

    rollouts_per_task: Literal[1] = 1
    """One physical rollout per task; no automatic task retries."""

    task_concurrency: int = Field(default=1, ge=1, le=50)
    """Maximum concurrently active task rollouts; each task remains sequential."""

    max_delegations: int = Field(default=12, ge=1)
    """Maximum child invocations within one parent rollout."""

    parent: AgentConfig
    """Planner model settings; only todo and delegate are exposed."""

    child: AgentConfig
    """Coding model settings; only container execution is exposed."""

    memory: MemoryConfig
    """Role-specific memory storage and retrieval settings."""

    docker: DockerConfig = Field(default_factory=DockerConfig)
    """Task container resource and execution limits."""


def load_config(path: Path) -> ExperimentConfig:
    """Read settings without starting clients, loading data, or contacting Docker.

    Args:
        path: YAML configuration path. Relative data paths use the current directory.

    Returns:
        Validated experiment settings.
    """
    return ExperimentConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
