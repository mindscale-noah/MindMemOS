"""Prepared local benchmark inputs and shareable run records."""

from pathlib import Path
from typing import Literal

from mindmemos_skill.api.typing import TrainingSplit
from mindmemos_skill.config.config import SkillConfig
from mindmemos_skill.envs.data.typing import DataSample
from pydantic import BaseModel, ConfigDict, Field

from .config import LocalSkillEvalConfig


class PreparedInputs(BaseModel):
    """Keep validated samples and their private scoring inputs in memory."""

    config: LocalSkillEvalConfig
    """Resolved and validated single-run configuration."""

    service: SkillConfig
    """Validated local service definition used to construct the service."""

    samples: list[DataSample]
    """Selected task and private reward data in split order."""

    split: TrainingSplit
    """Complete selected train, val, and test task membership."""

    config_path: Path
    """Absolute source path of the experiment YAML."""

    input_digest: str
    """Digest of configuration, selected data, split, and initial Skill files."""


class LocalRunManifest(BaseModel):
    """Track one recoverable benchmark without storing private task data."""

    model_config = ConfigDict(extra="forbid")
    """Reject unknown recovery fields."""

    input_digest: str
    """Immutable run identity computed during preflight."""

    config_path: str
    """Absolute path to the experiment definition."""

    project_id: str
    """Local Service project scope."""

    algorithm: str
    """Selected training pipeline."""

    environment: str
    """Selected execution environment implementation."""

    train_ids: list[str]
    """Ordered train task IDs."""

    val_ids: list[str]
    """Ordered validation task IDs."""

    test_ids: list[str]
    """Ordered held-out test task IDs."""

    skill_id: str | None = None
    """Registered Skill identity, or None before registration."""

    base_version_id: str | None = None
    """Exact parent version, or None before registration."""

    published_version_id: str | None = None
    """Exact published child version, or None before successful training."""

    status: Literal["prepared", "registered", "published", "completed", "failed"] = "prepared"
    """Current run state; failed runs retain exact version IDs for a compatible retry."""


class SkillEvalReport(BaseModel):
    """Report the completed pipeline test without running another evaluation."""

    model_config = ConfigDict(extra="forbid")
    """Reject undeclared report fields."""

    project_id: str
    """Local Service project scope."""

    skill_id: str
    """Trained Skill identity."""

    base_version_id: str
    """Parent version used for training."""

    version_id: str
    """Published selected Skill version."""

    algorithm: str
    """Training pipeline name."""

    environment: str
    """Execution environment implementation name."""

    test_count: int = Field(ge=0)
    """Number of completed held-out test tasks."""

    mean_score: float | None
    """Average of reported environment scores, or None when none completed."""

    success_rate: float | None
    """Fraction of completed tests reported successful by the environment."""

    test_task_ids: list[str]
    """Ordered IDs included in the pipeline's completed test index."""

    duration_s: float | None
    """Sum of reported rollout durations, or None when any duration is missing."""

    usage: dict[str, int] | None
    """Summed provider token counters, or None when no rollout reported usage."""
