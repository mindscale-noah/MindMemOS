"""Configuration for one local Skill training benchmark run."""

from pathlib import Path
from typing import Any, Literal

from mindmemos_skill.config.config import ImplementationConfig
from mindmemos_skill.envs.config import EnvironmentConfig
from pydantic import BaseModel, ConfigDict, Field


class InitialSkillConfig(BaseModel):
    """Select the Skill source registered before training."""

    model_config = ConfigDict(extra="forbid")
    """Reject undeclared source options."""

    path: Path
    """Directory containing SKILL.md or a direct path to that file."""

    name: str = Field(min_length=1)
    """Display name for the newly registered Skill identity."""


class SplitLimits(BaseModel):
    """Optionally take an ordered prefix of each supplied split."""

    model_config = ConfigDict(extra="forbid")
    """Reject unsupported split limits."""

    train: int | None = Field(default=None, ge=1)
    """Maximum train tasks; None keeps all supplied train IDs."""

    val: int | None = Field(default=None, ge=0)
    """Maximum final validation tasks; None keeps all supplied val IDs."""

    test: int | None = Field(default=None, ge=1)
    """Maximum test tasks; None keeps all supplied test IDs."""


class LocalSkillEvalConfig(BaseModel):
    """Describe one algorithm, environment, dataset, and initial Skill run."""

    model_config = ConfigDict(extra="forbid", use_attribute_docstrings=True)
    """Reject unknown experiment options."""

    service_config: Path
    """Local Skill service YAML containing storage and training profiles."""

    algorithm: Literal["treeskill", "skill_forest"]
    """Configured training pipeline name and expected trainer implementation."""

    environment: EnvironmentConfig
    """Execution environment implementation and options for this experiment."""

    data: ImplementationConfig
    """Built-in dataset loader name and its source options."""

    initial_skill: InitialSkillConfig
    """Source used to register an independent parent Skill version."""

    split_file: Path
    """JSON mapping train, val, and test to ordered task IDs."""

    limits: SplitLimits = Field(default_factory=SplitLimits)
    """Optional deterministic prefix lengths for each split."""


def local_skill_eval_schema() -> dict[str, Any]:
    """Generate experiment hints while reusing the registered environment config models.

    Returns:
        JSON Schema for one experiment with environment-specific field completions.
    """
    from mindmemos_skill.config.schema import environment_config_schema

    schema = LocalSkillEvalConfig.model_json_schema()
    environment = environment_config_schema()
    schema["$defs"].update(environment.pop("$defs"))
    environment.pop("$schema")
    schema["properties"]["environment"] = environment
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    return schema
