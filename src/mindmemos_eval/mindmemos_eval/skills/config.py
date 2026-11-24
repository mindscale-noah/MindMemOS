"""Shared model, forest, environment, and command configuration."""

from pathlib import Path
from typing import Literal, Self

import yaml
from mindmemos_skill.agents import OpenAIChatConfig
from mindmemos_skill.agents.buitin_agents.claude.config import ClaudeConfig
from mindmemos_skill.algos.train.skill_forest.config import ForestOptimizerConfig, SkillForestTrainerConfig
from mindmemos_skill.config.config import ImplementationConfig
from mindmemos_skill.envs import Environment
from mindmemos_skill.envs.config import EnvironmentConfig
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from .typing import ArmName, BenchmarkName

ROOT = Path(__file__).resolve().parents[4]
"""Repository containing committed resources and the default shared configuration."""

EVAL_CONFIGS = ROOT / "config/mindmemos_eval"
"""Default YAML directory for Skill evaluation commands."""

ENVIRONMENT_NAMES = {
    "livemath": "livemath",
    "alfworld": "alfworld_bounded_history",
    "spreadsheetbench": "spreadsheetbench_tool_calling",
    "searchqa": "searchqa",
    "officeqa": "officeqa",
}
"""Explicit benchmark-to-environment selections; no eval registration mechanism."""


class SkillGraphAccuracyConfig(BaseModel):
    """Configure the AgentSkillOS execution arm of the Skill Graph benchmark."""

    model_config = ConfigDict(extra="forbid")
    """Reject unsupported experiment settings."""

    benchmark_root: Path
    """Original AgentSkillOS checkout containing task assets and evaluator code."""

    agent: ClaudeConfig
    """Tool-using executor and model connection for the Setting 1 arm."""

    execution_top_k: int = Field(default=5, ge=1)
    """Maximum number of ranked Skills injected per task."""

    concurrency: int = Field(default=1, ge=1)
    """Maximum simultaneous task executions."""


class SkillGraphRunOptions(BaseModel):
    """Resolve inputs shared by the Skill Graph evaluation modes."""

    env: Literal["agentskillos"] = "agentskillos"
    """Registered environment supported by both Skill Graph modes."""

    eval_mode: Literal["retrieval", "accuracy"]
    """Score rankings against labels or score executed tasks."""

    tasks_dir: Path | None = None
    """Original task JSON directory used for retrieval scoring."""

    skills_dir: Path
    """Skill packages to index and rank."""

    service_config: Path = ROOT / "config/mindmemos_skill/dev.yaml"
    """Local Skill service configuration for indexing and retrieval."""

    output_dir: Path
    """Fresh directory for the graph, rankings, and results."""


class ModelConfig(BaseModel):
    """Configure target answering and optimizer connections independently."""

    model_config = ConfigDict(extra="forbid")
    """Reject misspelled connection groups."""

    target: OpenAIChatConfig
    """Answering transport injected into each environment."""

    optimizer: ForestOptimizerConfig
    """Existing forest optimizer model with its native validation."""


class EvalConfig(BaseModel):
    """Validate shared settings using implementation-owned models."""

    model_config = ConfigDict(extra="forbid")
    """Reject unsupported shared configuration fields."""

    models: ModelConfig
    """Target and optimizer model connections."""

    skill_forest: dict[str, JsonValue]
    """Existing trainer fields excluding optimizer, supplied by models.optimizer."""

    evaluation_concurrency: int = Field(default=4, ge=1)
    """Maximum simultaneous held-out environment calls."""

    environments: dict[BenchmarkName, dict[str, JsonValue]] = Field(default_factory=dict)
    """Native environment overrides keyed by benchmark; unknown fields are rejected."""

    output_dir: Path | None = Field(default=None, exclude=True)
    """YAML output directory. The run records the resolved path in RunOptions."""

    @model_validator(mode="after")
    def validate_implementations(self) -> Self:
        """Validate forest and environment settings without starting runtimes."""
        if "optimizer" in self.skill_forest:
            raise ValueError("Configure optimizer in models.optimizer")
        self.forest()
        for benchmark, overrides in self.environments.items():
            if "agent" in overrides or "simulator" in overrides:
                raise ValueError("Environment agent and simulator are supplied by the runner")
            options = {**overrides, "agent": self.models.target}
            if benchmark == "alfworld":
                options["simulator"] = {}
            Environment.resolve(ENVIRONMENT_NAMES[benchmark]).config_type.model_validate(options)
        return self

    def forest(self, smoke: bool = False) -> SkillForestTrainerConfig:
        """Build the native trainer configuration.

        Args:
            smoke: Reduce trees, candidates, depth, and expansion budget to one.

        Returns:
            Validated native forest settings with the shared optimizer.
        """
        options = dict(self.skill_forest)
        if smoke:
            options.update(num_trees=1, candidates_per_node=1, max_depth=1, expansion_budget=1)
        return SkillForestTrainerConfig.model_validate({**options, "optimizer": self.models.optimizer})

    def environment(self, benchmark: BenchmarkName, prepared: dict[str, JsonValue]) -> EnvironmentConfig:
        """Build a benchmark's final execution configuration.

        Args:
            benchmark: Supported benchmark name.
            prepared: Runtime inputs such as the resolved ALFWorld simulator YAML.

        Returns:
            Native environment selection with target connection and explicit overrides.
        """
        return EnvironmentConfig(
            class_name=ENVIRONMENT_NAMES[benchmark],
            config={**prepared, **self.environments.get(benchmark, {}), "agent": self.models.target},
        )


class RunOptions(BaseModel):
    """Validate only the supported command options."""

    model_config = ConfigDict(extra="forbid")
    """Reject unsupported execution modes."""

    benchmark: BenchmarkName = "livemath"
    """Dataset used for training and held-out evaluation."""

    algorithm: Literal["skill_forest"] = "skill_forest"
    """Only currently implemented algorithm."""

    smoke: bool = False
    """Use ordered 4/2/2 split prefixes and a minimal forest."""

    test_repeats: int = Field(default=1, ge=1)
    """Independent repetitions on identical test IDs and frozen versions."""

    arms: list[ArmName] = Field(default_factory=lambda: ["no_skill", "initial_skill", "trained_skill"], min_length=1)
    """Requested conditions; training runs only when trained_skill is selected."""

    data_root: Path = ROOT / "data/mindmemos_skill"
    """Benchmark parent directory; explicit CLI paths are relative to the caller."""

    output_dir: Path | None = None
    """Fresh run directory; None selects an automatic timestamp and UUID."""

    @model_validator(mode="after")
    def validate_arms(self) -> Self:
        """Reject repeated arms rather than silently executing them twice."""
        if len(set(self.arms)) != len(self.arms):
            raise ValueError("arms must be unique")
        return self


def load_config(path: Path, *, output_dir: Path | None = None) -> EvalConfig:
    """Read one shared YAML configuration.

    Args:
        path: Configuration path relative to the caller or absolute.
        output_dir: CLI output directory override, if supplied.

    Returns:
        Validated model, forest, and environment options.

    Raises:
        OSError: The file cannot be read.
        ValueError: YAML or native configuration validation fails.
    """
    payload = read_evaluation_yaml(path)
    if output_dir is not None:
        payload["output_dir"] = output_dir
    return EvalConfig.model_validate(payload)


def read_evaluation_yaml(path: Path) -> dict[str, object]:
    """Read an evaluation YAML file as a field mapping.

    Args:
        path: Evaluation configuration file.

    Returns:
        YAML keys and values for the selected evaluator's config model.

    Raises:
        ValueError: The YAML root is not a mapping.
    """
    payload = yaml.safe_load(path.expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return payload


def read_environment_yaml(path: Path, *, env: str | None, output_dir: Path | None) -> dict[str, object]:
    """Read YAML with explicit environment and output directory overrides.

    Args:
        path: Evaluation YAML file.
        env: Registered environment name supplied on the CLI.
        output_dir: Output directory supplied on the CLI.

    Returns:
        YAML fields for the selected evaluator's config model.
    """
    payload = read_evaluation_yaml(path)
    if env is not None:
        environment = payload.setdefault("environment", {})
        if not isinstance(environment, dict):
            raise ValueError("environment must be a YAML mapping")
        environment["class_name"] = env
    if output_dir is not None:
        payload["output_dir"] = output_dir
    return payload


class EnvTestConfig(BaseModel):
    """Configure direct registered-environment tests without a trainer or service."""

    model_config = ConfigDict(extra="forbid")
    """Reject unknown options before executing tasks."""

    environment: EnvironmentConfig
    """Registered environment name and its native, validated configuration."""

    data: ImplementationConfig | None = None
    """Registered Data implementation and source options; exclusive with samples_file."""

    samples_file: Path | None = None
    """JSON array of task/reward_data objects validated by the selected environment."""

    split_file: Path | None = None
    """Optional ordered split manifest; absent means use all loaded samples."""

    split: str = "test"
    """Partition selected from split_file when a manifest is provided."""

    limit: int | None = Field(default=None, ge=1)
    """Optional ordered prefix length applied after split selection."""

    skills: list[Path] = Field(default_factory=list)
    """Static Skill directories or SKILL.md paths, injected together in the given order."""

    skill_body_only: bool = False
    """Inject only the Markdown body when reproducing legacy plain-text Skills."""

    concurrency: int = Field(default=1, ge=1)
    """Maximum simultaneous task rollouts."""

    output_dir: Path | None = None
    """Empty result directory, or None to create a unique directory automatically."""

    @model_validator(mode="after")
    def validate_source(self) -> Self:
        """Require one source and validate registered Data options before execution."""
        from mindmemos_skill.envs.data import Data

        if (self.data is None) == (self.samples_file is None):
            raise ValueError("Provide exactly one of data or samples_file")
        if self.data is not None:
            implementation = Data.resolve(self.data.class_name)
            self.data.config = implementation.config_type.model_validate(self.data.config)
        paths = [(path.parent if path.name == "SKILL.md" else path).expanduser().resolve() for path in self.skills]
        if len(paths) != len(set(paths)):
            raise ValueError("Static Skill paths must be unique")
        return self


class SkillEvolveConfig(EnvTestConfig):
    """Configure repeated collection, isolated evolution, and matched retesting."""

    cases: list[str] | None = None
    """Ordered task IDs; exclusive with cases_file."""

    cases_file: Path | None = None
    """JSON array of task IDs or objects containing a name field."""

    exclude_cases: list[str] = Field(default_factory=list)
    """Task IDs omitted from both comparison arms."""

    case_skills: dict[str, list[Path]] = Field(default_factory=dict)
    """Complete initial Skill directories keyed by task ID."""

    repeats: int = Field(default=3, ge=1)
    """Independent executions per task and comparison arm."""

    service_config: Path = ROOT / "config/mindmemos_skill/dev.yaml"
    """Existing service configuration supplying agents and evolution profiles."""

    pipeline: Literal["offline"] = "offline"
    """Evolution profile selected from the service configuration."""

    pipeline_overrides: dict[str, JsonValue] = Field(default_factory=dict)
    """Recursively merged profile overrides, excluding credentials."""

    skip_perfect: bool = True
    """Reuse perfect collection scores without evolution or retesting."""

    network_retries: int = Field(default=3, ge=0)
    """Additional attempts after network, API, or authentication errors."""

    evolution_retries: int = Field(default=1, ge=0)
    """Additional evolution attempts before reusing initial scores."""

    reference_file: Path | None = None
    """Optional per-task baseline and offline-evolution reference scores."""

    @model_validator(mode="after")
    def validate_evolution(self) -> Self:
        """Reject ambiguous task selection and unsupported source transformations."""
        if self.cases is not None and self.cases_file is not None:
            raise ValueError("Provide cases or cases_file, not both")
        if self.skill_body_only:
            raise ValueError("Evolution requires complete Skills, not skill_body_only")
        if self.cases is not None and len(set(self.cases)) != len(self.cases):
            raise ValueError("Case IDs must be unique")
        return self


class TreeSkillEvalConfig(BaseModel):
    """Evaluate TreeSkill through an isolated local Skill service."""

    model_config = ConfigDict(extra="forbid")
    """Reject unknown evaluation options."""

    environment: ImplementationConfig = Field(
        default_factory=lambda: ImplementationConfig(class_name="spreadsheetbench_text_action", config={})
    )
    """Registered environment and its local execution settings."""

    data: ImplementationConfig | None = None
    """Registered loader for other environments; exclusive with samples_file."""

    samples_file: Path | None = None
    """Native task/reward_data JSON array for other environments, instead of a loader."""

    service_config: Path
    """Local Skill service configuration containing the dedicated training profile."""

    data_root: Path | None = None
    """Verified 400 release directory containing dataset.json and spreadsheet files."""

    split_file: Path
    """Ordered train/val/test task IDs from the chosen benchmark split."""

    initial_skill: Path
    """Initial static Skill used to start fresh training."""

    profile: str = "treeskill_mindmemos_eval"
    """Configured local training profile; its trainer must be treeskill."""

    concurrency: int = Field(default=32, ge=1)
    """Explicit worker limit for rollout, analysis, localization and held-out evaluation."""

    poll_interval: float = Field(default=5, ge=1, le=60)
    """Seconds between local task progress reads."""

    output_dir: Path | None = None
    """Empty local directory for the evaluation and its service state."""
