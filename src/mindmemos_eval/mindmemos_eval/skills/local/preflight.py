"""Validate one local Skill benchmark without constructing a service."""

import json
from hashlib import sha256
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml
from mindmemos_skill.algos.train.skill_forest.env_specific.selection import select_env_adapter
from mindmemos_skill.algos.train.treeskill.env_specific.selection import select_env_adapter as select_tree_adapter
from mindmemos_skill.api.typing import TrainingSplit
from mindmemos_skill.config.loader import read_skill_config
from mindmemos_skill.envs import Environment
from mindmemos_skill.envs.builtin_envs.spreadsheetbench.recalculation import preflight_recalculation_runtime
from mindmemos_skill.envs.data import Data
from mindmemos_skill.pipelines.builtin_pipelines.training.config import TrainingPipelineConfig

from .config import LocalSkillEvalConfig
from .typing import PreparedInputs

_DATA_PATH_FIELDS = ("data_path", "data_root", "docs_root", "game_root")
"""Known built-in dataset source paths resolved against the experiment YAML."""

_DATASET_BY_ENVIRONMENT = {
    "spreadsheetbench_text_action": "spreadsheetbench",
    "spreadsheetbench_tool_calling": "spreadsheetbench",
}
"""Environment implementations whose dataset loader has a different name."""


class LocalSkillEvalPreflight:
    """Load all local inputs and reject incompatible experiment definitions."""

    def prepare(self, config_path: str | Path) -> PreparedInputs:
        """Validate config, training profile, split, and local files without model calls.

        Args:
            config_path: Experiment YAML path, absolute or relative to the caller.

        Returns:
            Prepared and fingerprinted inputs for one run.

        Raises:
            ValueError: Configuration, dataset, split, or environment is incompatible.
            OSError: A required local file cannot be read.
        """
        source = Path(config_path).expanduser().resolve()
        config = LocalSkillEvalConfig.model_validate(yaml.safe_load(source.read_text(encoding="utf-8")))
        root = source.parent
        config.service_config = self._resolve(root, config.service_config)
        config.split_file = self._resolve(root, config.split_file)
        config.initial_skill.path = self._resolve(root, config.initial_skill.path)
        for field in _DATA_PATH_FIELDS:
            value = config.data.config.get(field)
            if value is not None:
                config.data.config[field] = str(self._resolve(root, Path(value)))

        deployment = read_skill_config(config.service_config)
        service = deployment.service
        if deployment.fastapi.auth.mode != "none":
            raise ValueError("Local Skill evaluation requires none authentication mode")
        if service.persistence is None or service.persistence.database.class_name != "sqlite":
            raise ValueError("Local Skill evaluation requires SQLite persistence")
        if service.repository_pipeline is None:
            raise ValueError("Local Skill evaluation requires a repository pipeline")
        pipeline = service.training_pipelines.get(config.algorithm)
        if (
            pipeline is None
            or TrainingPipelineConfig.model_validate(pipeline.config).trainer.class_name != config.algorithm
        ):
            raise ValueError(f"Training pipeline {config.algorithm!r} is missing or uses another trainer")
        spec = config.environment
        expected_data = _DATASET_BY_ENVIRONMENT.get(spec.class_name, spec.class_name)
        if config.data.class_name != expected_data:
            raise ValueError(f"Environment {spec.class_name!r} requires data loader {expected_data!r}")
        if config.data.class_name == "alfworld_bounded_history":
            if not spec.config.simulator:
                raise ValueError("ALFWorld requires the parsed simulator configuration")
            if find_spec("alfworld") is None:
                raise ValueError("ALFWorld runtime is unavailable; install mindmemos-skill[alfworld]")
        if config.data.class_name == "spreadsheetbench":
            if find_spec("openpyxl") is None or find_spec("pandas") is None:
                raise ValueError("SpreadsheetBench requires mindmemos-skill[spreadsheetbench]")
            expected_environment = (
                "spreadsheetbench_text_action" if config.algorithm == "treeskill" else "spreadsheetbench_tool_calling"
            )
            if spec.class_name != expected_environment:
                raise ValueError(f"SpreadsheetBench requires environment {expected_environment!r}")
            if config.algorithm == "treeskill" and spec.config.transactional_recalculation:
                with TemporaryDirectory(prefix="skill-eval-recalculation-") as temporary:
                    preflight_recalculation_runtime(Path(temporary))

        environment = Environment.build(spec.class_name, spec.config)
        if config.algorithm == "skill_forest":
            adapter = select_env_adapter(environment)
        else:
            adapter = select_tree_adapter(environment)
        adapter.validate_environment(environment)

        skill_path = config.initial_skill.path
        if skill_path.is_file() and skill_path.name != "SKILL.md":
            raise ValueError("Initial Skill file must be named SKILL.md")
        if not (skill_path / "SKILL.md" if skill_path.is_dir() else skill_path).is_file():
            raise ValueError("Initial Skill source must contain SKILL.md")
        data = Data.build(config.data.class_name, config.data.config)
        all_samples = data.load()
        selected = data.split(all_samples, split_file=config.split_file)
        if set(selected) != {"train", "val", "test"}:
            raise ValueError("Split file must contain exactly train, val, and test")
        limits = config.limits
        train = selected["train"][: limits.train]
        val = selected["val"][: limits.val]
        test = selected["test"][: limits.test]
        if not train or not test:
            raise ValueError("Both train and test splits must be nonempty")
        if config.algorithm == "treeskill" and val:
            raise ValueError("TreeSkill requires an empty external val split")
        split = TrainingSplit(
            train=[sample.task.id for sample in train],
            val=[sample.task.id for sample in val],
            test=[sample.task.id for sample in test],
        )
        samples = train + val + test
        digest = sha256()
        for content in (
            config.model_dump_json(),
            service.model_dump_json(),
            split.model_dump_json(),
            *(sample.model_dump_json() for sample in samples),
        ):
            digest.update(content.encode("utf-8"))
        for sample in samples:
            digest.update(
                json.dumps(environment.input_identity(sample.task, sample.reward_data), sort_keys=True).encode("utf-8")
            )
        skill_root = skill_path if skill_path.is_dir() else skill_path.parent
        for path in sorted(skill_root.rglob("*")):
            if path.is_file():
                digest.update(str(path.relative_to(skill_root)).encode("utf-8"))
                digest.update(path.read_bytes())
        return PreparedInputs(
            config=config,
            service=deployment,
            samples=samples,
            split=split,
            config_path=source,
            input_digest=digest.hexdigest(),
        )

    @staticmethod
    def _resolve(root: Path, value: Path) -> Path:
        """Resolve an input path relative to the experiment definition."""
        return (root / value).expanduser().resolve()
