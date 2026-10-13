"""Configured Skill training and held-out evaluation."""

from pathlib import Path

from filelock import AsyncFileLock
from pydantic import BaseModel, JsonValue

from mindmemos_skill.common.ids import generate_stable_id

from ....algos import Algorithm
from ....algos.train.base import SkillTrainer
from ....algos.train.skill_forest.trainer import SkillForestTrainer
from ....api.typing import AlgoSkillEvolveResult, TrainingRequest
from ....common.checkpoint import fingerprint
from ....common.json import read_json, write_json
from ....common.secrets import redact_credentials
from ....common.skill_runtime import load_runtime_skill
from ....common.skill_versions import snapshot_from_version
from ....envs import Environment
from ....envs.data.typing import DataSample
from ....errors import SkillConflictError
from ....logging import get_logger, stage
from ....persistence.db_services.skills.typing import SkillVersionRecord
from ....persistence.store import SkillPersistence
from ....skill import Skill
from ...base import Pipeline
from ...config import PipelineComponentConfig
from ...train import TrainingPipeline
from .config import TrainingPipelineConfig
from .evaluation import evaluate_skill
from .typing import HeldOutEvaluation


def training_run_id(project_id: str, work_dir: str | Path) -> str:
    """Derive the stable publication and observability identity for a run.

    Args:
        project_id: Local repository scope.
        work_dir: Caller-provided isolated output directory.

    Returns:
        Run identity stable across process restarts.
    """
    root = Path(work_dir).expanduser().resolve()
    return generate_stable_id(f"{project_id}:{root}", "train_")


@Pipeline.register("training")
class ConfiguredTrainingPipeline(TrainingPipeline[TrainingPipelineConfig]):
    """Execute any configured SkillTrainer and publish its frozen result."""

    config_type = TrainingPipelineConfig
    """Trainer configuration validated by the factory."""

    def __init__(self, config: TrainingPipelineConfig, *, persistence: SkillPersistence):
        """Build the trainer while borrowing repository storage.

        Args:
            config: Trainer and evaluation settings.
            persistence: Shared storage owned and closed by the service.
        """
        if not isinstance(persistence, SkillPersistence):
            raise TypeError("Training pipeline requires SkillPersistence")
        super().__init__(config, persistence=persistence)
        self.trainer = Algorithm.build(config.trainer.class_name, config.trainer.config)

    async def train(
        self,
        request: TrainingRequest,
        *,
        env: Environment,
        adapter: PipelineComponentConfig | None = None,
    ) -> AlgoSkillEvolveResult:
        """Load a version, train in the supplied directory, evaluate and publish.

        Args:
            request: Explicit repository scope, base version, tasks, splits and directory.
            env: Environment selected for this run.
            adapter: Optional adaptation bound to this environment and trainer.

        Returns:
            An immutable published version and stable run identity.

        Raises:
            SkillConflictError: The current selection or publication inputs changed.
            ValueError: Scope, data, directory or trainer inputs are invalid.
        """
        request = TrainingRequest.model_validate(request)
        if not isinstance(env, Environment):
            raise TypeError("Training requires an Environment")
        trainer = self._select_trainer(adapter)

        root = Path(request.work_dir).expanduser().resolve()
        root.parent.mkdir(parents=True, exist_ok=True)
        run_id = training_run_id(request.project_id, root)
        config = self._run_config(request, trainer, env)
        config_hash = fingerprint(config)
        async with AsyncFileLock(str(root.parent / f".{root.name}.pipeline.lock"), timeout=0, run_in_executor=False):
            with stage("training.prepare"):
                self._prepare_directory(root, run_id, config_hash, config)
                store = self.persistence.skills
                published = await store.algorithm_publication(run_id, config_hash)
                if published is not None:
                    return self._result(request, published, run_id)
                owner, base = await store.get_with_owner(request.init_skill_version_id)
                samples = {sample.task.id: sample for sample in request.tasks}
                initial_skill = load_runtime_skill(snapshot_from_version(base))

            with stage("training.train"):
                selected = await trainer.train(
                    skill=initial_skill,
                    env=env,
                    train_samples=[samples[key] for key in request.split.train],
                    validation_samples=[samples[key] for key in request.split.val],
                    work_dir=root / "train",
                )
            report = trainer.build_report(root / "train")
            write_json(root / "selected_skill.json", selected.snapshot())
            report = await self._evaluate_splits(request, trainer, selected, env, samples, root, report)

            with stage("training.publish"):
                version = await store.commit_algorithm(
                    owner,
                    selected.snapshot(),
                    run_id=run_id,
                    input_hash=config_hash,
                    algorithm=self.config.trainer.class_name.replace("_", "-"),
                    base_version_id=base.version_id,
                    active=request.auto_accept,
                    report=report,
                )
            result = self._result(request, version, run_id)
            write_json(root / "publication.json", result)
            get_logger(__name__).info(
                "algorithm.progress",
                extra={"event_body": {"stage": "published", "message": "Selected Skill version published"}},
            )
            return result

    def _select_trainer(self, adapter: PipelineComponentConfig | None) -> SkillTrainer:
        """Apply a per-run adapter override without mutating the configured trainer."""
        trainer = self.trainer
        if adapter is not None:
            if self.config.trainer.class_name in {"skill_forest", "treeskill"}:
                raise ValueError(
                    f"{self.config.trainer.class_name} selects its adapter automatically; remove the adapter override"
                )
            options = self.config.trainer.config
            trainer_options = options.model_dump() if isinstance(options, BaseModel) else options
            trainer_options = {**trainer_options, "env_adapter": adapter.model_dump()}
            trainer = Algorithm.build(self.config.trainer.class_name, trainer_options)
        return trainer

    def _run_config(self, request: TrainingRequest, trainer: SkillTrainer, env: Environment) -> JsonValue:
        """Capture sanitized execution inputs for restart consistency checks."""
        return redact_credentials(
            {
                "request": request.model_dump(mode="json", exclude={"work_dir", "env"}),
                "trainer": {
                    "class_name": self.config.trainer.class_name,
                    "config": trainer.config.model_dump(mode="json"),
                },
                "environment": {
                    "class": f"{type(env).__module__}.{type(env).__qualname__}",
                    "config": env.config.model_dump(mode="json"),
                },
                "evaluation_concurrency": self.config.evaluation_concurrency,
            },
            preserve_execution_settings=True,
        )

    @staticmethod
    def _prepare_directory(root: Path, run_id: str, config_hash: str, config: JsonValue) -> None:
        """Accept an existing matching run or initialize an empty directory under the lock."""
        config_path = root / "config.json"
        if config_path.exists():
            if read_json(config_path)["config_hash"] != config_hash:
                raise SkillConflictError("Training run was configured with different inputs")
        else:
            if root.exists() and any(root.iterdir()):
                raise ValueError("Training directory has no config.json; use a new work_dir")
            write_json(config_path, {"run_id": run_id, "config_hash": config_hash, "config": config})

    async def _evaluate_splits(
        self,
        request: TrainingRequest,
        trainer: SkillTrainer,
        selected: Skill,
        env: Environment,
        samples: dict[str, DataSample],
        root: Path,
        report: dict[str, JsonValue] | None,
    ) -> dict[str, JsonValue] | None:
        """Evaluate the frozen selection and merge split reports into publication metadata."""
        for name, ids in (("val", request.split.val), ("test", request.split.test)):
            if not ids:
                continue
            _log_evaluation_progress(name, len(ids), completed=False)
            with stage("evaluate"):
                evaluation = await evaluate_skill(
                    skill=selected,
                    env=env,
                    samples=[samples[key] for key in ids],
                    directory=root / name,
                    concurrency=self.config.evaluation_concurrency,
                    stage_name=f"skill_forest.{name}" if isinstance(trainer, SkillForestTrainer) else None,
                )
            report = report or {"algorithm": self.config.trainer.class_name}
            report.setdefault("evaluations", {})[name] = evaluation.report.model_dump(mode="json")
            if isinstance(trainer, SkillForestTrainer):
                _record_forest_evaluation(root / "train", name, evaluation)
            _log_evaluation_progress(name, len(ids), completed=True)
        return report

    @staticmethod
    def _result(request: TrainingRequest, version: SkillVersionRecord, run_id: str) -> AlgoSkillEvolveResult:
        """Return the durable publication identity without changing current selection."""
        return AlgoSkillEvolveResult(
            skill_id=version.skill_id,
            base_version_id=request.init_skill_version_id,
            version_id=version.version_id,
            run_id=run_id,
            content=version.content,
            resources=version.resources,
            content_hash=version.content_hash,
            skill_hash=version.skill_hash,
        )


def _log_evaluation_progress(name: str, total: int, *, completed: bool) -> None:
    """Emit the shared progress event for an evaluation split."""
    get_logger(__name__).info(
        "algorithm.progress",
        extra={
            "event_body": {
                "stage": name,
                "message": f"{'Evaluated' if completed else 'Evaluating'} {name} samples",
                "completed": total if completed else 0,
                "total": total,
            }
        },
    )


def _record_forest_evaluation(directory: Path, name: str, evaluation: HeldOutEvaluation) -> None:
    """Keep the forest result artifact synchronized with final split evaluations."""
    result_path = directory / "result.json"
    result = read_json(result_path)
    result["validation" if name == "val" else "test"] = {
        **evaluation.summary.model_dump(mode="json"),
        "tasks": [task.model_dump(mode="json") for task in evaluation.tasks],
    }
    write_json(result_path, result)
