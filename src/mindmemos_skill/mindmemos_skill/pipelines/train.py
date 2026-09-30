"""Business contract for Skill training with environment feedback."""

from abc import ABC, abstractmethod
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from filelock import AsyncFileLock

from ..algos import Algorithm
from ..common.asyncio import run_sync
from ..common.checkpoint import boundary, fingerprint, redact
from ..common.json import write_json
from ..envs import Environment
from ..errors import SkillConflictError
from ..persistence.store import SkillPersistence
from ..registry.base import ConfigT
from ..skill import Skill
from .base import Pipeline
from .config import TrainingPipelineConfig
from .content import snapshot_from_version
from .evaluation import evaluate_skill
from .typing import SkillPipelineResult, TrainingRequest


def training_run_id(project_id: str, work_dir: str | Path) -> str:
    """Derive the stable publication and observability identity for a run.

    Args:
        project_id: Local repository scope.
        work_dir: Caller-provided isolated output directory.

    Returns:
        Run identity stable across process restarts.
    """
    root = Path(work_dir).expanduser().resolve()
    return "train_" + uuid5(NAMESPACE_URL, f"{project_id}:{root}").hex


class TrainingPipeline(Pipeline[ConfigT], ABC):
    """Prepare training inputs, invoke a trainer, and persist its selection.

    Concrete implementations load the requested Skill version and partition
    supplied tasks according to the request's train/val/test membership. They
    supply the environment, training/validation samples, and an isolated run
    directory to the trainer. Missing validation means an empty validation set;
    an algorithm requiring validation must reject that input before execution.
    Test samples stay outside the trainer and are evaluated only after final
    candidate selection, without tuning or reselection based on test feedback.
    Missing test samples skip this final evaluation.
    Publishing pipelines pass their fixed algorithm identifier through
    VersionNaming.algorithm when committing; labels are allocated by persistence.
    The trainer controls candidate-dependent rollouts without direct or indirect database
    access. Pipelines manage resources they create and borrow injected backends.
    Private scoring inputs must remain separate from public task material.
    """

    @abstractmethod
    async def train(self, request: TrainingRequest) -> SkillPipelineResult:
        """Train one Skill and commit a child of its captured current version.

        Implementations must preserve project scope and source data, avoid
        holding database transactions during training, retain the candidate on
        persistence failure, and reject concurrent version changes. They must
        not mutate the request or return success before persistence.

        Args:
            request: Complete tasks, explicit train/val/test membership, project
                and Skill selection, and the required input version.

        Returns:
            The final selected and persisted Skill version, its parent version,
            and the execution identity, after any requested held-out evaluation.

        Raises:
            ValueError: Inputs or required training/validation data are invalid.
            SkillConflictError: The expected current version changed.
        """


@Pipeline.register("training")
class ConfiguredTrainingPipeline(TrainingPipeline[TrainingPipelineConfig]):
    """Execute any configured SkillTrainer and publish its frozen result."""

    config_type = TrainingPipelineConfig
    """Nested trainer and environment definitions validated by the factory."""

    def __init__(self, config: TrainingPipelineConfig, *, persistence: SkillPersistence):
        """Build stateless execution components and borrow repository storage.

        Args:
            config: Complete trainer, environment, and evaluation settings.
            persistence: Shared storage owned and closed by the service.
        """
        if not isinstance(persistence, SkillPersistence):
            raise TypeError("Training pipeline requires SkillPersistence")
        super().__init__(config, persistence=persistence)
        self.trainer = Algorithm.build(config.trainer.class_name, config.trainer.config)
        self.environment = Environment.build(config.environment.class_name, config.environment.config)

    async def train(self, request: TrainingRequest) -> SkillPipelineResult:
        """Load a version, train in the supplied directory, evaluate and publish.

        Args:
            request: Explicit repository scope, base version, tasks, splits and directory.

        Returns:
            An immutable published version and stable run identity.

        Raises:
            SkillConflictError: The captured parent or publication inputs changed.
            ValueError: Scope, data, directory or trainer inputs are invalid.
        """
        request = TrainingRequest.model_validate(request)
        if request.project_id is None:
            raise ValueError("project_id is required")
        root = Path(request.work_dir).expanduser().resolve()
        root.parent.mkdir(parents=True, exist_ok=True)
        run_id = training_run_id(request.project_id, root)
        identity = {
            "request": request.model_dump(mode="json", exclude={"pipeline", "work_dir"}),
            "directory": str(root),
            "config": redact(self.config.model_dump(mode="json")),
        }
        input_hash = fingerprint(identity)
        async with AsyncFileLock(str(root.parent / f".{root.name}.pipeline.lock"), timeout=0):
            store = self.persistence.skills
            published = await run_sync(store.training_publication, request.project_id, run_id, input_hash)
            if published is not None:
                return self._result(request, published, run_id)
            base = await run_sync(store.version_by_id, request.project_id, request.base_version_id)
            owner = await run_sync(store.resolve, request.project_id, base.skill_id)
            if owner.current_version_id != base.version_id:
                raise SkillConflictError("base_version_id is not the current version")
            if root.exists() and any(root.iterdir()) and not (root / "request.json").exists():
                raise ValueError("Training directory has no business input record; use a new work_dir")
            # This is the business input record, never an algorithm checkpoint.
            boundary(root / "request.json", identity, payload={"run_id": run_id})
            samples = {sample.task.id: sample for sample in request.tasks}
            selected = await self.trainer.train(
                skill=Skill.from_snapshot(snapshot_from_version(base)),
                env=self.environment,
                train_samples=[samples[key] for key in request.split.train],
                validation_samples=[samples[key] for key in request.split.val],
                work_dir=root / "train",
            )
            report = self.trainer.build_report(root / "train")
            write_json(root / "selected_skill.json", selected.snapshot())
            for name, ids in (("val", request.split.val), ("test", request.split.test)):
                if ids:
                    await run_sync(
                        evaluate_skill,
                        skill=selected,
                        env=self.environment,
                        samples=[samples[key] for key in ids],
                        directory=root / name,
                        concurrency=self.config.evaluation_concurrency,
                    )
            version = await run_sync(
                store.commit_training,
                owner,
                selected.snapshot(),
                run_id=run_id,
                input_hash=input_hash,
                algorithm=self.config.trainer.class_name.replace("_", "-"),
                report=report,
            )
            result = self._result(request, version, run_id)
            write_json(root / "publication.json", result)
            return result

    @staticmethod
    def _result(request, version, run_id):
        """Return the durable publication identity without changing current selection."""
        return SkillPipelineResult(
            project_id=request.project_id,
            skill_id=version.skill_id,
            base_version_id=request.base_version_id,
            version_id=version.version_id,
            run_id=run_id,
        )
