"""Configured Skill training and held-out evaluation."""

from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from filelock import AsyncFileLock
from pydantic import BaseModel

from ....algos import Algorithm
from ....algos.train.skill_forest.trainer import SkillForestTrainer
from ....api.typing import AlgoSkillEvolveResult, TrainingRequest
from ....common.asyncio import run_sync
from ....common.checkpoint import boundary, fingerprint
from ....common.json import read_json, write_json
from ....common.skill_runtime import load_runtime_skill
from ....common.skill_versions import snapshot_from_version
from ....envs import Environment
from ....infra.task_manager.progress import NullProgressReporter, ProgressReporter
from ....persistence.store import SkillPersistence
from ...base import Pipeline
from ...config import PipelineComponentConfig
from ...train import TrainingPipeline
from .config import TrainingPipelineConfig
from .evaluation import evaluate_skill


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
        reporter: ProgressReporter | None = None,
    ) -> AlgoSkillEvolveResult:
        """Load a version, train in the supplied directory, evaluate and publish.

        Args:
            request: Explicit repository scope, base version, tasks, splits and directory.
            env: Environment selected for this run.
            adapter: Optional adaptation bound to this environment and trainer.
            reporter: Optional progress sink shared with the trainer.

        Returns:
            An immutable published version and stable run identity.

        Raises:
            SkillConflictError: The current selection or publication inputs changed.
            ValueError: Scope, data, directory or trainer inputs are invalid.
        """
        request = TrainingRequest.model_validate(request)
        if not isinstance(env, Environment):
            raise TypeError("Training requires an Environment")
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
        active_reporter = reporter or NullProgressReporter()
        if request.project_id is None:
            raise ValueError("project_id is required")
        root = Path(request.work_dir).expanduser().resolve()
        root.parent.mkdir(parents=True, exist_ok=True)
        run_id = training_run_id(request.project_id, root)
        request_identity = request.model_dump(mode="json", exclude={"work_dir"})
        identity = {
            "request": request_identity,
            "directory": str(root),
            "config": self.config.model_dump(mode="json"),
        }
        identity["selected_environment"] = {
            "class": f"{type(env).__module__}.{type(env).__qualname__}",
            "config": env.config.model_dump(mode="json"),
            "trainer_config": trainer.config.model_dump(mode="json"),
        }
        input_hash = fingerprint(identity)
        async with AsyncFileLock(str(root.parent / f".{root.name}.pipeline.lock"), timeout=0):
            store = self.persistence.skills
            published = await run_sync(store.algorithm_publication, request.project_id, run_id, input_hash)
            if published is not None:
                return self._result(request, published, run_id)
            base = await run_sync(store.version_by_id, request.project_id, request.init_skill_version_id)
            owner = await run_sync(store.resolve_family, request.project_id, base.skill_id)
            if root.exists() and any(root.iterdir()) and not (root / "request.json").exists():
                raise ValueError("Training directory has no business input record; use a new work_dir")
            # This is the business input record, never an algorithm checkpoint.
            boundary(root / "request.json", identity, payload={"run_id": run_id})
            samples = {sample.task.id: sample for sample in request.tasks}
            trainer_input = {
                "skill": load_runtime_skill(snapshot_from_version(base)),
                "env": env,
                "train_samples": [samples[key] for key in request.split.train],
                "validation_samples": [samples[key] for key in request.split.val],
                "work_dir": root / "train",
            }
            if reporter is not None:
                trainer_input["progress"] = reporter
            selected = await trainer.train(**trainer_input)
            report = trainer.build_report(root / "train")
            write_json(root / "selected_skill.json", selected.snapshot())
            for name, ids in (("val", request.split.val), ("test", request.split.test)):
                if ids:
                    await active_reporter.report(
                        stage=name, message=f"Evaluating {name} samples", completed=0, total=len(ids)
                    )
                    evaluation = await evaluate_skill(
                        skill=selected,
                        env=env,
                        samples=[samples[key] for key in ids],
                        directory=root / name,
                        concurrency=self.config.evaluation_concurrency,
                    )
                    if isinstance(trainer, SkillForestTrainer):
                        result_path = root / "train" / "result.json"
                        result_data = read_json(result_path)
                        result_data["validation" if name == "val" else "test"] = {
                            **evaluation,
                            "tasks": read_json(root / name / "results.json")["tasks"],
                        }
                        write_json(result_path, result_data)
                    await active_reporter.report(
                        stage=name, message=f"Evaluated {name} samples", completed=len(ids), total=len(ids)
                    )
            version = await run_sync(
                store.commit_algorithm,
                owner,
                selected.snapshot(),
                run_id=run_id,
                input_hash=input_hash,
                algorithm=self.config.trainer.class_name.replace("_", "-"),
                base_version_id=base.version_id,
                active=request.auto_accept,
                report=report,
            )
            result = self._result(request, version, run_id)
            write_json(root / "publication.json", result)
            await active_reporter.report(stage="published", message="Selected Skill version published")
            return result

    @staticmethod
    def _result(request, version, run_id):
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
