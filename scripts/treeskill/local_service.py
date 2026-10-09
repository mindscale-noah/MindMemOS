"""Run a local service, report sampled trajectories, train TreeSkill, and compare tests."""

import argparse
import asyncio
from pathlib import Path

from filelock import FileLock
from mindmemos_skill.api import create_service
from mindmemos_skill.api.typing import (
    FileSkillInfo,
    RegisterFromLocalRequest,
    ReportTrajectoryRequest,
    TrainingRequest,
    TrainingSplit,
)
from mindmemos_skill.common.checkpoint import fingerprint
from mindmemos_skill.common.input_files import model_file_fingerprints
from mindmemos_skill.common.json import append_jsonl, read_json, write_json
from mindmemos_skill.common.skill_runtime import load_runtime_skill
from mindmemos_skill.config.loader import read_skill_config
from mindmemos_skill.envs import Environment
from mindmemos_skill.envs.config import EnvironmentConfig
from mindmemos_skill.envs.data import Data
from mindmemos_skill.skill import Skill, SkillSet

from scripts.treeskill.report import build_comparison
from scripts.treeskill.run import _evaluate


class TrajectoryReporter:
    """Report the trainer's existing rollouts before its analysis stage."""

    def __init__(self, service, root: Path, project: str, version: str):
        """Bind the service and exact initial version for this experiment."""
        self.service, self.root, self.project, self.version = service, root, project, version

    async def report(self, *, stage, message, completed=None, total=None):
        """Save progress and upload sampled messages once per completed slot."""
        append_jsonl(
            self.root / "service_progress.jsonl", dict(stage=stage, message=message, completed=completed, total=total)
        )
        print(f"{stage}: {message}", flush=True)
        if stage != "collection":
            return
        receipts = self.root / "trajectory_receipts"
        for path in sorted((self.root / "training/train/rollouts/train").glob("*/completed.json")):
            receipt = receipts / f"{path.parent.name}.json"
            if receipt.exists():
                continue
            saved = read_json(path)
            # Checkpoints wrap the completed rollout in a payload.
            payload = saved.get("payload", saved)
            if "failure" in payload:
                continue
            result = payload["result"]
            record = await self.service.report_trajectory(
                ReportTrajectoryRequest(
                    project_id=self.project,
                    skill_version_id=self.version,
                    session_id=f"{self.root.name}:{path.parent.name}",
                    messages=result["trajectory"],
                    usage=result.get("usage"),
                    started_at=result.get("started_at"),
                    finished_at=result.get("finished_at"),
                )
            )
            write_json(receipt, record.model_dump(mode="json"))


async def run(args):
    """Execute an isolated direct-service experiment without a Kafka consumer."""
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    config = read_skill_config(args.config).service
    config.task_manager = None
    config.evolution_pipelines = {}
    # This trusted local host authorizes only its explicit experiment project.
    from mindmemos_skill.infra.database import Scope

    def authorize(principal, project):
        if principal != args.project or project != args.project:
            raise PermissionError("Unexpected local experiment project")
        return Scope(fields={"project_id": project})

    pipeline = config.training_pipelines["treeskill"].config
    algorithm = pipeline.trainer.config
    if algorithm.samples_per_task != 1:
        raise ValueError("Expected one trajectory per training task")
    env_spec = EnvironmentConfig(
        class_name="spreadsheetbench_text_action",
        config={
            "agent": algorithm.optimizer.model_dump(mode="json"),
            "max_turns": 30,
            "transactional_recalculation": False,
        },
    )
    data = Data.build("spreadsheetbench", {"data_path": args.data_root / "dataset.json", "data_root": args.data_root})
    split = data.split(data.load(), split_file=args.split.resolve())
    train, test = split["train"][:30], split["test"][:200]
    if len(train) != 30 or len(test) != 200 or split["val"]:
        raise ValueError("Expected 30 training tasks, 200 test tasks, and no validation tasks")
    skill = Skill.from_path(args.initial_skill.resolve())
    identity = fingerprint(
        dict(
            config=config.model_dump(mode="json"),
            env=env_spec.model_dump(mode="json"),
            skill=skill.snapshot().model_dump(mode="json"),
            samples=[s.model_dump(mode="json") for s in [*train, *test]],
            input_files={s.task.id: model_file_fingerprints(s.task, s.reward_data) for s in [*train, *test]},
        )
    )
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        if read_json(manifest_path)["identity"] != identity:
            raise ValueError("Inputs changed; use a new output directory")
    elif any(p.name != ".run.lock" for p in root.iterdir()):
        raise ValueError("Output directory is not empty")
    manifest = dict(
        identity=identity,
        status="running",
        protocol={
            "train_count": 30,
            "test_count": 200,
            "samples_per_task": 1,
            "transactional_recalculation": False,
            "test_concurrency": 32,
            "rollout_concurrency": algorithm.rollout_concurrency,
            "analysis_concurrency": algorithm.analysis_concurrency,
            "localization_concurrency": algorithm.localization_concurrency,
        },
    )
    write_json(manifest_path, manifest)
    write_json(
        root / "split.json", {k: [s.task.id for s in v] for k, v in dict(train=train, val=[], test=test).items()}
    )
    # Validate the exact JSON round trip before any expensive model calls.
    request_options = dict(
        project_id=args.project,
        algo="treeskill",
        env=env_spec,
        init_skill_version_id="preflight",
        work_dir=str(root / "training"),
        tasks=train,
        split=TrainingSplit(train=[s.task.id for s in train]),
    )
    request = TrainingRequest(**request_options)
    TrainingRequest.model_validate_json(request.model_dump_json())
    if args.preflight_only:
        print("Input and training request round-trip validation passed")
        return
    service = await create_service(config, project_authorizer=authorize)
    try:
        with service.request_principal(args.project):
            registration = root / "registration.json"
            if not registration.exists():
                registered = await service.register_from_local(
                    RegisterFromLocalRequest(
                        project_id=args.project,
                        alias=f"local-tree-{identity[:16]}",
                        skill=FileSkillInfo(path=str(args.initial_skill.resolve())),
                    )
                )
                write_json(registration, registered.model_dump(mode="json"))
            version = read_json(registration)["skill_version_id"]
            env = Environment.build(env_spec.class_name, env_spec.config)
            baseline = await _evaluate(
                env=env,
                skills=SkillSet({"0": skill}, max_concurrency=1),
                samples=test,
                root=root,
                phase="baseline",
                concurrency=32,
            )
            if not baseline["complete"]:
                raise RuntimeError("Baseline has execution failures; inspect logs before starting training")
            request = request.model_copy(update={"init_skill_version_id": version})
            result = await service.run_training(
                request, reporter=TrajectoryReporter(service, root, args.project, version)
            )
            write_json(root / "training_result.json", result.model_dump(mode="json"))
            exact = await service.get_skill_version(result.version_id, project_id=args.project)
            from mindmemos_skill.common.skill_versions import snapshot_from_version

            evolved = load_runtime_skill(snapshot_from_version(exact))
            evolved.export(root / "evolved_skill", overwrite=True)
            evaluation = await _evaluate(
                env=env,
                skills=SkillSet({"0": evolved}, max_concurrency=1),
                samples=test,
                root=root,
                phase="evolved",
                concurrency=32,
            )
            if not evaluation["complete"]:
                raise RuntimeError("Evolved test has execution failures; comparison is incomplete")
            manifest["status"] = "completed"
    except Exception as exc:
        manifest.update(status="failed", error_type=type(exc).__name__)
        raise
    finally:
        write_json(manifest_path, manifest)
        build_comparison(root)
        await service.close()


def main():
    """Parse local experiment paths and own one output-directory lock."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/mindmemos_skill/dev.yaml"))
    parser.add_argument("--project", default="proj-dev-memory-skill")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--split",
        type=Path,
        default=Path("resources/mindmemos_skill_eval/treeskill-spreadsheetbench/data_splits/splits.json"),
    )
    parser.add_argument(
        "--initial-skill",
        type=Path,
        default=Path("resources/mindmemos_skill_eval/treeskill-spreadsheetbench/initial_skill"),
    )
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(args.output / ".run.lock"), timeout=0):
        asyncio.run(run(args))


if __name__ == "__main__":
    main()
