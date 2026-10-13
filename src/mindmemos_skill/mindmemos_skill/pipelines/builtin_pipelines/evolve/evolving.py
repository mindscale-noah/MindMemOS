"""Stored-evidence Skill evolution pipeline."""

import random
from pathlib import Path

from filelock import AsyncFileLock

from mindmemos_skill.common.ids import generate_id

from ....algos import Algorithm
from ....algos.evolve.typing import SkillTrajectory
from ....api.typing import AlgoSkillEvolveResult, SkillPipelineRequest
from ....common.checkpoint import fingerprint
from ....common.json import read_json, write_json
from ....common.secrets import redact_credentials
from ....common.skill_runtime import load_runtime_skill
from ....common.skill_versions import snapshot_from_version
from ....common.trajectories import collect_skill_trajectories
from ....envs.typing import Reward, Task
from ....errors import SkillConflictError
from ....logging import get_logger, stage
from ....persistence import SkillRecord, TrajectoryRecord
from ....persistence.store import SkillPersistence
from ....skill import Skill
from ....skill.typing import SkillSnapshot
from ...base import Pipeline
from ...evolve import EvolutionPipeline
from .config import SimpleEvolutionPipelineConfig


@Pipeline.register("simple_evolution")
class SimpleEvolutionPipeline(EvolutionPipeline[SimpleEvolutionPipelineConfig]):
    """Sample stored executions, evolve a Skill and publish a new version."""

    config_type = SimpleEvolutionPipelineConfig
    """Evidence bounds, working directory and registered evolver."""

    def __init__(self, config: SimpleEvolutionPipelineConfig, *, persistence: SkillPersistence):
        """Build the selected evolver and borrow the scoped repository.

        Args:
            config: Validated evidence and algorithm settings.
            persistence: Shared service-owned Skill store.
        """
        if not isinstance(persistence, SkillPersistence):
            raise TypeError("Evolution pipeline requires SkillPersistence")
        super().__init__(config, persistence=persistence)
        self.evolver = Algorithm.build(config.evolver.class_name, config.evolver.config)

    async def evolve(self, request: SkillPipelineRequest) -> AlgoSkillEvolveResult:
        """Collect, sample, merge batch patches and publish a changed child version.

        Args:
            request: Authorized project, Skill reference and optional expected parent.

        Returns:
            Identity of the published child, or the original version when unchanged.

        Raises:
            ValueError: Evidence is insufficient or the candidate patch is invalid.
            SkillConflictError: The current selection changed before publication.
        """
        request = SkillPipelineRequest.model_validate(request)
        run_id = request.run_id or generate_id("evolve_")
        root = Path(self.config.work_dir).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        directory = root / run_id
        config = redact_credentials(
            {
                "request": request.model_dump(mode="json", exclude={"run_id"}),
                "evolver": {
                    "class_name": self.config.evolver.class_name,
                    "config": self.evolver.config.model_dump(mode="json"),
                },
                "traj_collect": self.config.traj_collect.model_dump(mode="json"),
            },
            preserve_execution_settings=True,
        )
        config_hash = fingerprint(config)
        store = self.persistence.skills
        async with AsyncFileLock(str(root / f".{run_id}.lock"), run_in_executor=False):
            config_path = directory / "config.json"
            if config_path.exists():
                if read_json(config_path)["config_hash"] != config_hash:
                    raise SkillConflictError("Evolution run was configured with different inputs")
            else:
                if directory.exists() and any(directory.iterdir()):
                    raise ValueError("Evolution directory has no config.json; use a new run_id")
                write_json(config_path, {"run_id": run_id, "config_hash": config_hash, "config": config})
            published = await store.algorithm_publication(run_id, config_hash, operation="evolution")
            if published is not None:
                return self._result(published, run_id)
            request_path = directory / "request.json"
            if request_path.exists():
                saved = read_json(request_path)
                owner = SkillRecord.model_validate(saved["owner"])
                base = await store.get(saved["base_version_id"])
            else:
                owner = await store.resolve_family(request.skill_ref)
                base = await store.version(owner, request.base_version_id)
                saved = {"owner": owner.model_dump(mode="json"), "base_version_id": base.version_id}
                write_json(request_path, saved)
            candidate_path = directory / "candidate.json"
            if candidate_path.exists():
                snapshot = SkillSnapshot.model_validate(read_json(candidate_path))
            else:
                with stage("evolve.prepare"):
                    trajectories = await self._evidence(request, owner, base, directory)
                with stage("evolve.evolving"):
                    candidate = await self.evolver.evolve(
                        skill=load_runtime_skill(snapshot_from_version(base)), trajectories=trajectories
                    )
                snapshot = candidate.snapshot()
                write_json(candidate_path, snapshot)
                get_logger(__name__).info(
                    "algorithm.progress",
                    extra={"event_body": {"stage": "evolved", "message": "Skill candidate created"}},
                )
            evidence_path = directory / "evidence.json"
            if not evidence_path.exists():
                raise ValueError("Evolution candidate requires its sampled evidence checkpoint")
            evidence = read_json(evidence_path)
            await self.persistence.trajectories.mark_consumed(
                base.version_id, [item["record"]["trajectory_id"] for item in evidence["samples"]], run_id
            )
            if snapshot.skill_hash == base.skill_hash:
                get_logger(__name__).info(
                    "algorithm.progress",
                    extra={"event_body": {"stage": "unchanged", "message": "No supported Skill changes"}},
                )
                return AlgoSkillEvolveResult(
                    skill_id=base.skill_id,
                    base_version_id=base.version_id,
                    version_id=base.version_id,
                    run_id=run_id,
                    content=base.content,
                    resources=base.resources,
                    content_hash=base.content_hash,
                    skill_hash=base.skill_hash,
                )
            with stage("evolve.publish"):
                version = await store.commit_algorithm(
                    owner,
                    snapshot,
                    run_id=run_id,
                    input_hash=config_hash,
                    operation="evolution",
                    algorithm=self.config.evolver.class_name.replace("_", "-"),
                    commit_source=self.config.evolver.class_name,
                    base_version_id=base.version_id,
                    active=request.auto_accept,
                )
            get_logger(__name__).info(
                "algorithm.progress",
                extra={"event_body": {"stage": "published", "message": "Evolved Skill version published"}},
            )
            return self._result(version, run_id)

    @staticmethod
    def _result(version, run_id):
        """Return the immutable result without changing the current selection."""
        return AlgoSkillEvolveResult(
            skill_id=version.skill_id,
            base_version_id=version.parent_version_ids[0],
            version_id=version.version_id,
            run_id=run_id,
            content=version.content,
            resources=version.resources,
            content_hash=version.content_hash,
            skill_hash=version.skill_hash,
        )

    async def _evidence(self, request, owner, base, directory):
        """Reuse the exact sampled records after an interrupted model call."""
        evidence_path = directory / "evidence.json"
        if evidence_path.exists():
            saved = read_json(evidence_path)
            sample = [
                (TrajectoryRecord.model_validate(item["record"]), item["version_id"]) for item in saved["samples"]
            ]
        else:
            records = await collect_skill_trajectories(
                self.persistence,
                skill_version_id=base.version_id,
                limit=self.config.traj_collect.trajectory_limit,
                max_page_size=self.config.traj_collect.max_page_size,
            )
            eligible = [
                (record, base.version_id)
                for record in records
                if record.task_instruction.strip() and record.skill_version_id == base.version_id
            ]
            get_logger(__name__).info(
                "algorithm.progress",
                extra={
                    "event_body": {
                        "stage": "evidence",
                        "message": f"Collected {len(records)} trajectories; {len(eligible)} eligible",
                        "completed": len(eligible) if records else None,
                        "total": len(records) if records else None,
                    }
                },
            )
            if not eligible:
                raise ValueError("No eligible historical trajectories for this Skill version")
            if len(eligible) < self.config.traj_collect.sample_size:
                raise ValueError("Fewer eligible trajectories than sample_size")
            sample = (
                random.SystemRandom().sample(eligible, self.config.traj_collect.sample_size)
                if self.config.traj_collect.seed is None
                else random.Random(self.config.traj_collect.seed).sample(eligible, self.config.traj_collect.sample_size)
            )
            write_json(
                evidence_path,
                {
                    "samples": [
                        {"record": record.model_dump(mode="json"), "version_id": version_id}
                        for record, version_id in sample
                    ]
                },
            )
            write_json(
                directory / "input.json",
                {
                    "project_id": request.project_id,
                    "skill_id": owner.skill_id,
                    "base_version_id": base.version_id,
                    "auto_accept": request.auto_accept,
                    "collected_trajectory_count": len(records),
                    "eligible_trajectory_count": len(eligible),
                    "sampled_trajectory_count": len(sample),
                    "sampled_trajectory_ids": [record.trajectory_id for record, _ in sample],
                },
            )
        get_logger(__name__).info(
            "algorithm.progress",
            extra={
                "event_body": {
                    "stage": "sampled",
                    "message": f"Selected {len(sample)} historical trajectories",
                    "completed": len(sample),
                    "total": len(sample),
                }
            },
        )
        trajectories = []
        for record, version_id in sample:
            if version_id != base.version_id or record.skill_version_id != base.version_id:
                raise ValueError("Sampled trajectory does not match the selected Skill version")
            success = record.reward_metadata.get("success")
            reward = None
            if record.reward_score is not None and isinstance(success, bool):
                reward = Reward(score=record.reward_score, success=success, detail=record.reward_detail or "")
            trajectories.append(
                SkillTrajectory(
                    trajectory_id=record.trajectory_id,
                    task=Task(id=record.task_id, instruction=record.task_instruction),
                    task_system_prompt=record.task_system_prompt,
                    task_metadata=record.task_metadata,
                    skills={owner.skill_id: Skill.from_snapshot(snapshot_from_version(base))},
                    trajectory=[event for event in record.trajectory if isinstance(event, dict)],
                    reward=reward,
                    status=record.status,
                    observed_score=record.reward_score,
                    feedback_detail=record.reward_detail,
                )
            )
        return trajectories
