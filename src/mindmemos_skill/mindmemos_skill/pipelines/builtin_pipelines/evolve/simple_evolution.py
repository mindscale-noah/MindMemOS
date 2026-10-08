"""Stored-evidence Skill evolution pipeline."""

import random
from pathlib import Path
from uuid import uuid4

from filelock import AsyncFileLock

from ....algos import Algorithm
from ....algos.evolve.typing import SkillTrajectory
from ....api.typing import AlgoSkillEvolveResult, SkillPipelineRequest
from ....common.asyncio import run_sync
from ....common.checkpoint import boundary, fingerprint
from ....common.json import read_json, write_json
from ....common.skill_runtime import load_runtime_skill
from ....common.skill_versions import snapshot_from_version
from ....common.time_utils import utcnow
from ....common.trajectories import collect_skill_trajectories
from ....envs.typing import Reward, Task
from ....infra.task_manager.progress import ProgressReporter
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

    async def evolve(
        self, request: SkillPipelineRequest, *, reporter: ProgressReporter | None = None
    ) -> AlgoSkillEvolveResult:
        """Collect, randomly sample, reflect and publish an exact child version.

        Args:
            request: Authorized project, Skill reference and optional expected parent.
            reporter: Optional progress sink for evidence, evolution, and publication.

        Returns:
            Identity of the version durably published by this run.

        Raises:
            ValueError: No usable evidence or no changed candidate exists.
            SkillConflictError: The current selection changed before publication.
        """
        request = SkillPipelineRequest.model_validate(request)
        run_id = request.run_id or "evolve_" + uuid4().hex
        root = Path(self.config.work_dir).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        directory = root / run_id
        identity = {"request": request.model_dump(mode="json"), "config": self.config.model_dump(mode="json")}
        input_hash = fingerprint(identity)
        store = self.persistence.skills
        async with AsyncFileLock(str(root / f".{run_id}.lock")):
            published = await run_sync(
                store.algorithm_publication, request.project_id, run_id, input_hash, operation="evolution"
            )
            if published is not None:
                return self._result(published, run_id)
            saved = boundary(directory / "request.json", identity)
            if saved is None:
                owner = await run_sync(store.resolve_family, request.project_id, request.skill_ref)
                base = await run_sync(store.version, owner, request.base_version_id)
                saved = boundary(
                    directory / "request.json",
                    identity,
                    payload={"owner": owner.model_dump(mode="json"), "base_version_id": base.version_id},
                )
            owner = SkillRecord.model_validate(saved["owner"])
            base = await run_sync(store.version_by_id, request.project_id, saved["base_version_id"])
            candidate_path = directory / "candidate.json"
            if candidate_path.exists():
                snapshot = SkillSnapshot.model_validate(read_json(candidate_path))
            else:
                trajectories = await self._evidence(request, owner, base, directory, input_hash, reporter)
                candidate = await self.evolver.evolve(
                    skill=load_runtime_skill(snapshot_from_version(base)), trajectories=trajectories
                )
                snapshot = candidate.snapshot()
                write_json(candidate_path, snapshot)
                if reporter is not None:
                    await reporter.report(stage="evolved", message="Skill candidate created")
            version = await run_sync(
                store.commit_algorithm,
                owner,
                snapshot,
                run_id=run_id,
                input_hash=input_hash,
                operation="evolution",
                algorithm=self.config.evolver.class_name.replace("_", "-"),
                commit_source=self.config.evolver.class_name,
                base_version_id=base.version_id,
                active=request.auto_accept,
            )
            if reporter is not None:
                await reporter.report(stage="published", message="Evolved Skill version published")
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

    async def _evidence(self, request, owner, base, directory, input_hash, reporter):
        """Reuse the exact sampled records after an interrupted model call."""
        store = self.persistence.skills
        evidence_path = directory / "evidence.json"
        saved = boundary(evidence_path, input_hash)
        if saved is not None:
            sample = [
                (TrajectoryRecord.model_validate(item["record"]), item["version_id"]) for item in saved["samples"]
            ]
        else:
            records = await run_sync(
                collect_skill_trajectories,
                self.persistence,
                skill_id=owner.skill_id,
                limit=self.config.trajectory_limit,
                scan_limit=self.config.scan_limit,
                created_from=self.config.created_from,
                created_to=self.config.created_to or utcnow(),
                status=self.config.status,
                source=self.config.source,
                max_record_bytes=self.config.max_record_bytes,
                max_total_bytes=self.config.max_total_bytes,
            )
            eligible = []
            for record in records:
                if not record.task_instruction.strip():
                    continue
                version_id = record.skill_version_id or _bound_version(record.skill_bindings, owner.skill_id, store)
                if version_id is None:
                    version_id = _bound_version(record.injected_skills, owner.skill_id, store)
                if version_id is None:
                    continue
                historical = await run_sync(store.get, version_id)
                if (
                    historical is None
                    or historical.project_id != request.project_id
                    or historical.skill_id != owner.skill_id
                ):
                    continue
                eligible.append((record, version_id))
            if reporter is not None:
                await reporter.report(
                    stage="evidence",
                    message=f"Collected {len(records)} trajectories; {len(eligible)} eligible",
                    completed=len(eligible) if records else None,
                    total=len(records) if records else None,
                )
            if not eligible:
                raise ValueError("No eligible historical trajectories for this Skill")
            if len(eligible) < self.config.sample_size:
                raise ValueError("Fewer eligible trajectories than sample_size")
            sample = (
                random.SystemRandom().sample(eligible, self.config.sample_size)
                if self.config.seed is None
                else random.Random(self.config.seed).sample(eligible, self.config.sample_size)
            )
            boundary(
                evidence_path,
                input_hash,
                payload={
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
                    "config": self.config.model_dump(mode="json"),
                },
            )
        if reporter is not None:
            await reporter.report(
                stage="sampled",
                message=f"Selected {len(sample)} historical trajectories",
                completed=len(sample),
                total=len(sample),
            )
        trajectories = []
        for record, version_id in sample:
            historical = await run_sync(store.version_by_id, request.project_id, version_id)
            success = record.reward_metadata.get("success")
            reward = None
            if record.reward_score is not None and isinstance(success, bool):
                reward = Reward(score=record.reward_score, success=success, detail=record.reward_detail or "")
            trajectories.append(
                SkillTrajectory(
                    task=Task(id=record.task_id, instruction=record.task_instruction),
                    task_system_prompt=record.task_system_prompt,
                    task_metadata=record.task_metadata,
                    skills={owner.skill_id: Skill.from_snapshot(snapshot_from_version(historical))},
                    trajectory=[event for event in record.trajectory if isinstance(event, dict)],
                    reward=reward,
                    status=record.status,
                    observed_score=record.reward_score,
                    feedback_detail=record.reward_detail,
                )
            )
        return trajectories


def _bound_version(bindings: list, skill_id: str, store) -> str | None:
    """Find a historical exact version in a local execution binding."""
    for binding in bindings:
        if not isinstance(binding, dict):
            continue
        value = binding.get("version_id")
        if isinstance(value, str) and value:
            version = store.get(value)
            if version is not None and version.skill_id == skill_id:
                return value
    return None
