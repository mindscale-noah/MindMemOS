"""Generate complete Skill outputs and publish their versions atomically."""

from pathlib import Path

from filelock import AsyncFileLock

from ....algos import Algorithm
from ....algos.evolve.offline_evolution.config import EvolutionOptions
from ....algos.evolve.offline_evolution.typing import EvolutionSnapshot, HistoricalTrajectory, SkillEvolutionResult
from ....api.typing import (
    CollectionEvolutionRequest,
    EvolutionRequest,
    EvolutionRunResult,
    SkillPipelineRequest,
)
from ....common.checkpoint import boundary, fingerprint, read_boundary
from ....common.ids import generate_id
from ....common.json import read_json, write_json
from ....common.secrets import redact_credentials
from ....common.skill_runtime import load_runtime_skill
from ....common.trajectories import collect_skill_trajectories
from ....envs.typing import Reward
from ....errors import SkillConflictError, SkillNotFoundError
from ....logging import get_logger, stage
from ....persistence import TrajectoryRecord
from ....persistence.store import SkillPersistence
from ....skill import Skill
from ...base import Pipeline
from ...evolve import EvolutionPipeline
from .config import OfflineEvolutionPipelineConfig
from .publication import publish_candidate, read_publication
from .typing import CapturedEvolutionInputs, EvolutionInputMember, EvolutionRunManifest, HistoricalTask


@Pipeline.register("offline_evolution")
class OfflineEvolutionPipeline(EvolutionPipeline[OfflineEvolutionPipelineConfig]):
    """Freeze readable inputs, generate candidates, and publish with optional activation."""

    config_type = OfflineEvolutionPipelineConfig
    """Algorithm selection, run directory and evidence bounds."""

    def __init__(self, config: OfflineEvolutionPipelineConfig, *, persistence: SkillPersistence):
        """Build a registered evolver and borrow the global repository.

        Args:
            config: Validated algorithm, directory and evidence collection settings.
            persistence: Shared project-scoped Skill persistence.
        """
        if not isinstance(persistence, SkillPersistence):
            raise TypeError("Evolution pipeline requires SkillPersistence")
        super().__init__(config, persistence=persistence)
        self.evolver = Algorithm.build(config.evolver.class_name, config.evolver.config)

    async def prepare_request(self, request: EvolutionRequest) -> CollectionEvolutionRequest:
        """Capture one exact version or validate a complete readable collection.

        Args:
            request: Authorized single-Skill selection or exact version collection.

        Returns:
            Exact owned collection with validated editing permissions.
        """
        if isinstance(request, SkillPipelineRequest):
            owner = await self.persistence.skills.resolve_family(request.skill_ref)
            version = await self.persistence.skills.version(owner, request.base_version_id)
            request = CollectionEvolutionRequest(
                project_id=request.project_id,
                pipeline=request.pipeline,
                skill_version_ids=[version.version_id],
                auto_accept=request.auto_accept,
                run_id=request.run_id,
            )
        request = CollectionEvolutionRequest.model_validate(request)
        versions = [await self.persistence.skills.get(value) for value in request.skill_version_ids]
        owners = {version.skill_id for version in versions}
        if len(owners) != len(versions):
            raise ValueError("Evolution accepts at most one version per Skill family")
        if request.editable_skills is not None and not set(request.editable_skills) <= owners:
            raise ValueError("Editable Skill IDs must belong to the input collection")
        return request

    async def evolve(self, request: EvolutionRequest) -> EvolutionRunResult:
        """Generate complete outputs and save versions with optional activation.

        Args:
            request: Authorized exact versions, mutation permissions and activation choice.

        Returns:
            Persisted versions and snapshots for every input and newly created Skill.

        Raises:
            ValueError: Inputs, evidence or checkpoint contents are invalid.
            SkillConflictError: A run is reused with different execution inputs.
        """
        request = await self.prepare_request(request)
        request = request.model_copy(deep=True, update={"run_id": request.run_id or generate_id("evolve_")})
        configuration = redact_credentials(
            {
                "evolver": {
                    "class_name": self.config.evolver.class_name,
                    "config": self.evolver.config.model_dump(mode="json"),
                },
                "traj_collect": self.config.traj_collect.model_dump(mode="json"),
            },
            preserve_execution_settings=True,
        )
        identity = {"request": request.model_dump(mode="json", exclude={"run_id"}), "configuration": configuration}
        input_hash = fingerprint(identity)
        root, directory = self._directory(request.run_id)
        async with AsyncFileLock(str(root / f".{request.run_id}.lock"), run_in_executor=False):
            path = directory / "config.json"
            if path.exists():
                manifest = self._manifest(directory, request.project_id)
                if manifest.input_hash != input_hash:
                    raise SkillConflictError("Evolution run was configured with different inputs")
            else:
                if directory.exists() and any(directory.iterdir()):
                    raise ValueError("Evolution directory has no config.json; use a new run_id")
                manifest = EvolutionRunManifest(
                    run_id=request.run_id,
                    project_id=request.project_id,
                    input_hash=input_hash,
                    request=request,
                    algorithm=self.config.evolver.class_name,
                    configuration=configuration,
                )
                write_json(path, manifest)
            published = read_publication(directory, manifest)
            if published is not None:
                return published
            candidate_path = directory / "candidate.json"
            if candidate_path.exists():
                captured, candidate = self._candidate(directory, manifest)
            else:
                with stage("evolve.prepare"):
                    captured = await self._capture(directory, request, input_hash)
                    trajectories = await self._evidence(directory, captured.inputs)
                options = EvolutionOptions(
                    editable_skills=request.editable_skills, allow_create_skills=request.allow_create_skills
                )
                with stage("evolve.evolving"):
                    result = await self.evolver.evolve(
                        skills={key: load_runtime_skill(item.base.snapshot()) for key, item in captured.inputs.items()},
                        trajectories=trajectories,
                        work_dir=directory / "algorithm",
                        options=options,
                    )
                candidate = result.snapshot()
                boundary(candidate_path, input_hash, payload=candidate.model_dump(mode="json"))
            await self._consume(directory, captured.inputs)
            with stage("evolve.publish"):
                result = await publish_candidate(self.persistence, directory, manifest, captured, candidate)
            self._progress("published", f"Published {len(result.members)} member outcomes")
            return result

    def _directory(self, run_id):
        """Resolve the validated execution ID beneath this pipeline's configured root."""
        root = Path(self.config.work_dir).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        return root, root / run_id

    @staticmethod
    def _manifest(directory, project_id):
        """Read a scoped execution manifest without exposing foreign project artifacts."""
        path = directory / "config.json"
        if not path.is_file():
            raise SkillNotFoundError("Evolution run not found")
        manifest = EvolutionRunManifest.model_validate(read_json(path))
        if (
            manifest.project_id != project_id
            or manifest.request.project_id != project_id
            or manifest.run_id != directory.name
            or manifest.request.run_id != manifest.run_id
        ):
            raise SkillNotFoundError("Evolution run not found in this project")
        expected = fingerprint(
            {
                "request": manifest.request.model_dump(mode="json", exclude={"run_id"}),
                "configuration": manifest.configuration,
            }
        )
        if expected != manifest.input_hash:
            raise SkillConflictError("Evolution manifest integrity changed")
        return manifest

    @staticmethod
    def _candidate(directory, manifest):
        """Validate a complete frozen candidate against its captured request."""
        path = directory / "candidate.json"
        if not path.is_file():
            raise SkillNotFoundError("Evolution candidate is not complete")
        saved = read_boundary(path)
        if saved.identity != fingerprint(manifest.input_hash):
            raise SkillConflictError("Evolution candidate belongs to different inputs")
        candidate = EvolutionSnapshot.model_validate(saved.payload)
        captured = CapturedEvolutionInputs.model_validate(boundary(directory / "request.json", manifest.input_hash))
        expected = set(manifest.request.skill_version_ids)
        if {item.base.version_id for item in captured.inputs.values()} != expected:
            raise SkillConflictError("Candidate source versions changed")
        if any(
            key != item.owner.skill_id or item.owner.project_id != manifest.project_id
            for key, item in captured.inputs.items()
        ):
            raise SkillConflictError("Candidate ownership changed")
        options = EvolutionOptions(
            editable_skills=manifest.request.editable_skills,
            allow_create_skills=manifest.request.allow_create_skills,
        )
        SkillEvolutionResult(
            skills={key: Skill.from_snapshot(value) for key, value in candidate.skills.items()},
            created_skills={key: Skill.from_snapshot(value) for key, value in candidate.created_skills.items()},
        ).validate_against({key: item.base.snapshot() for key, item in captured.inputs.items()}, options)
        return captured, candidate

    async def _capture(self, directory, request, input_hash):
        """Freeze visible owners and exact versions once, before algorithm execution."""
        path = directory / "request.json"
        if path.exists():
            return CapturedEvolutionInputs.model_validate(boundary(path, input_hash))
        inputs = {}
        for version_id in request.skill_version_ids:
            owner, base = await self.persistence.skills.get_with_owner(version_id)
            if owner.skill_id in inputs:
                raise ValueError("Evolution accepts at most one version per Skill family")
            inputs[owner.skill_id] = EvolutionInputMember(owner=owner, base=base)
        EvolutionOptions(
            editable_skills=request.editable_skills, allow_create_skills=request.allow_create_skills
        ).allowed_ids(inputs)
        captured = CapturedEvolutionInputs(inputs=inputs)
        boundary(path, input_hash, payload=captured.model_dump(mode="json"))
        return captured

    async def _evidence(self, directory, inputs):
        """Collect all matching evidence within optional read limits and freeze it once."""
        path = directory / "evidence.json"
        by_version = {item.base.version_id: item for item in inputs.values()}
        if path.exists():
            records = [TrajectoryRecord.model_validate(item) for item in read_json(path)["samples"]]
        else:
            pool = {}
            for version_id in by_version:
                limit = self.config.traj_collect.trajectory_limit
                if limit is None:
                    limit = (await self.persistence.trajectories.counts(version_id)).total
                if limit == 0:
                    continue
                rows = await collect_skill_trajectories(
                    self.persistence,
                    skill_version_id=version_id,
                    limit=limit,
                    max_page_size=self.config.traj_collect.max_page_size,
                )
                for record in rows:
                    if record.skill_version_id == version_id:
                        pool[record.trajectory_id] = record
            if not pool:
                raise ValueError("No trajectories available for the supplied Skill versions")
            records = list(pool.values())
            write_json(path, {"samples": [item.model_dump(mode="json") for item in records]})
            write_json(
                directory / "input.json",
                {
                    "trajectory_count": len(records),
                    "trajectory_ids": [item.trajectory_id for item in records],
                    "input_version_ids": list(by_version),
                },
            )
        trajectories = []
        for record in records:
            if record.skill_version_id not in by_version:
                raise ValueError("Collected evidence does not match a supplied exact version")
            member = by_version[record.skill_version_id]
            success = record.reward_metadata.get("success")
            reward = None
            if record.reward_score is not None and isinstance(success, bool):
                reward = Reward(score=record.reward_score, success=success, detail=record.reward_detail or "")
            trajectories.append(
                HistoricalTrajectory(
                    trajectory_id=record.trajectory_id,
                    task=HistoricalTask(id=record.task_id, instruction=record.task_instruction),
                    task_system_prompt=record.task_system_prompt,
                    task_metadata=record.task_metadata,
                    skills={member.owner.skill_id: Skill.from_snapshot(member.base.snapshot())},
                    trajectory=record.trajectory,
                    reward=reward,
                    status=record.status,
                    observed_score=record.reward_score,
                    feedback_detail=record.reward_detail,
                )
            )
        self._progress("evidence_ready", f"Collected {len(trajectories)} historical trajectories")
        return trajectories

    async def _consume(self, directory, inputs):
        """Record candidate-producing evidence use without changing repository Skills."""
        saved = read_json(directory / "evidence.json")
        records = [TrajectoryRecord.model_validate(item) for item in saved["samples"]]
        for member in inputs.values():
            ids = [item.trajectory_id for item in records if item.skill_version_id == member.base.version_id]
            await self.persistence.trajectories.mark_consumed(member.base.version_id, ids, directory.name)

    @staticmethod
    def _progress(name, message):
        """Report actual completed boundaries through the existing task logger."""
        get_logger(__name__).info("algorithm.progress", extra={"event_body": {"stage": name, "message": message}})
