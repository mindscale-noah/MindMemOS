"""Adapt in-memory Skill collections to the evidence-gated file engine."""

import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path

from filelock import AsyncFileLock
from pydantic import JsonValue

from ....common.checkpoint import boundary
from ....common.json import write_json
from ....common.secrets import redact_credentials
from ....common.skill_runtime import load_runtime_skill
from ....skill import Skill, decode_resources, encode_resources
from ....skill.io import read_skill
from ...base import Algorithm
from ..typing import SkillTrajectory
from .config import EvolutionOptions, OfflineEvolutionConfig
from .engine import OfflineEvolution
from .typing import EvolutionInput, EvolutionSnapshot, SkillEvolutionResult


@Algorithm.register("offline_evolution")
class OfflineSkillEvolver(Algorithm[OfflineEvolutionConfig]):
    """Analyze complete inputs and return only permission-checked file candidates."""

    config_type = OfflineEvolutionConfig
    """Native Agent connection and deterministic gate execution settings."""

    async def evolve(
        self,
        *,
        skills: Mapping[str, Skill],
        trajectories: Sequence[SkillTrajectory],
        work_dir: Path,
        options: EvolutionOptions | None = None,
    ) -> SkillEvolutionResult:
        """Analyze all supplied Skills while enforcing an editable subset.

        Args:
            skills: Stable input IDs mapped to complete readable source Skills.
            trajectories: Previously selected historical evidence, never fetched lazily.
            work_dir: Algorithm-owned directory; persistence and recovery belong to the implementation.
            options: Permissions; None allows all input edits and creations.

        Returns:
            Complete input candidates and new Skills without database writes.

        Raises:
            ValueError: Inputs, permissions, candidates or input immutability are invalid.
        """
        root = Path(work_dir).expanduser().resolve()
        request = EvolutionInput(
            skills=dict(skills),
            trajectories=list(trajectories),
            work_dir=root,
            options=options if options is not None else EvolutionOptions(),
        )
        originals = {key: value.snapshot() for key, value in request.skills.items()}
        evidence = _trajectory_payload(request.trajectories)
        result = await self._evolve(request)
        if any(request.skills[key].snapshot().model_dump() != value.model_dump() for key, value in originals.items()):
            raise ValueError("Evolution mutated an input Skill")
        if _trajectory_payload(request.trajectories) != evidence:
            raise ValueError("Evolution mutated historical evidence")
        result.validate_against(originals, request.options)
        return result

    async def _evolve(self, request: EvolutionInput) -> SkillEvolutionResult:
        """Bind exported inputs to this offline run before reusing its checkpoints."""
        originals = {key: value.snapshot() for key, value in request.skills.items()}
        identity = {
            "algorithm": f"{type(self).__module__}.{type(self).__qualname__}",
            "config": redact_credentials(self.config.model_dump(mode="json"), preserve_execution_settings=True),
            "skills": {key: value.model_dump(mode="json") for key, value in originals.items()},
            "trajectories": [
                {
                    **record.model_dump(mode="json", exclude={"skills"}),
                    "skills": {key: value.snapshot().model_dump(mode="json") for key, value in record.skills.items()},
                }
                for record in request.trajectories
            ],
            "public_trajectories": [_public_trajectory(record) for record in request.trajectories],
            "options": request.options.model_dump(mode="json"),
        }
        root = request.work_dir
        root.parent.mkdir(parents=True, exist_ok=True)
        async with AsyncFileLock(str(root.parent / f".{root.name}.lock"), timeout=0, run_in_executor=False):
            manifest = root / "input.json"
            if root.exists() and any(root.iterdir()) and not manifest.exists():
                raise ValueError("Evolution directory has no input checkpoint")
            boundary(manifest, identity, payload={})
            saved = boundary(root / "result.json", identity)
            if saved is not None:
                snapshots = EvolutionSnapshot.model_validate(saved)
                return SkillEvolutionResult(
                    skills={key: load_runtime_skill(value) for key, value in snapshots.skills.items()},
                    created_skills={key: load_runtime_skill(value) for key, value in snapshots.created_skills.items()},
                )
            result = await self._run(request)
            result.validate_against(originals, request.options)
            boundary(root / "result.json", identity, payload=result.snapshot().model_dump(mode="json"))
            return result

    async def _run(self, request: EvolutionInput) -> SkillEvolutionResult:
        """Rebuild complete public exports, resume the file engine, and restore candidate Skills."""
        sources = request.work_dir / "source_skills"
        evidence = request.work_dir / "source_trajectories"
        names = {key: value.resolve_export_directory(sources).name for key, value in request.skills.items()}
        if len(set(names.values())) != len(names):
            raise ValueError("Input Skills have colliding export directory names")
        for directory in (sources, evidence):
            if directory.exists():
                shutil.rmtree(directory)
        for skill in request.skills.values():
            skill.export(sources)
        evidence.mkdir(parents=True)
        for index, record in enumerate(request.trajectories):
            write_json(
                evidence / f"trajectory-{index + 1:06d}.json",
                _public_trajectory(record),
            )
        editable = request.options.allowed_ids(request.skills)
        options = EvolutionOptions(
            editable_skills=[names[key] for key in request.skills if key in editable],
            allow_create_skills=request.options.allow_create_skills,
        )
        result = await OfflineEvolution(self.config, options).evolve(
            skills=sources, trajectories=evidence, work_dir=request.work_dir / "engine"
        )
        survivors = {}
        for key, source in request.skills.items():
            directory = result.skills / names[key]
            if not (directory / "SKILL.md").is_file():
                raise ValueError(f"Evolution must retain input Skill {key!r}")
            survivors[key] = _read_candidate(source, directory)
        created = {}
        for directory in sorted(result.skills.iterdir()):
            if directory.name in names.values():
                continue
            if not directory.is_dir() or not (directory / "SKILL.md").is_file():
                raise ValueError("Evolution output contains an incomplete new Skill")
            created[f"new/{directory.name}"] = Skill.from_path(directory)
        return SkillEvolutionResult(skills=survivors, created_skills=created)


def _read_candidate(source: Skill, directory: Path) -> Skill:
    """Read existing members using their original format and runtime implementation."""
    if source.source.standard_format:
        candidate = read_skill(directory)
        return source.with_content(candidate.content).with_resources(candidate.resources)
    entrypoint = directory / "SKILL.md"
    if entrypoint.is_symlink():
        raise ValueError("SKILL.md must not be a symbolic link")
    from mindmemos_sdk.skills.cli_files import collect_resources

    resources = decode_resources(collect_resources(directory))
    return source.with_content(entrypoint.read_bytes().decode("utf-8")).with_resources(resources)


def _historical_skill(skill: Skill):
    """Expose complete historical source bytes without runtime credentials or record-only digests."""
    source = skill.source
    return {
        "content": source.content,
        "resources": encode_resources(source.resources),
        "name": source.name,
        "description": source.description,
        "standard_format": source.standard_format,
    }


def _public_trajectory(record: SkillTrajectory) -> dict[str, JsonValue]:
    """Preserve recorded events and execution status in the evidence visible to S1."""
    return {
        "trajectory_id": record.trajectory_id,
        "task": record.task.model_dump(mode="json"),
        "task_system_prompt": record.task_system_prompt,
        "task_metadata": record.task_metadata,
        "skills_used": {key: _historical_skill(value) for key, value in record.skills.items()},
        "trajectory": record.trajectory,
        "status": record.status,
    }


def _trajectory_payload(records):
    """Capture detached record fields and complete historical Skill snapshots for identity checks."""
    return [
        {
            **record.model_dump(mode="json", exclude={"skills"}),
            "skills": {key: value.snapshot().model_dump(mode="json") for key, value in record.skills.items()},
        }
        for record in records
    ]
