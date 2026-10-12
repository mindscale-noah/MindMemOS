"""Run one local Skill training benchmark through the in-process service."""

from hashlib import sha256
from pathlib import Path

from filelock import AsyncFileLock
from mindmemos_skill.api import create_service
from mindmemos_skill.api.typing import FileSkillInfo, RegisterFromLocalRequest, ShowRequest, TrainingRequest
from mindmemos_skill.common.json import read_json, write_json
from mindmemos_skill.errors import SkillAliasConflictError, SkillNotFoundError
from mindmemos_skill.skill import Skill

from .preflight import LocalSkillEvalPreflight
from .report import build_report
from .typing import LocalRunManifest, PreparedInputs, SkillEvalReport


class LocalSkillEvalRunner:
    """Own one run directory, service lifetime, and exact Skill version lineage."""

    async def run(self, config_path: str | Path, output_dir: str | Path) -> SkillEvalReport:
        """Train and report one configured benchmark, resuming compatible work.

        Args:
            config_path: Experiment YAML describing one algorithm and environment.
            output_dir: Isolated directory for this run and its recoverable artifacts.

        Returns:
            Report of the training pipeline's completed final test split.

        Raises:
            ValueError: Existing run artifacts conflict with current inputs.
        """
        prepared = LocalSkillEvalPreflight().prepare(config_path)
        await LocalSkillEvalPreflight().check_runtime(prepared)
        root = Path(output_dir).expanduser().resolve()
        skill_path = prepared.config.initial_skill.path
        skill_root = skill_path if skill_path.is_dir() else skill_path.parent
        if root.is_relative_to(skill_root):
            raise ValueError("Run directory must not be inside the initial Skill source")
        root.parent.mkdir(parents=True, exist_ok=True)
        lock_path = root.parent / f".{root.name}.eval.lock"
        async with AsyncFileLock(str(lock_path), timeout=0):
            try:
                return await self._run_locked(prepared, root)
            except Exception:
                manifest_path = root / "manifest.json"
                if manifest_path.exists():
                    saved = LocalRunManifest.model_validate(read_json(manifest_path))
                    if saved.input_digest == prepared.input_digest:
                        write_json(manifest_path, saved.model_copy(update={"status": "failed"}))
                raise

    async def _run_locked(self, prepared: PreparedInputs, root: Path) -> SkillEvalReport:
        """Run one validated experiment while holding its output directory lock."""
        manifest_path = root / "manifest.json"
        if manifest_path.exists():
            manifest = LocalRunManifest.model_validate(read_json(manifest_path))
            if (
                manifest.input_digest != prepared.input_digest
                or manifest.project_id != prepared.service.fastapi.auth.default_project_id
            ):
                raise ValueError("Run directory belongs to different evaluation inputs")
        else:
            if root.exists() and any(root.iterdir()):
                raise ValueError("Run directory is nonempty and has no evaluation manifest")
            manifest = LocalRunManifest(
                input_digest=prepared.input_digest,
                config_path=str(prepared.config_path),
                project_id=prepared.service.fastapi.auth.default_project_id,
                algorithm=prepared.config.algorithm,
                environment=prepared.config.environment.class_name,
                train_ids=prepared.split.train,
                val_ids=prepared.split.val,
                test_ids=prepared.split.test,
            )
            write_json(manifest_path, manifest)

        async with await create_service(prepared.service) as service:
            if manifest.base_version_id is None:
                alias_identity = f"{manifest.project_id}:{root}:{manifest.input_digest}"
                alias = "eval-" + sha256(alias_identity.encode("utf-8")).hexdigest()[:24]
                request = ShowRequest(project_id=manifest.project_id, skill_id=alias)
                try:
                    detail = await service.show_skill(request)
                except SkillNotFoundError:
                    try:
                        await service.register_from_local(
                            RegisterFromLocalRequest(
                                project_id=manifest.project_id,
                                skill=FileSkillInfo(path=str(prepared.config.initial_skill.path)),
                                alias=alias,
                            )
                        )
                        detail = await service.show_skill(request)
                    except SkillAliasConflictError:
                        detail = await service.show_skill(request)
                source_snapshot = Skill.from_path(prepared.config.initial_skill.path).snapshot()
                if detail.version.skill_hash != source_snapshot.skill_hash:
                    raise ValueError("Recovered Skill alias points to a different initial source")
                manifest = manifest.model_copy(
                    update={
                        "skill_id": detail.skill.skill_id,
                        "base_version_id": detail.version.version_id,
                        "status": "registered",
                    }
                )
                write_json(manifest_path, manifest)
            else:
                base = await service.get_skill_version(manifest.base_version_id, project_id=manifest.project_id)
                if base.skill_id != manifest.skill_id:
                    raise ValueError("Registered Skill version does not match the run manifest")

            result = await service.run_training(
                TrainingRequest(
                    algo=prepared.config.algorithm,
                    env=prepared.config.environment,
                    project_id=manifest.project_id,
                    init_skill_version_id=manifest.base_version_id,
                    work_dir=str(root / "training"),
                    tasks=prepared.samples,
                    split=prepared.split,
                )
            )
            published = await service.get_skill_version(result.version_id, project_id=manifest.project_id)
            if published.skill_id != manifest.skill_id or result.base_version_id != manifest.base_version_id:
                raise ValueError("Published Skill lineage does not match the run manifest")
            manifest = manifest.model_copy(update={"published_version_id": result.version_id, "status": "published"})
            write_json(manifest_path, manifest)

        report = build_report(manifest, root / "training")
        write_json(root / "summary.json", report)
        manifest = manifest.model_copy(update={"status": "completed"})
        write_json(manifest_path, manifest)
        return report


async def run_local_eval(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    preflight_only: bool = False,
) -> SkillEvalReport | None:
    """Run preflight or the complete local benchmark from the CLI.

    Args:
        config_path: Path to one experiment YAML.
        output_dir: Run directory, unused during preflight.
        preflight_only: Validate inputs without creating a service or run files.

    Returns:
        Completed report, or None after successful preflight.
    """
    if preflight_only:
        prepared = LocalSkillEvalPreflight().prepare(config_path)
        await LocalSkillEvalPreflight().check_runtime(prepared)
        return None
    return await LocalSkillEvalRunner().run(config_path, output_dir)
