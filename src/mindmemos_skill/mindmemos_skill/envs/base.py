"""Execute and resume one persisted skill evaluation rollout."""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import ClassVar

from ..common.asyncio import run_sync
from ..common.input_files import model_file_fingerprints
from ..common.skill_runtime import restore_assembled_skill
from ..registry.base import ConfigT, Registrable
from ..skill import Skill, SkillSet
from .data.typing import RewardData
from .typing import RolloutRequest, RolloutResult, Task


class Environment(Registrable[ConfigT], ABC):
    """Own rollout persistence while subclasses execute agents and score outputs."""

    task_type: ClassVar[type[Task]] = Task
    """Public task model used to restore serialized training requests."""

    reward_data_type: ClassVar[type[RewardData]] = RewardData
    """Private scoring model used to restore serialized training requests."""

    result_type: ClassVar[type[RolloutResult]] = RolloutResult
    """Result schema used for validation and resume; override for typed reward subclasses."""

    supports_resume: ClassVar[bool] = True
    """Whether saved rollout results and assembly can be reused; subclasses may disable this capability."""

    def input_identity(self, task: Task, reward_data: RewardData) -> dict[str, str]:
        """Compute external input identities without requiring caller-supplied hashes.

        Args:
            task: Public task containing typed input file paths.
            reward_data: Private scoring inputs containing typed reference paths.

        Returns:
            File paths mapped to content fingerprints. Subclasses include supporting
            files or external evaluator sources when those also affect execution.

        Raises:
            OSError: An input file cannot be read.
        """
        return model_file_fingerprints(task, reward_data)

    async def rollout(
        self,
        task: Task,
        work_dir: str | Path,
        *,
        reward_data: RewardData,
        skill_set: SkillSet | None = None,
        resume: bool = True,
    ) -> RolloutResult:
        """Execute a task or load its matching completed result.

        Args:
            task: Initial task object passed to the environment implementation.
            work_dir: Root directory for one independent rollout, containing
                private audit/ records and an agent-facing workspace/ directory.
                Concurrent calls must use different roots.
            skill_set: Candidate collection with task-time selection and bounded
                parallel assembly, or None for a baseline run.
            reward_data: Private scoring inputs, persisted only in audit/.
            resume: Reuse a completed result when True. False reruns matching
                inputs without clearing agent-created workspace files. Environments
                with supports_resume=False require resume=False, even on a first run.

        Returns:
            The validated output, trajectory, and reward.

        Raises:
            NotImplementedError: Resume is requested but the environment does not support it.
            ValueError: Stored inputs differ, persisted data is invalid, or
                audit/ or workspace/ is a symbolic link.
            OSError: The workspace or rollout files cannot be read or written.
            Exception: Agent execution or scoring failed; no result is cached.
        """
        if resume and not self.supports_resume:
            raise NotImplementedError(f"{type(self).__name__} does not support resume; pass resume=False")
        skill_set = SkillSet({}, max_concurrency=1) if skill_set is None else skill_set
        if not isinstance(skill_set, SkillSet):
            raise TypeError("skill_set must be a SkillSet")
        skills = skill_set.skills
        request = RolloutRequest(
            task={
                "type": f"{type(task).__module__}.{type(task).__qualname__}",
                "data": task.model_dump(mode="json"),
            },
            reward_data={
                "type": f"{type(reward_data).__module__}.{type(reward_data).__qualname__}",
                "data": reward_data.model_dump(mode="json"),
            },
            skills=[skill.snapshot() for skill in skills.values()],
            skill_set=skill_set.identity(),
            environment=f"{type(self).__module__}.{type(self).__qualname__}",
            config=self.config.model_dump(mode="json"),
            input_identity=await run_sync(self.input_identity, task, reward_data),
        )
        work_dir = Path(work_dir).expanduser().resolve()
        audit_dir = work_dir / "audit"
        workspace = work_dir / "workspace"
        if audit_dir.is_symlink() or workspace.is_symlink():
            raise ValueError("audit/ and workspace/ must be separate directories, not symbolic links")
        audit_dir.mkdir(parents=True, exist_ok=True)
        request_path = audit_dir / "request.json"
        result_path = audit_dir / "result.json"

        previous = None
        if request_path.exists():
            previous = RolloutRequest.model_validate_json(request_path.read_text(encoding="utf-8"))
            source_request = previous.model_copy(
                update={"assembled_skills": None, "assembled_hashes": None, "selected_skill_ids": None}
            )
            # Legacy requests had an ordered static list and no selection policy.
            if previous.skill_set is None and type(skill_set) is SkillSet:
                source_request.skill_set = request.skill_set
            if source_request != request:
                raise ValueError("Rollout inputs changed; use a new work_dir")
        elif result_path.exists():
            raise ValueError("Stored result has no request; use a new work_dir")

        if resume and result_path.exists():
            if previous is None or previous.assembled_skills is None:
                raise ValueError("Stored result has no assembled Skill record")
            result = self.result_type.model_validate_json(result_path.read_text(encoding="utf-8"))
            if result.started_at is None:
                raise ValueError("Stored result has no execution timing; rerun with resume=False")
            return result

        workspace.mkdir(parents=True, exist_ok=True)
        # Remove a previous completion before a forced rerun can fail.
        result_path.unlink(missing_ok=True)
        temporary_request = audit_dir / "request.json.tmp"
        started_at = datetime.now(timezone.utc)
        start = perf_counter()
        if resume and previous is not None and previous.assembled_skills is not None:
            selected_ids = previous.selected_skill_ids
            if selected_ids is None:
                selected_ids = list(skills)
            prepared_skills = [
                restore_assembled_skill(skills[key], snapshot)
                for key, snapshot in zip(selected_ids, previous.assembled_skills, strict=True)
            ]
        else:
            # Clear prior assembly before a forced rerun that may fail to assemble.
            temporary_request.write_text(request.model_dump_json(indent=2), encoding="utf-8")
            temporary_request.replace(request_path)
            selected_ids = list(await run_sync(skill_set.select, task))
            prepared_skills = await skill_set.assemble(task, selected_ids=selected_ids)
        snapshots = [skill.snapshot() for skill in prepared_skills]
        request.assembled_skills = snapshots
        request.assembled_hashes = [snapshot.skill_hash for snapshot in snapshots]
        request.selected_skill_ids = selected_ids
        request = RolloutRequest.model_validate_json(request.model_dump_json())
        temporary_request.write_text(request.model_dump_json(indent=2), encoding="utf-8")
        temporary_request.replace(request_path)
        if await run_sync(self.input_identity, task, reward_data) != request.input_identity:
            raise ValueError("Rollout input files changed during assembly; use a new work_dir")
        result = await run_sync(
            self._run, task=task, reward_data=reward_data, skills=prepared_skills, workspace=workspace
        )
        if await run_sync(self.input_identity, task, reward_data) != request.input_identity:
            raise ValueError("Rollout input files changed during execution; use a new work_dir")
        duration_s = perf_counter() - start
        finished_at = datetime.now(timezone.utc)
        result.started_at = started_at
        result.finished_at = finished_at
        result.duration_s = duration_s
        # Revalidate even mutated model instances before publishing completion.
        result = self.result_type.model_validate_json(result.model_dump_json())
        temporary_path = audit_dir / "result.json.tmp"
        temporary_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        temporary_path.replace(result_path)
        return result

    @abstractmethod
    def _run(self, *, task: Task, reward_data: RewardData, skills: list[Skill], workspace: Path) -> RolloutResult:
        """Run the agent with skills in workspace and score its actual trajectory.

        Skills retain their concrete classes, auxiliary files, and export methods.
        Their content is already assembled for this task; never reassemble here.
        Implementations own agent invocation, task preparation, and reward
        calculation. Pass only workspace and public task materials to the agent.
        Keep reward_data private and use it only for evaluation.
        The backend must restrict agent file access to workspace; setting its
        working directory alone does not prevent access to audit/. Do not change
        the process working directory. Record all execution-affecting settings in
        self.config, including agent/model and evaluator versions. Raise on
        execution or scoring errors; return a Reward for evaluated failures.
        Aggregate LLM usage, including scoring calls, into result.usage. Timing
        is populated by rollout and includes assembly and this method.
        """
