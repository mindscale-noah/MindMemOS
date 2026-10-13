"""Execute an artifact-producing task and score official pass/fail outcomes."""

import json
from pathlib import Path
from tempfile import mkdtemp

from mindmemos_skill.common.ids import generate_id

from ....agents.buitin_agents.codex.agent import CodexAgent
from ....infra.execution import execute
from ....skill import Skill
from ...base import Environment
from ...data import RewardData
from ...typing import Reward, RolloutResult, Task
from ..common import total_usage
from .config import AgentSkillOSConfig
from .reward import compute_metrics
from .source import evaluator_digest, relative_path
from .typing import AgentSkillOSRewardData, AgentSkillOSTask, Ranking


@Environment.register("agentskillos")
class AgentSkillOSEnvironment(Environment[AgentSkillOSConfig]):
    """Use official task success as reward and retain retrieval only as diagnostics."""

    task_type = AgentSkillOSTask
    """Public AgentSkillOS task schema."""

    reward_data_type = AgentSkillOSRewardData
    """Private AgentSkillOS scoring schema."""

    config_type = AgentSkillOSConfig
    """Native agent and private official scoring settings."""

    def __init__(self, config: AgentSkillOSConfig, *, agent: CodexAgent | None = None) -> None:
        """Construct the executor without starting a model or subprocess.

        Args:
            config: Explicit execution model and fingerprinted official scorer.
            agent: Optional matching executor for testing or dependency injection.
        """
        super().__init__(config)
        self._evaluator_digest = evaluator_digest(config.benchmark_root)
        if agent is not None and agent.config != config.agent:
            raise ValueError("Injected agent config must match environment agent config")
        self.agent = agent if agent is not None else CodexAgent(config.agent)

    def input_identity(self, task: Task, reward_data: RewardData) -> dict[str, str]:
        """Bind public files and official evaluator sources to recovery records.

        Args:
            task: Public task and input files.
            reward_data: Private official evaluation definition.

        Returns:
            File and evaluator identities computed from the current sources.
        """
        identity = super().input_identity(task, reward_data)
        digest = evaluator_digest(self.config.benchmark_root)
        if digest != self._evaluator_digest:
            raise ValueError("Official evaluator sources changed; construct a new environment")
        identity[f"evaluator:{self.config.benchmark_root}"] = digest
        return identity

    async def _score(self, task_path: Path, output_dir: Path, report_path: Path, *, preflight: bool = False) -> None:
        """Run the isolated official scorer and fail on process or setup errors."""
        if evaluator_digest(self.config.benchmark_root) != self._evaluator_digest:
            raise ValueError("Official evaluator sources changed; construct a new environment")
        command = [
            *self.config.evaluator.python_command,
            "-I",
            str(Path(__file__).with_name("evaluator_worker.py")),
            "--source",
            str(self.config.benchmark_root),
            "--task",
            str(task_path),
            "--workspace",
            str(output_dir),
            "--output",
            str(report_path),
        ]
        if preflight:
            command.append("--preflight")
        result = await execute(command, task_path.parent, self.config.evaluator)
        if result.returncode:
            raise RuntimeError(f"Official AgentSkillOS scoring failed: {result.stderr}")

    async def _run(self, *, task: Task, reward_data: RewardData, skills: list[Skill], workspace: Path) -> RolloutResult:
        """Prepare public materials, execute native tools, then privately score artifacts."""
        from ....skill.content import read_frontmatter

        if not isinstance(task, AgentSkillOSTask) or not isinstance(reward_data, AgentSkillOSRewardData):
            raise TypeError("AgentSkillOS requires official task and private evaluator data")
        if (
            reward_data.task_config.get("task_id") != task.id
            or reward_data.task_config.get("prompt") != task.instruction
        ):
            raise ValueError("Public task and private official definition must match")
        ranking = []
        for skill in skills:
            metadata = read_frontmatter(skill.content)
            ranking.append(metadata.get("name"))
        ranking = Ranking(skill_ids=ranking).skill_ids
        for name in ranking:
            if "/" in relative_path(name):
                raise ValueError("Skill names must be single path components")
        # Fresh execution roots preserve failed attempts and prevent stale artifact passes.
        attempt = Path(mkdtemp(prefix="attempt-", dir=workspace))
        outputs = attempt / "workspace"
        outputs.mkdir()
        for name, source in task.files.items():
            relative_path(name)
            if name.split("/")[0] == "workspace":
                raise ValueError("Public inputs must not prepopulate the output directory")
            content = source.read_bytes()
            destination = attempt / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
        for name, skill in zip(ranking, skills, strict=True):
            skill.export(attempt / ".agents" / "skills" / name)
        audit = workspace.parent / "audit"
        identifier = generate_id()
        task_path = audit / f"official-task-{identifier}.json"
        report_path = audit / f"official-score-{identifier}.json"
        task_path.write_text(json.dumps(reward_data.task_config, ensure_ascii=False), encoding="utf-8")
        (await self._score(task_path, outputs, report_path, preflight=True))
        system = (
            "Complete the user's task by producing actual deliverable files. "
            "Available Skills and their resources are in .agents/skills/. Read relevant SKILL.md files "
            "and use their instructions and tools. Input paths in the task are relative to your working directory. "
            "Write all final deliverables, including supporting output files, under workspace/. "
            "Inspect and validate the deliverables before reporting completion. "
            "Do not read files outside this execution directory."
        )
        user = task.instruction + "\n\nOutput directory: workspace/\nExpected deliverables:\n" + "\n".join(task.outputs)
        response = await self.agent.run(system=system, user=user, workspace=attempt)
        if not isinstance(response.output, str):
            raise TypeError("AgentSkillOS executor must return a text completion")
        # Avoid allowing generated links to redirect private scoring outside the outputs.
        if outputs.is_symlink() or any(path.is_symlink() for path in outputs.rglob("*")):
            raise ValueError("Deliverables must not contain symlinks")
        (await self._score(task_path, outputs, report_path))
        official = json.loads(report_path.read_text(encoding="utf-8"))
        if official.get("task_id") != task.id or type(official.get("passed")) is not bool:
            raise ValueError("Official scoring returned a mismatched or invalid task result")
        retrieval = {
            "ranked_skill_ids": ranking,
            "label_source": reward_data.relevance_source,
            "metrics": compute_metrics(ranking, reward_data.core_gt_ids) if reward_data.core_gt_ids else None,
        }
        details = {"task_evaluation": official, "retrieval": retrieval}
        return RolloutResult(
            output={"response": response.output, "artifact_directory": str(outputs.relative_to(workspace))},
            trajectory=response.trajectory,
            usage=total_usage([response]),
            reward=Reward(
                score=float(official["passed"]),
                success=official["passed"],
                detail=json.dumps(details, ensure_ascii=False, sort_keys=True),
            ),
        )
