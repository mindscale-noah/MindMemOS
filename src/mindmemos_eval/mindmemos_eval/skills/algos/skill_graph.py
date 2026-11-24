"""Evaluate Skill Graph recommendations on AgentSkillOS."""

import asyncio
from pathlib import Path

from mindmemos_skill.api.typing import OrganizeRequest, RecommendRequest, RegisterRequest, StructuredSkillInfo
from mindmemos_skill.common.json import write_json
from mindmemos_skill.common.secrets import redact_credentials
from mindmemos_skill.config.config import ServiceConfig
from mindmemos_skill.config.loader import read_skill_config
from mindmemos_skill.envs.builtin_envs.agentskillos import (
    AgentSkillOSConfig,
    AgentSkillOSData,
    AgentSkillOSDataConfig,
    AgentSkillOSEnvironment,
)
from mindmemos_skill.envs.builtin_envs.agentskillos.reward import aggregate_metrics, compute_metrics
from mindmemos_skill.envs.builtin_envs.agentskillos.typing import RelevanceRecord, SkillID
from mindmemos_skill.envs.typing import Task
from mindmemos_skill.skill import Skill, SkillSet
from mindmemos_skill.skill.content import encode_resources
from pydantic import JsonValue, TypeAdapter

from ..config import SkillGraphAccuracyConfig
from ..rollout import run_tasks
from ..service import open_skill_service
from ..typing import AgentSkillOSRetrievalTask, RetrievalScoreReport, SkillGraphAccuracyReport

PROJECT_ID = "agentskillos-eval"
RETRIEVAL_TOP_K = 50
GRAPH_BUILD_TIMEOUT_S = 7200


async def run_retrieval(tasks_dir: Path, skills_dir: Path, service_config: Path, output: Path) -> RetrievalScoreReport:
    """Generate and score rankings against official AgentSkillOS labels.

    Args:
        tasks_dir: Directory of original task JSON files.
        skills_dir: Directory containing Skill packages indexed by source ID.
        service_config: Local graph-management service settings.
        output: Fresh directory for the graph, rankings, and report.

    Returns:
        Per-task retrieval scores and their macro averages.
    """
    tasks = load_retrieval_tasks(tasks_dir)
    rankings = await _retrieve(
        {key: task.prompt for key, task in tasks.items()}, _load_skills(skills_dir), service_config, output
    )
    report = score_retrieval(tasks, rankings)
    write_json(output / "summary.json", report)
    return report


async def run_accuracy(
    config: SkillGraphAccuracyConfig,
    skills_dir: Path,
    service_config: Path,
    output: Path,
) -> SkillGraphAccuracyReport:
    """Inject ranked Skill packages, execute tasks, and score real deliverables.

    Args:
        config: Original benchmark, agent settings, and injection limit.
        skills_dir: Directory containing Skill packages indexed by source ID.
        service_config: Local graph-management service settings.
        output: Fresh directory for the report, selected IDs, and task rollouts.

    Returns:
        Official accuracy and scoring coverage, with execution errors kept separate.
    """
    env_config = AgentSkillOSConfig(benchmark_root=config.benchmark_root, agent=config.agent)
    config.agent.require_connection()
    samples = AgentSkillOSData(AgentSkillOSDataConfig(benchmark_root=env_config.benchmark_root)).load()
    queries = {sample.task.id: sample.task.instruction for sample in samples}
    if len(queries) != len(samples):
        raise ValueError("Duplicate AgentSkillOS task IDs")
    skills = _load_skills(skills_dir)
    rankings = await _retrieve(queries, skills, service_config, output, max(RETRIEVAL_TOP_K, config.execution_top_k))
    selected = {task_id: ranking[: config.execution_top_k] for task_id, ranking in rankings.items()}
    selected_ids = {skill_id for ranking in selected.values() for skill_id in ranking}
    selected_skills = {skill_id: skill for skill_id, skill in skills.items() if skill_id in selected_ids}

    # Score task outputs separately from the retrieval rankings.
    samples = [
        sample.model_copy(update={"reward_data": sample.reward_data.model_copy(update={"core_gt_ids": None})})
        for sample in samples
    ]
    write_json(output / "config.json", redact_credentials(config.model_dump(mode="json")))
    write_json(output / "selected_skills.json", selected)
    env = AgentSkillOSEnvironment(env_config)
    cases = await run_tasks(
        env, samples, _RankedSkills(selected_skills, selected), output / "tasks", config.concurrency
    )
    scored = [case for case in cases if case.error is None]
    report = SkillGraphAccuracyReport(
        model=env_config.agent.model,
        execution_top_k=config.execution_top_k,
        accuracy=sum(case.success is True for case in scored) / len(scored) if scored else None,
        scoring_coverage=len(scored) / len(samples),
        errors={case.task_id: case.error for case in cases if case.error is not None},
    )
    write_json(output / "summary.json", report)
    return report


def _load_skills(directory: Path) -> dict[str, Skill]:
    """Load local Skill directories, using each folder name as its source ID."""
    skill_id = TypeAdapter(SkillID)
    skills = {
        skill_id.validate_python(path.name): Skill.from_path(path)
        for path in sorted(directory.iterdir())
        if path.is_dir() and (path / "SKILL.md").is_file()
    }
    if not skills:
        raise ValueError(f"No Skill directories with SKILL.md found in {directory}")
    return skills


async def _retrieve(
    queries: dict[str, str],
    skills: dict[str, Skill],
    service_config: Path,
    output: Path,
    top_k: int = RETRIEVAL_TOP_K,
) -> dict[str, list[str]]:
    """Build a fixed snapshot and publish rankings only after all queries succeed."""
    source = read_skill_config(service_config).service
    if source.manage_pipeline is None:
        raise ValueError("The local Skill configuration must enable a management pipeline")
    resources = ServiceConfig(home=output / "state", agents=source.agents, manage_pipeline=source.manage_pipeline)
    output.mkdir(parents=True, exist_ok=False)
    registrations = {}
    rankings = {}
    async with open_skill_service(resources, PROJECT_ID) as service:
        for source_id, skill in skills.items():
            registered = await service.register_skill(
                RegisterRequest(
                    project_id=PROJECT_ID,
                    skill=StructuredSkillInfo(content=skill.content, resources=encode_resources(skill.resources)),
                )
            )
            registrations[source_id] = registered.skill_version_id
        source_by_version = {version_id: source_id for source_id, version_id in registrations.items()}
        if len(source_by_version) != len(skills):
            raise ValueError("Multiple Skill sources registered as the same version")
        write_json(output / "registrations.json", registrations)

        graph = await asyncio.wait_for(
            service.organize_skills(OrganizeRequest(project_id=PROJECT_ID, skill_refs=list(registrations.values()))),
            timeout=GRAPH_BUILD_TIMEOUT_S,
        )
        if len(graph.skill_version_ids) != len(skills) or set(graph.skill_version_ids) != set(source_by_version):
            raise ValueError("Published graph does not contain exactly the registered Skill versions")
        write_json(output / "graph.json", graph)

        for task_id, prompt in queries.items():
            try:
                result = await service.recommend_skills(
                    RecommendRequest(
                        project_id=PROJECT_ID,
                        graph_id=graph.graph_id,
                        graph_version_id=graph.graph_version_id,
                        task=prompt,
                        top_k=top_k,
                    )
                )
                version_ids = [score.skill_version_id for score in result.scores]
                if result.graph_version_id != graph.graph_version_id:
                    raise ValueError("Service returned a different graph snapshot")
                if len(version_ids) > top_k or len(set(version_ids)) != len(version_ids):
                    raise ValueError("Service returned duplicate Skill versions or exceeded the retrieval limit")
                if set(version_ids) - source_by_version.keys():
                    raise ValueError("Service returned Skill versions outside the graph")
                rankings[task_id] = [source_by_version[version_id] for version_id in version_ids]
            except Exception as exc:
                raise RuntimeError(f"Retrieval failed for {task_id}: {exc}") from exc

    write_json(output / "rankings.json", rankings)
    return rankings


class _RankedSkills(SkillSet):
    """Expose the saved per-task ranking to the shared rollout."""

    def __init__(self, skills: dict[str, Skill], rankings: dict[str, list[str]]) -> None:
        super().__init__(skills)
        self._rankings = rankings

    async def select(self, task: Task) -> list[str]:
        """Return the selected source IDs for the public task."""
        return list(self._rankings[task.id])

    def config_snapshot(self) -> dict[str, JsonValue]:
        """Bind each selected Skill order to the rollout inputs."""
        return {"rankings": self._rankings}


def load_retrieval_tasks(tasks_dir: Path) -> dict[str, AgentSkillOSRetrievalTask]:
    """Validate the original task files before starting retrieval.

    Args:
        tasks_dir: Directory containing the original AgentSkillOS task JSON files.

    Returns:
        Validated tasks indexed by their original IDs in file order.

    Raises:
        ValueError: No task files were found, or tasks or labels are invalid.
    """
    paths = sorted(tasks_dir.glob("*.json"))
    if not paths:
        raise ValueError(f"No AgentSkillOS task JSON files found in {tasks_dir}")

    tasks = {}
    for path in paths:
        try:
            task = AgentSkillOSRetrievalTask.model_validate_json(path.read_bytes())
            RelevanceRecord(core_gt_ids=task.skills)
        except ValueError as exc:
            raise ValueError(f"Invalid AgentSkillOS task file {path}: {exc}") from exc
        if task.task_id in tasks:
            raise ValueError(f"Duplicate AgentSkillOS task ID: {task.task_id}")
        tasks[task.task_id] = task
    return tasks


def score_retrieval(
    tasks: dict[str, AgentSkillOSRetrievalTask], rankings: dict[str, list[str]]
) -> RetrievalScoreReport:
    """Score the completed run using the original binary Skill labels.

    Args:
        tasks: Original task records keyed by task ID.
        rankings: Ordered source Skill IDs keyed by the same task IDs.

    Returns:
        Per-task metrics and their macro average.
    """
    if tasks.keys() != rankings.keys():
        raise ValueError("Ranking task IDs do not match the source tasks")
    rows = {task_id: compute_metrics(rankings[task_id], task.skills) for task_id, task in tasks.items()}
    return RetrievalScoreReport(metrics=aggregate_metrics(list(rows.values())), per_task=rows)
