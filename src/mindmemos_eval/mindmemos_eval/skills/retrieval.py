"""Read original AgentSkillOS tasks and score the resulting rankings."""

from pathlib import Path

from mindmemos_skill.envs.builtin_envs.agentskillos.reward import aggregate_metrics, compute_metrics
from mindmemos_skill.envs.builtin_envs.agentskillos.typing import RelevanceRecord

from .typing import AgentSkillOSRetrievalTask, RetrievalScoreReport


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
