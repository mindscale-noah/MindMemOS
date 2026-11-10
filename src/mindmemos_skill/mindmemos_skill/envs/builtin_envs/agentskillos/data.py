"""Adapt original AgentSkillOS tasks and assets to the shared environment API."""

import json

from pydantic import TypeAdapter

from ...data import Data, DataSample
from .config import AgentSkillOSDataConfig
from .source import relative_path
from .typing import AgentSkillOSRewardData, AgentSkillOSTask, Ranking, RelevanceRecord


@Data.register("agentskillos")
class AgentSkillOSData(Data[AgentSkillOSDataConfig]):
    """Load public prompts separately from private evaluator rules and Skill labels."""

    config_type = AgentSkillOSDataConfig
    """Original task definitions and optional retrieval ground truth."""

    def _load(self) -> list[DataSample]:
        """Read original JSON definitions and locate public task material."""
        root = self.config.benchmark_root.expanduser().resolve() / "benchmark" / "AgentSkillOS_bench"
        paths = sorted((root / "tasks").glob("*.json"))
        if not paths:
            raise ValueError("No official AgentSkillOS tasks found")
        labels = None
        if self.config.relevance_path is not None:
            raw_labels = self.config.relevance_path.expanduser().read_bytes()
            labels = TypeAdapter(dict[str, RelevanceRecord]).validate_json(raw_labels)
        samples = []
        for path in paths:
            raw = path.read_bytes()
            record = json.loads(raw)
            task_id = relative_path(record["task_id"])
            if "/" in task_id:
                raise ValueError("Task IDs must be single path components")
            files = {}
            # This pinned official release stores task4's project under task5.
            asset_id = "data_computation_task5" if task_id == "data_computation_task4" else task_id
            assets = root / "task_data" / asset_id
            if task_id == "data_computation_task5":
                assets = root / "task_data" / "unused-prompt-only-task"
            for source in sorted(assets.rglob("*")) if assets.exists() else []:
                if source.is_symlink():
                    raise ValueError("Public task inputs must not contain symlinks")
                if source.is_file():
                    name = source.relative_to(assets).as_posix()
                    if task_id == "data_computation_task1":
                        name = f"artifacts/{name}"
                    files[relative_path(name)] = source
            if (
                task_id
                in {
                    "data_computation_task1",
                    "data_computation_task4",
                    "document_creation_task2",
                    "document_creation_task3",
                    "web_interaction_task3",
                }
                and not files
            ):
                raise ValueError(f"Missing bundled public inputs for {task_id}")
            if labels is not None:
                if task_id not in labels:
                    raise ValueError(f"Missing retrieval labels for {task_id}")
                core_ids = labels[task_id].core_gt_ids
            else:
                core_ids = Ranking(skill_ids=record.get("skills", [])).skill_ids or None
            if not record.get("evaluators"):
                raise ValueError(f"Task {task_id} has no official evaluators")
            samples.append(
                DataSample(
                    task=AgentSkillOSTask(
                        id=task_id,
                        instruction=record["prompt"],
                        files=files,
                        outputs=[relative_path(name) for name in record["outputs"]],
                    ),
                    reward_data=AgentSkillOSRewardData(
                        task_config=record,
                        core_gt_ids=core_ids,
                        relevance_source="legacy_core_gt_ids" if labels is not None else "official_task_skills",
                    ),
                )
            )
        if labels is not None and set(labels) != {sample.task.id for sample in samples}:
            raise ValueError("Retrieval labels must exactly match the official task IDs")
        return samples
