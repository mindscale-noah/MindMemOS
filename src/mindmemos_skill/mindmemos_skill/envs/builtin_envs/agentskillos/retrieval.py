"""Load the prepared AgentSkillOS retrieval dataset without label leakage."""

import json

from pydantic import TypeAdapter

from ...data import Data, DataSample
from ...typing import Task
from .config import AgentSkillOSRetrievalDataConfig
from .typing import RelevanceRecord, RetrievalRewardData, TaskRecord


class AgentSkillOSRetrievalData(Data[AgentSkillOSRetrievalDataConfig]):
    """Preserve source ordering and require complete task/label correspondence."""

    config_type = AgentSkillOSRetrievalDataConfig
    """Explicit prepared source file locations."""

    def _load(self) -> list[DataSample]:
        """Read both files once and separate private relevance labels."""
        tasks_bytes = self.config.tasks_path.expanduser().read_bytes()
        relevance_bytes = self.config.relevance_path.expanduser().read_bytes()
        tasks = [TaskRecord.model_validate_json(line) for line in tasks_bytes.splitlines() if line.strip()]
        relevance = TypeAdapter(dict[str, RelevanceRecord]).validate_python(json.loads(relevance_bytes))
        if not tasks or {task.task_id for task in tasks} != set(relevance):
            raise ValueError("Tasks must be nonempty and exactly match relevance task IDs")
        return [
            DataSample(
                task=Task(id=task.task_id, instruction=task.instruction_text),
                reward_data=RetrievalRewardData(
                    core_gt_ids=relevance[task.task_id].core_gt_ids,
                ),
            )
            for task in tasks
        ]
