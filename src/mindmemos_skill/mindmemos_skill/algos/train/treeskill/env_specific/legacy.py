"""Preserve the task-time message protocol of pre-adapter TreeSkill snapshots."""

import json

from ..env_adapter import TreeSkillEnvAdapter
from .spreadsheetbench_prompts import ROUTING_SYSTEM_PROMPT


@TreeSkillEnvAdapter.register("legacy")
class TreeSkillLegacyEnvAdapter(TreeSkillEnvAdapter):
    """Restore version-one legacy routing without altering stored Skill content."""

    def build_routing_messages(self, context) -> list[dict[str, str]]:
        """Reconstruct the original version-one messages.

        Args:
            context: Public task and legacy heading tree.

        Returns:
            Historical system and JSON user messages.
        """
        return [
            {
                "role": "system",
                "content": self.config.routing_system
                if self.config.routing_system is not None
                else ROUTING_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": "Route skill subtrees for this task.\n\n"
                + json.dumps(
                    {"task": context.task.model_dump(mode="json"), "skill_tree": context.tree},
                    ensure_ascii=False,
                    indent=2,
                ),
            },
        ]
