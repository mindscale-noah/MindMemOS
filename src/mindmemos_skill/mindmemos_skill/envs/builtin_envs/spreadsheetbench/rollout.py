"""Run the workspace ReAct protocol and retain exact evaluation feedback."""

from pathlib import Path

from ....agents import OpenAIChatAgent
from ....agents.buitin_agents.openai_chat.callbacks import (
    native_observation,
    parse_native_tools,
    parse_text_action,
    text_format_error,
    text_observation,
)
from ....skill import Skill
from ...base import Environment
from ...data import RewardData
from ...typing import RolloutResult, Task
from ..common import export_skills
from .bash_only import run_bash_only
from .config import SpreadsheetBenchConfig
from .reward import evaluate
from .typing import SpreadsheetBenchRewardData, SpreadsheetBenchTask
from .workbooks import prepare_cases
from .workspace_tools import run_workspace_tools


@Environment.register("spreadsheetbench")
class SpreadsheetBenchEnvironment(Environment[SpreadsheetBenchConfig]):
    """Edit workbooks through environment tools, then score private references."""

    config_type = SpreadsheetBenchConfig
    """Model, tool protocol, turn budget, and local execution settings."""

    def __init__(self, config: SpreadsheetBenchConfig, *, agent: OpenAIChatAgent | None = None) -> None:
        """Construct the configured tool protocol around a model transport.

        Args:
            config: Recorded rollout settings.
            agent: Optional matching model transport for dependency injection.
        """
        super().__init__(config)
        textual = config.tool_agent == "text_action"
        self.agent = OpenAIChatAgent(
            config.agent,
            model=agent,
            parse_response=parse_text_action if textual else parse_native_tools,
            format_error=text_format_error if textual else str,
            observation=text_observation if textual else native_observation,
            native_tools=not textual,
        )

    def _run(self, *, task: Task, reward_data: RewardData, skills: list[Skill], workspace: Path) -> RolloutResult:
        """Dispatch public interaction, then evaluate every output identically."""
        if not isinstance(task, SpreadsheetBenchTask) or not isinstance(reward_data, SpreadsheetBenchRewardData):
            raise TypeError("SpreadsheetBench requires SpreadsheetBenchTask and SpreadsheetBenchRewardData")
        if self.config.interaction_mode == "bash_only" and len(task.cases) != 1:
            raise ValueError("bash_only requires exactly one workbook case")
        directories = prepare_cases(task, reward_data, workspace)
        text = ""
        for directory in directories:
            skill_directory = (
                directory / "skills/xlsx" if self.config.interaction_mode == "workspace_tools" else directory
            )
            text = export_skills(skills, skill_directory)
        if self.config.interaction_mode == "workspace_tools":
            interaction = run_workspace_tools(task, text, directories, workspace, self.config, self.agent)
        else:
            interaction = run_bash_only(task, text, directories[0], workspace, self.config, self.agent)
        reward = evaluate(reward_data, directories).model_copy(update=interaction.diagnostics)
        trajectory = [*interaction.trajectory, {"role": "system", "content": reward.detail}]
        return RolloutResult(output=interaction.output, trajectory=trajectory, usage=interaction.usage, reward=reward)
