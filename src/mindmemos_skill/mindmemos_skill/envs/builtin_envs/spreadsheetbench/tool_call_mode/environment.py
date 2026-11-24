"""SpreadsheetBench execution through native structured tool calls."""

from pathlib import Path

from pydantic import JsonValue

from .....agents import OpenAIChatAgent
from .....agents.buitin_agents.openai_chat.callbacks import native_observation, parse_native_tools
from .....agents.typing import AgentResult
from .....skill import Skill
from ....base import Environment
from ...common import export_skills
from ..base import SpreadsheetExecutionEnvironment
from ..config import SpreadsheetBenchToolCallingConfig
from ..typing import SpreadsheetBenchTask, SpreadsheetContext, SpreadsheetObservation
from . import workspace_tools


@Environment.register("spreadsheetbench_tool_calling")
class SpreadsheetBenchToolCallingEnvironment(SpreadsheetExecutionEnvironment[SpreadsheetBenchToolCallingConfig]):
    """Expose workspace tools through the model API's native tool-call protocol."""

    config_type = SpreadsheetBenchToolCallingConfig
    """Native tool schemas, execution limits, and Skill-loading settings."""

    def __init__(self, config: SpreadsheetBenchToolCallingConfig, *, agent: OpenAIChatAgent | None = None) -> None:
        """Create the native tool parser around an optional model transport.

        Args:
            config: Validated native-tool execution settings.
            agent: Optional model transport for offline tests or caller injection.
        """
        super().__init__(config, protocol="tool_calling")
        self.agent = OpenAIChatAgent(
            config.agent,
            model=agent,
            parse_response=parse_native_tools,
            format_error=str,
            observation=native_observation,
            native_tools=True,
        )

    def _prepare_context(self, task: SpreadsheetBenchTask, directory: Path, skills: list[Skill]) -> SpreadsheetContext:
        """Export instructions into the workspace Skill directory."""
        skill_directory = directory / "skills/xlsx"
        text = export_skills(skills, skill_directory)
        return SpreadsheetContext(
            task=task,
            directory=directory,
            skill_directory=skill_directory,
            skill_text=text,
            tools=self.config.tools
            if self.config.tools is not None
            else workspace_tools.available_tools(self.config, text),
        )

    def _system_prompt(self, context: SpreadsheetContext) -> str:
        """Build instructions for workspace tool use."""
        return workspace_tools.system_prompt(context, self.config)

    def _user_prompt(self, context: SpreadsheetContext) -> str:
        """Render public workbook instructions."""
        return workspace_tools.user_prompt(context, self.execution)

    async def _get_observation(self, response: AgentResult, context: SpreadsheetContext) -> SpreadsheetObservation:
        """Execute native tool calls and return tool-role messages."""
        return await workspace_tools.get_observation(response, context, self.execution, self.agent)

    def _initial_skill_loaded(self, context: SpreadsheetContext) -> bool:
        """Detect preloaded instructions."""
        return self.config.skill_loading_mode == "system_prompt" and bool(context.skill_text.strip())

    def _diagnostics(
        self, context: SpreadsheetContext, loaded: bool, finished: bool, failure: str
    ) -> dict[str, JsonValue]:
        """Record workspace Skill loading behavior."""
        return {
            "skill_loaded": loaded,
            "skill_loading_mode": self.config.skill_loading_mode,
            "skill_tool_exposed": any(tool["function"]["name"] == "skill" for tool in context.tools),
        }
