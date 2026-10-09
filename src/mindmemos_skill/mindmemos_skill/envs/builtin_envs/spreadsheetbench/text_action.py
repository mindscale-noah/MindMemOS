"""SpreadsheetBench execution through text Action/Observation messages."""

import json
from pathlib import Path

from pydantic import JsonValue

from ....agents import OpenAIChatAgent
from ....agents.typing import AgentResult
from ....skill import Skill
from ...base import Environment
from ..common import export_skills
from .base import SpreadsheetExecutionEnvironment
from .config import SpreadsheetBenchTextActionConfig
from .text_action_policy import (
    PolicyResponseType,
    build_reference_messages,
    format_reference_observation,
    parse_policy_response,
    run_reference_bash,
)
from .typing import SpreadsheetBenchTask, SpreadsheetContext, SpreadsheetObservation


def spreadsheet_preview(file_path: Path, max_rows: int = 5) -> str:
    """Return the workbook preview consumed by the TreeSkill routing prompt.

    Args:
        file_path: Public input workbook used to build the first-sheet preview.
        max_rows: Maximum number of preview rows before a truncation notice.
    """

    try:
        import openpyxl

        workbook = openpyxl.load_workbook(file_path, data_only=True)
        worksheet = workbook.active
        lines: list[str] = []
        for index, row in enumerate(worksheet.iter_rows(values_only=True), 1):
            if index > max_rows:
                lines.append(f"... ({worksheet.max_row - max_rows} more rows)")
                break
            values = [str(cell) if cell is not None else "" for cell in row]
            lines.append(str(tuple(values)))
        workbook.close()
        return "\n".join(lines)
    except Exception as exc:
        return f"[Could not read spreadsheet: {exc}]"


@Environment.register("spreadsheetbench_text_action")
class SpreadsheetBenchTextActionEnvironment(SpreadsheetExecutionEnvironment[SpreadsheetBenchTextActionConfig]):
    """Execute bash actions parsed from model text using the shared workbook lifecycle."""

    config_type = SpreadsheetBenchTextActionConfig
    """Text-action execution limits and recalculation settings."""

    def __init__(self, config: SpreadsheetBenchTextActionConfig, *, agent: OpenAIChatAgent | None = None) -> None:
        """Construct a raw-text model for the TreeSkill response hook.

        Args:
            config: TreeSkill prompt, execution, and interaction settings.
            agent: Optional raw-text agent for injection and offline testing.
        """
        super().__init__(config)
        self.agent = agent if agent is not None else OpenAIChatAgent(config.agent)

    def _system_prompt(self, context: SpreadsheetContext) -> str:
        """Render the pinned system message from prepared Skill content."""
        return build_reference_messages(
            task=context.task,
            working_dir=context.directory,
            input_file=context.directory / "input.xlsx",
            output_file=context.directory / "output.xlsx",
            spreadsheet_content="",
            skill_content=context.skill_text,
            skill_dir=context.skill_directory,
            transactional_recalculation=False,
        )[0]["content"]

    def _user_prompt(self, context: SpreadsheetContext) -> str:
        """Render the pinned task prompt and optional recalculation instructions."""
        return build_reference_messages(
            task=context.task,
            working_dir=context.directory,
            input_file=context.directory / "input.xlsx",
            output_file=context.directory / "output.xlsx",
            spreadsheet_content=spreadsheet_preview(context.directory / "input.xlsx"),
            skill_content=context.skill_text,
            skill_dir=context.skill_directory,
            transactional_recalculation=self.config.transactional_recalculation,
        )[1]["content"]

    def _get_observation(self, response: AgentResult, context: SpreadsheetContext) -> SpreadsheetObservation:
        """Parse TreeSkill actions, execute bash, and encode its text feedback."""
        if not isinstance(response.output, str):
            raise TypeError("Text-action protocol requires text output")
        content = response.output
        assistant = {"role": "assistant", "content": content}
        parsed = parse_policy_response(content)
        if parsed.response_type is PolicyResponseType.TASK_COMPLETE:
            return SpreadsheetObservation(assistant=assistant, complete=True, output=content)
        if parsed.response_type is PolicyResponseType.FORMAT_ERROR:
            feedback = format_reference_observation(parsed.error_message or "Invalid action format.")
            signature = json.dumps(
                {"response_type": "format_error", "response": content, "feedback": feedback},
                sort_keys=True,
                ensure_ascii=False,
            )
            kind = "format error"
        else:
            action = parsed.action
            if action.name != "bash":
                observation = f"Error: unknown tool {action.name!r}; only 'bash' is available"
            else:
                command = action.arguments.get("command")
                observation = (
                    run_reference_bash(
                        command, working_dir=context.directory, timeout_seconds=self.config.shell_timeout_seconds
                    )
                    if isinstance(command, str) and command.strip()
                    else "Error: bash arguments.command must be a non-empty string"
                )
            feedback = format_reference_observation(observation)
            signature = json.dumps(
                {
                    "response_type": "action",
                    "name": action.name,
                    "arguments": action.arguments,
                    "observation": observation,
                },
                sort_keys=True,
                ensure_ascii=False,
                default=str,
            )
            kind = "action/result pair"
        return SpreadsheetObservation(
            assistant=assistant,
            messages=[{"role": "user", "content": feedback}],
            signature=signature,
            repeated_kind=kind,
            output=content,
        )

    def _prepare_context(self, task: SpreadsheetBenchTask, directory: Path, skills: list[Skill]) -> SpreadsheetContext:
        """Preload Skill text and resources without exposing native tool schemas."""
        text = export_skills(skills, directory)
        return SpreadsheetContext(task=task, directory=directory, skill_directory=directory, skill_text=text, tools=[])

    def _validate_task(self, task: SpreadsheetBenchTask) -> None:
        """Require exactly one workbook for the text-action task prompt."""
        if len(task.cases) != 1:
            raise ValueError("spreadsheetbench_text_action requires exactly one workbook case")

    def _initial_skill_loaded(self, context: SpreadsheetContext) -> bool:
        """Treat nonempty instructions as preloaded into the system prompt."""
        return bool(context.skill_text.strip())

    def _completion_error(self, context: SpreadsheetContext) -> str | None:
        """Require output.xlsx before accepting the task-complete signal."""
        if not (context.directory / "output.xlsx").is_file():
            return f"Output file was not created: {context.directory / 'output.xlsx'}"
        return None

    def _completion_reminder(self, context: SpreadsheetContext, error: str) -> str:
        """Request the missing output before accepting the text completion signal."""
        return (
            f"[System Check] The output file was NOT created at: {context.directory / 'output.xlsx'}\n"
            "Please return to work again until you create the output file at the exact path "
            "specified above, then signal ACTION: TASK_COMPLETE again."
        )

    def _turn_limit_output(self, output: str) -> str:
        """Report the exhausted text-action turn budget."""
        return "Max turns exceeded. Could not complete the task."

    def _diagnostics(
        self, context: SpreadsheetContext, loaded: bool, finished: bool, failure: str
    ) -> dict[str, JsonValue]:
        """Record text-action completion and early-stop reasons."""
        return {"finished": finished, "stop_reason": failure}
