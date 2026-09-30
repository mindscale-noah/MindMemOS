"""Interact through single bash actions with output and stagnation checks."""

import json
import re
import subprocess
from pathlib import Path

from ....agents.typing import ToolTurn
from ....infra.execution import ExecutionConfig, execute
from ....infra.execution.constants import INPUT_FILE_ENV, OUTPUT_FILE_ENV
from .prompts.bash_only import NO_SKILL, WITH_SKILL
from .typing import SpreadsheetObservation


def run_bash_action(action: dict, workspace: Path, config: ExecutionConfig) -> str:
    """Execute a bash action and preserve released observation formatting.

    Args:
        action: Parsed tool name and arguments.
        workspace: Public input/output directory.
        config: Local runtime settings.

    Returns:
        Tool observation with the original head/tail truncation format.
    """
    if action["name"] != "bash":
        return f"Error: Unknown tool '{action['name']}'"
    command = action["arguments"].get("command")
    if not isinstance(command, str) or not command.strip():
        return "Error: bash arguments.command must be a non-empty string"
    config = config.model_copy(
        update={
            "environment": {
                **config.environment,
                INPUT_FILE_ENV: str(workspace / "input.xlsx"),
                OUTPUT_FILE_ENV: str(workspace / "output.xlsx"),
            }
        }
    )
    try:
        result = execute([*config.shell_command, command], workspace, config)
    except subprocess.TimeoutExpired:
        output = f"[ERROR] Command timed out after {config.timeout_s:g} seconds"
    except Exception as exc:
        output = f"[ERROR] Failed to execute command: {exc}"
    else:
        output = result.stdout or ""
        if result.stderr:
            output += f"\n[STDERR]\n{result.stderr}" if output else result.stderr
        if result.returncode:
            output += f"\n[Exit code: {result.returncode}]"
        output = output.strip() or "[Command completed with no output]"
    if len(output) > 6000:
        warning = (
            "\n\n[WARNING: Output truncated. Showing first 3000 and last 3000 characters. "
            f"{len(output) - 6000} characters elided.]\n\n"
        )
        output = output[:3000] + warning + output[-3000:]
    return output


def _preview(path: Path) -> str:
    """Render the released first-sheet preview with string-valued tuples."""
    import openpyxl

    workbook = openpyxl.load_workbook(path, data_only=True)
    try:
        sheet = workbook.active
        lines = []
        for index, row in enumerate(sheet.iter_rows(values_only=True), 1):
            if index > 5:
                lines.append(f"... ({sheet.max_row - 5} more rows)")
                break
            lines.append(str(tuple(str(value) if value is not None else "" for value in row)))
        return "\n".join(lines)
    finally:
        workbook.close()


def system_prompt(context, config):
    """Render system instructions for the configured protocol.

    Args:
        context: Public task, prepared paths, Skill text, and tools for one case.
        config: Environment execution and protocol settings.

    Returns:
        Initial system message.
    """
    text = re.sub(r"^---\s*\n.*?\n---\s*\n?", "", context.skill_text, count=1, flags=re.DOTALL).lstrip("\n")
    return (
        (WITH_SKILL if text.strip() else NO_SKILL)
        .replace("{skill_content}", text)
        .replace("{skill_dir}", str(context.skill_directory.resolve()))
    )


def user_prompt(context, config):
    """Render public task instructions.

    Args:
        context: Public task, prepared paths, Skill text, and tools for one case.
        config: Environment execution and protocol settings.

    Returns:
        Initial user message.
    """
    task, directory = context.task, context.directory
    visible = directory.resolve()
    text = context.skill_text
    reminder = (
        f"**REMINDER**: Write files ONLY in `{visible}`. Save output to exact path: `{visible / 'output.xlsx'}`"
        if text.strip()
        else f"**REMINDER**: You can ONLY access files within `{visible}`. Save output to the exact path: "
        f"`{visible / 'output.xlsx'}`"
    )
    user = (
        "Task: Below is the spreadsheet manipulation question you need to solve:\n\n"
        f"### working_directory\n{visible}\n\n### instruction\n{task.instruction}\n\n"
        f"### spreadsheet_path\n{visible / 'input.xlsx'}\n\n"
        f"### spreadsheet_content\n{_preview(directory / 'input.xlsx')}\n\n"
        f"### instruction_type\n{task.instruction_type}\n\n### answer_position\n{task.answer_position}\n\n"
        f"### output_path\n{visible / 'output.xlsx'}\n\n---\n"
        f"{reminder}\n---\n\n"
        "Solve the question and save the modified spreadsheet to the exact output_path shown above."
    )
    if config.execution.shell_command[0].lower().endswith(("cmd.exe", "powershell.exe", "pwsh.exe", "pwsh")):
        user += "\n\nShell command prefix: " + json.dumps(config.execution.shell_command)
    return user


def get_observation(response, context, config, agent):
    """Execute parsed tool calls and encode feedback for the shared loop.

    Args:
        response: Agent result containing a parsed tool turn.
        context: Public workspace and tool definitions for this case.
        config: Execution limits and protocol settings.
        agent: Agent providing the observation encoding callback.

    Returns:
        Assistant history, tool feedback, and completion diagnostics.
    """
    turn = response.output
    if not isinstance(turn, ToolTurn):
        raise TypeError("Bash-only mode requires a text policy output")
    text = turn.text
    history = text.rsplit("</think>", 1)[-1].lstrip("\n") if "</think>" in text else text
    assistant = {"role": "assistant", "content": history}
    if turn.error:
        observation = turn.error
        signature = json.dumps(["format_error", text, observation])
        kind = "format error"
    elif not turn.calls:
        return SpreadsheetObservation(
            assistant=assistant, complete=True, output=text.split("ACTION: TASK_COMPLETE", 1)[0].strip()
        )
    else:
        call = turn.calls[0]
        action = {"name": call.name, "arguments": call.arguments}
        observation = call.error or run_bash_action(action, context.directory, config.execution)
        signature = json.dumps([action, observation], sort_keys=True)
        kind = "action/result pair"
    return SpreadsheetObservation(
        assistant=assistant,
        messages=[{"role": "user", "content": f"Observation: {observation}"}],
        signature=signature,
        repeated_kind=kind,
        output=text,
    )
