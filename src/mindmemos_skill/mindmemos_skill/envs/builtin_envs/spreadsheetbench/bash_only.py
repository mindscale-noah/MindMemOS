"""Interact through single bash actions with output and stagnation checks."""

import json
import re
import subprocess
from pathlib import Path

from ....agents import OpenAIChatAgent
from ....agents.typing import ToolTurn
from ....infra.execution import ExecutionConfig, execute
from ....infra.execution.constants import INPUT_FILE_ENV, OUTPUT_FILE_ENV
from ..common import total_usage
from .config import SpreadsheetBenchConfig
from .prompts.bash_only import NO_SKILL, WITH_SKILL
from .typing import SpreadsheetBenchTask, SpreadsheetInteractionResult


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
        warning = f"\n\n[WARNING: Output truncated. Showing first 3000 and last 3000 characters. {len(output) - 6000} characters elided.]\n\n"
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


def run_bash_only(
    task: SpreadsheetBenchTask,
    text: str,
    directory: Path,
    workspace: Path,
    config: SpreadsheetBenchConfig,
    agent: OpenAIChatAgent,
) -> SpreadsheetInteractionResult:
    """Run a single-case textual bash loop with bounded recovery.

    Args:
        task: Public task instructions and answer-position hints.
        text: Rendered skill guidance injected into the system prompt.
        directory: Prepared public workbook directory.
        workspace: Directory for prompt snapshots.
        config: Interaction and execution settings.
        agent: Text-action protocol adapter.

    Returns:
        Public trajectory, usage, and completion diagnostics.
    """
    visible = directory.resolve()
    text = re.sub(r"^---\s*\n.*?\n---\s*\n?", "", text, count=1, flags=re.DOTALL).lstrip("\n")
    if text:
        (directory / "SKILL.md").unlink(missing_ok=True)
        (directory / "SKILL.md").write_text(text, encoding="utf-8")
    system = (
        (WITH_SKILL if text.strip() else NO_SKILL).replace("{skill_content}", text).replace("{skill_dir}", str(visible))
    )
    reminder = (
        f"**REMINDER**: Write files ONLY in `{visible}`. Save output to exact path: `{visible / 'output.xlsx'}`"
        if text.strip()
        else f"**REMINDER**: You can ONLY access files within `{visible}`. Save output to the exact path: "
        f"`{visible / 'output.xlsx'}`"
    )
    user = (
        "Task: Below is the spreadsheet manipulation question you need to solve:\n\n"
        f"### working_directory\n{visible}\n\n### instruction\n{task.instruction}\n\n"
        f"### spreadsheet_path\n{visible / 'input.xlsx'}\n\n### spreadsheet_content\n{_preview(directory / 'input.xlsx')}\n\n"
        f"### instruction_type\n{task.instruction_type}\n\n### answer_position\n{task.answer_position}\n\n"
        f"### output_path\n{visible / 'output.xlsx'}\n\n---\n"
        f"{reminder}\n---\n\n"
        "Solve the question and save the modified spreadsheet to the exact output_path shown above."
    )
    if config.execution.shell_command[0].lower().endswith(("cmd.exe", "powershell.exe", "pwsh.exe", "pwsh")):
        user += "\n\nShell command prefix: " + json.dumps(config.execution.shell_command)
    (workspace / "system.md").write_text(system, encoding="utf-8")
    (workspace / "task.md").write_text(user, encoding="utf-8")
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    trajectory, responses = list(messages), []
    previous, repeats, reminded, finished = "", 0, False, False
    output, failure = "", ""
    for _ in range(config.max_turns):
        response = agent.run(messages=messages, tools=[])
        if not isinstance(response.output, ToolTurn):
            raise TypeError("Bash-only mode requires a text policy output")
        responses.append(response)
        output = response.output.text
        history_text = output.rsplit("</think>", 1)[-1].lstrip("\n") if "</think>" in output else output
        event = {"role": "assistant", "content": history_text}
        messages.append(event)
        trajectory.append(response.trajectory[-1])
        turn = response.output
        if turn.error:
            observation = turn.error
            signature = json.dumps(["format_error", output, observation])
            repeated_kind = "format error"
        else:
            action = {"name": turn.calls[0].name, "arguments": turn.calls[0].arguments} if turn.calls else None
            if action is None:
                if (directory / "output.xlsx").is_file():
                    finished = True
                    output = output.split("ACTION: TASK_COMPLETE", 1)[0].strip()
                    break
                if reminded or len(responses) >= config.max_turns:
                    failure = f"Output file was not created: {visible / 'output.xlsx'}"
                    break
                reminded = True
                observation = (
                    f"[System Check] The output file was NOT created at: {visible / 'output.xlsx'}\n"
                    "Please return to work again until you create the output file at the exact path "
                    "specified above, then signal ACTION: TASK_COMPLETE again."
                )
                feedback = {"role": "user", "content": observation}
                messages.append(feedback)
                trajectory.append(feedback)
                continue
            observation = run_bash_action(action, directory, config.execution)
            signature = json.dumps([action, observation], sort_keys=True)
            repeated_kind = "action/result pair"
        feedback = {"role": "user", "content": f"Observation: {observation}"}
        messages.append(feedback)
        trajectory.append(feedback)
        repeats = repeats + 1 if signature == previous else 1
        previous = signature
        if config.stagnation_limit is not None and repeats >= config.stagnation_limit:
            failure = f"ReAct agent repeated an identical {repeated_kind} {config.stagnation_limit} consecutive times"
            break
    if not finished and not failure:
        failure = "Max turns exceeded"
        output = "Max turns exceeded. Could not complete the task."

    return SpreadsheetInteractionResult(
        output=output,
        trajectory=trajectory,
        usage=total_usage(responses),
        diagnostics={"finished": finished, "stop_reason": failure, "n_turns": len(responses)},
    )
