"""Trace2Skill-compatible SpreadsheetBench policy prompting and action parsing."""

from __future__ import annotations

import re
import subprocess
from importlib.resources import files
from pathlib import Path

from .....common.json import extract_json_object
from .....infra.execution import ExecutionConfig, execute
from ..typing import SpreadsheetBenchTask as Task
from .typing import PolicyAction, PolicyResponse, PolicyResponseType

TASK_COMPLETE_SIGNAL = "ACTION: TASK_COMPLETE"
_MAX_OBSERVATION_LENGTH = 6000
_OBSERVATION_EDGE_LENGTH = 3000

FORMAT_ERROR_MESSAGE = """Failed to parse your action. Please use the correct format.

To execute a tool, use this EXACT format:

Action:
{
    "name": "<tool_name>",
    "arguments": {"command": "<command_here>"}
}

To complete the task, output exactly:

ACTION: TASK_COMPLETE

Please try again with the correct format."""


def build_reference_messages(
    *,
    task: Task,
    working_dir: Path,
    input_file: Path,
    output_file: Path,
    spreadsheet_content: str,
    skill_content: str,
    skill_dir: Path | None,
) -> list[dict[str, str]]:
    """Render the released preloaded-Skill policy prompt and task contract.

    Args:
        task: Stage audit label or public task supplied by the caller.
        working_dir: Agent-visible task workspace where tools execute.
        input_file: Public input workbook path.
        output_file: Exact workbook output path required by the task.
        spreadsheet_content: Public first-sheet preview inserted into the prompt.
        skill_content: Prepared full or routed Skill Markdown.
        skill_dir: Exported Skill resource directory, or None without a Skill.
    """

    prompt_name = (
        "trace2skill_preloaded_system_prompt.txt" if skill_content.strip() else "trace2skill_no_skill_system_prompt.txt"
    )
    template = files(f"{__package__}.prompts").joinpath(prompt_name).read_text(encoding="utf-8")
    system = template.replace("{skill_content}", _strip_frontmatter(skill_content))
    system = system.replace("{skill_dir}", str(skill_dir.resolve()) if skill_dir is not None else "")
    user = f"""Task: Below is the spreadsheet manipulation question you need to solve:

### working_directory
{working_dir.resolve()}

### instruction
{task.instruction}

### spreadsheet_path
{input_file.resolve()}

### spreadsheet_content
{spreadsheet_content}

### instruction_type
{task.instruction_type or ""}

### answer_position
{task.answer_position or ""}

### output_path
{output_file.resolve()}

---
**REMINDER**: Write files ONLY in `{working_dir.resolve()}`. Save output to exact path: `{output_file.resolve()}`
---

Solve the question and save the modified spreadsheet to the exact output_path shown above."""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_policy_response(response: str) -> PolicyResponse:
    """Parse the released text ReAct action format without counting braces in strings.

    Args:
        response: Raw textual policy response.
    """

    if TASK_COMPLETE_SIGNAL in response:
        return PolicyResponse(PolicyResponseType.TASK_COMPLETE)
    marker = "Action:"
    if marker not in response:
        return PolicyResponse(PolicyResponseType.TASK_COMPLETE)

    action_text = response[response.find(marker) + len(marker) :].strip()
    action = _load_action(action_text)
    if action is None:
        return PolicyResponse(
            PolicyResponseType.FORMAT_ERROR,
            error_message=FORMAT_ERROR_MESSAGE,
        )
    return PolicyResponse(PolicyResponseType.ACTION, action=action)


async def run_reference_bash(
    command: str, *, working_dir: Path, timeout_seconds: int, execution: ExecutionConfig | None = None
) -> str:
    """Execute bash with the released policy tool's observable result contract.

    Args:
        command: Model-proposed shell command executed in the task workspace.
        working_dir: Agent-visible task workspace where tools execute.
        timeout_seconds: Explicit timeout seconds for this operation.
        execution: Preinstalled Python/uv launcher; None uses the current interpreter.
    """

    try:
        config = (execution or ExecutionConfig()).model_copy(update={"timeout_s": timeout_seconds})
        result = await execute([*config.shell_command, command], working_dir, config)
        output = result.stdout or ""
        if result.stderr:
            output += f"\n[STDERR]\n{result.stderr}" if output else result.stderr
        if result.returncode != 0:
            output += f"\n[Exit code: {result.returncode}]"
        output = output.strip() if output.strip() else "[Command completed with no output]"
    except subprocess.TimeoutExpired:
        output = f"[ERROR] Command timed out after {timeout_seconds} seconds"
    except Exception as exc:
        output = f"[ERROR] Failed to execute command: {exc}"
    return _truncate_observation(output)


def format_reference_observation(observation: str) -> str:
    """Wrap reference-agent feedback in the released ReAct observation format.

    Args:
        observation: Tool result or format-error feedback to encode.
    """

    return f"Observation: {observation}"


def _strip_frontmatter(content: str) -> str:
    if not content.startswith("---"):
        return content
    match = re.match(r"^---\s*\n.*?\n---\s*\n?", content, flags=re.DOTALL)
    return content[match.end() :].lstrip("\n") if match is not None else content


def _load_action(text: str) -> PolicyAction | None:
    """Decode the shared JSON format and enforce the policy action shape."""
    try:
        payload = extract_json_object(text, allow_incomplete=True)
    except ValueError:
        return None
    if not isinstance(payload.get("name"), str) or not isinstance(payload.get("arguments", {}), dict):
        return None
    return PolicyAction(name=payload["name"], arguments=payload.get("arguments", {}))


def _truncate_observation(text: str) -> str:
    if len(text) <= _MAX_OBSERVATION_LENGTH:
        return text
    elided = len(text) - (2 * _OBSERVATION_EDGE_LENGTH)
    warning = (
        f"\n\n[WARNING: Output truncated. Showing first {_OBSERVATION_EDGE_LENGTH} and "
        f"last {_OBSERVATION_EDGE_LENGTH} characters. {elided} characters elided.]\n\n"
    )
    return text[:_OBSERVATION_EDGE_LENGTH] + warning + text[-_OBSERVATION_EDGE_LENGTH:]


__all__ = [
    "FORMAT_ERROR_MESSAGE",
    "PolicyAction",
    "PolicyResponse",
    "PolicyResponseType",
    "TASK_COMPLETE_SIGNAL",
    "build_reference_messages",
    "format_reference_observation",
    "parse_policy_response",
    "run_reference_bash",
]
