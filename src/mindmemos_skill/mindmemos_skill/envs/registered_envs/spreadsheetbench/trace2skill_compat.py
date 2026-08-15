"""Trace2Skill-compatible SpreadsheetBench policy prompting and action parsing."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from enum import StrEnum
from importlib.resources import files
from pathlib import Path
from typing import Any

from ....typing import Task
from .recalculation import append_recalculation_instructions

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


class PolicyResponseType(StrEnum):
    ACTION = "action"
    TASK_COMPLETE = "task_complete"
    FORMAT_ERROR = "format_error"


@dataclass(frozen=True, slots=True)
class PolicyAction:
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class PolicyResponse:
    response_type: PolicyResponseType
    action: PolicyAction | None = None
    error_message: str | None = None


def build_reference_messages(
    *,
    task: Task,
    working_dir: Path,
    input_file: Path,
    output_file: Path,
    spreadsheet_content: str,
    skill_content: str,
    skill_dir: Path | None,
    transactional_recalculation: bool,
) -> list[dict[str, str]]:
    """Render the released preloaded-Skill policy prompt and task contract."""

    prompt_name = (
        "trace2skill_preloaded_system_prompt.txt" if skill_content.strip() else "trace2skill_no_skill_system_prompt.txt"
    )
    template = files(f"{__package__}.prompt_templates").joinpath(prompt_name).read_text(encoding="utf-8")
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
{task.metadata.get("instruction_type") or ""}

### answer_position
{task.metadata.get("answer_position") or ""}

### output_path
{output_file.resolve()}

---
**REMINDER**: Write files ONLY in `{working_dir.resolve()}`. Save output to exact path: `{output_file.resolve()}`
---

Solve the question and save the modified spreadsheet to the exact output_path shown above."""
    if transactional_recalculation:
        user = append_recalculation_instructions(
            user,
            working_dir=working_dir,
            output_file=output_file,
        )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_policy_response(response: str) -> PolicyResponse:
    """Parse the released text ReAct action format without counting braces in strings."""

    if TASK_COMPLETE_SIGNAL in response:
        return PolicyResponse(PolicyResponseType.TASK_COMPLETE)
    marker = "Action:"
    if marker not in response:
        return PolicyResponse(PolicyResponseType.TASK_COMPLETE)

    action_text = response[response.find(marker) + len(marker) :].strip()
    json_text = _extract_json_object(action_text)
    action = _load_action(json_text) if json_text is not None else None
    if action is None:
        return PolicyResponse(
            PolicyResponseType.FORMAT_ERROR,
            error_message=FORMAT_ERROR_MESSAGE,
        )
    return PolicyResponse(PolicyResponseType.ACTION, action=action)


def run_reference_bash(command: str, *, working_dir: Path, timeout_seconds: int) -> str:
    """Execute bash with the released policy tool's observable result contract."""

    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=working_dir,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
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
    """Wrap reference-agent feedback in the released ReAct observation format."""

    return f"Observation: {observation}"


def _strip_frontmatter(content: str) -> str:
    if not content.startswith("---"):
        return content
    match = re.match(r"^---\s*\n.*?\n---\s*\n?", content, flags=re.DOTALL)
    return content[match.end() :].lstrip("\n") if match is not None else content


def _extract_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    quote = ""
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                in_string = False
            continue
        if char in {'"', "'"}:
            in_string = True
            quote = char
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
            if depth < 0:
                return None
    if depth > 0 and not in_string:
        return text[start:].strip() + ("}" * depth)
    return None


def _load_action(text: str) -> PolicyAction | None:
    for candidate in (text, text.replace("'", '"')):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict) or not isinstance(payload.get("name"), str):
            continue
        arguments = payload.get("arguments", {})
        if not isinstance(arguments, dict):
            continue
        return PolicyAction(name=payload["name"], arguments=arguments)
    return None


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
