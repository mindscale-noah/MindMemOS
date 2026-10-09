"""Interact through workspace file, shell, and optional skill tools."""

import json

from pydantic import JsonValue

from ....agents import OpenAIChatAgent
from ....agents.typing import AgentResult, ToolTurn
from ....infra.execution import ExecutionConfig
from .config import SpreadsheetBenchToolCallingConfig
from .prompts.workspace_tools import SYSTEM_PROMPT
from .tools import TOOLS, run_tool
from .typing import SpreadsheetContext, SpreadsheetObservation


def available_tools(config: SpreadsheetBenchToolCallingConfig, text: str) -> list[dict[str, JsonValue]]:
    """Return tool declarations for this interaction.

    Args:
        config: Environment mode and Skill-loading settings.
        text: Already assembled Skill instructions.

    Returns:
        Tool schemas exposed to the current agent.
    """
    if config.skill_loading_mode == "system_prompt" and text.strip():
        return [tool for tool in TOOLS if tool["function"]["name"] != "skill"]
    return TOOLS


def system_prompt(context: SpreadsheetContext, config: SpreadsheetBenchToolCallingConfig) -> str:
    """Render system instructions for the configured protocol.

    Args:
        context: Public task, prepared paths, Skill text, and tools for one case.
        config: Environment execution and protocol settings.

    Returns:
        Initial system message.
    """
    system = SYSTEM_PROMPT
    if config.skill_loading_mode == "system_prompt" and context.skill_text.strip():
        system += f"\n\n## Skill\n{context.skill_text.strip()}"
    elif context.skill_text.strip():
        system += (
            "\n\nA `skill` tool is available with expert skills: xlsx. Call it first "
            '(e.g. skill(name="xlsx")) to load detailed guidance and the absolute '
            "path to reusable reference scripts before you start."
        )
    return system


def user_prompt(context: SpreadsheetContext, execution: ExecutionConfig) -> str:
    """Render public task instructions.

    Args:
        context: Public task, prepared paths, Skill text, and tools for one case.
        execution: Managed interpreter and shell settings.

    Returns:
        Initial user message.
    """
    task = context.task
    parts = ["The source file 'input.xlsx' is in your working directory.", f"Task:\n{task.instruction}"]
    if task.answer_position.strip():
        parts.append(
            "Answer position (cells to verify) — these are the cells that will be "
            "graded, so make sure their final computed values are correct:\n" + task.answer_position.strip()
        )
    parts.append("Complete the task and save the result as 'output.xlsx' (do not modify 'input.xlsx').")
    user = "\n\n".join(parts)
    if execution.shell_command[0].lower().endswith(("cmd.exe", "powershell.exe", "pwsh.exe", "pwsh")):
        user += "\n\nShell command prefix: " + json.dumps(execution.shell_command)
    return user


def get_observation(
    response: AgentResult,
    context: SpreadsheetContext,
    execution: ExecutionConfig,
    agent: OpenAIChatAgent,
) -> SpreadsheetObservation:
    """Execute parsed tool calls and encode feedback for the shared loop.

    Args:
        response: Agent result containing a parsed tool turn.
        context: Public workspace and tool definitions for this case.
        execution: Managed interpreter and shell settings.
        agent: Agent providing the observation encoding callback.

    Returns:
        Assistant history, tool feedback, and completion diagnostics.
    """
    turn = response.output
    if not isinstance(turn, ToolTurn):
        raise TypeError("Tool agent must return ToolTurn")
    event = response.trajectory[-1]
    assistant = {key: event[key] for key in ("role", "content", "tool_calls") if key in event}
    if turn.error:
        return SpreadsheetObservation(
            assistant=assistant, messages=[{"role": "user", "content": f"Observation: {turn.error}"}], output=turn.text
        )
    feedback = []
    loaded = False
    for call in turn.calls:
        if call.name not in {tool["function"]["name"] for tool in context.tools}:
            observation = f"Error: unknown tool {call.name}"
        else:
            observation = call.error or run_tool(call.name, call.arguments, context.directory, execution)
        if call.name == "skill" and not call.error and not observation.startswith("Error:"):
            loaded = True
            feedback.extend(
                [
                    agent.observation(call, "Result of 'skill' delivered in the following user message."),
                    {"role": "user", "content": observation},
                ]
            )
        else:
            feedback.append(agent.observation(call, observation))
    return SpreadsheetObservation(
        assistant=assistant, messages=feedback, complete=not turn.calls, skill_loaded=loaded, output=turn.text
    )
