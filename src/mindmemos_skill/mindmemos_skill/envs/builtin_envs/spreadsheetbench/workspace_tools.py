"""Interact through workspace file, shell, and optional skill tools."""

import json
from pathlib import Path

from ....agents import OpenAIChatAgent
from ....agents.typing import ToolTurn
from ..common import total_usage
from .config import SpreadsheetBenchConfig
from .prompts.workspace_tools import SYSTEM_PROMPT
from .tools import TOOLS, run_tool
from .typing import SpreadsheetBenchTask, SpreadsheetInteractionResult


def run_workspace_tools(
    task: SpreadsheetBenchTask,
    text: str,
    directories: list[Path],
    workspace: Path,
    config: SpreadsheetBenchConfig,
    agent: OpenAIChatAgent,
) -> SpreadsheetInteractionResult:
    """Run an independent workspace-tool conversation for each case.

    Args:
        task: Public instructions and case identities.
        text: Rendered skill guidance.
        directories: Prepared public case directories.
        workspace: Directory for prompt snapshots.
        config: Interaction and execution settings.
        agent: Configured tool protocol adapter.

    Returns:
        Public trajectory, usage, and skill-loading diagnostics.
    """
    system = SYSTEM_PROMPT
    tools = TOOLS
    if config.skill_loading_mode == "system_prompt" and text.strip():
        tools = [tool for tool in TOOLS if tool["function"]["name"] != "skill"]
        system += f"\n\n## Skill\n{text.strip()}"
    elif text.strip():
        system += (
            "\n\nA `skill` tool is available with expert skills: xlsx. Call it first "
            '(e.g. skill(name="xlsx")) to load detailed guidance and the absolute '
            "path to reusable reference scripts before you start."
        )
    if config.tool_agent == "text_action":
        system += (
            '\n\nCall tools using Action: followed by a JSON object with "name" and "arguments". '
            "Wait for Observation: before continuing. To finish, output ACTION: TASK_COMPLETE.\n"
            f"Available tools: {json.dumps(tools)}"
        )
    parts = ["The source file 'input.xlsx' is in your working directory.", f"Task:\n{task.instruction}"]
    if task.answer_position.strip():
        parts.append(
            "Answer position (cells to verify) — these are the cells that will be "
            "graded, so make sure their final computed values are correct:\n" + task.answer_position.strip()
        )
    parts.append("Complete the task and save the result as 'output.xlsx' (do not modify 'input.xlsx').")
    user = "\n\n".join(parts)
    if config.execution.shell_command[0].lower().endswith(("cmd.exe", "powershell.exe", "pwsh.exe", "pwsh")):
        user += "\n\nShell command prefix: " + json.dumps(config.execution.shell_command)
    (workspace / "system.md").write_text(system, encoding="utf-8")
    (workspace / "task.md").write_text(user, encoding="utf-8")
    responses, trajectory = [], []
    output = ""
    skill_loaded_all = True
    for case, directory in zip(task.cases, directories):
        if config.skill_loading_mode == "tool":
            skill_path = directory / "skills/xlsx/SKILL.md"
            skill_path.parent.mkdir(parents=True, exist_ok=True)
            skill_path.write_text(text, encoding="utf-8")
        loaded = config.skill_loading_mode == "system_prompt" and bool(text.strip())
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        if trajectory:
            trajectory.append({"role": "system", "content": f"[CASE {case.number}]"})
        trajectory.extend(messages)
        for _ in range(config.max_turns):
            response = agent.run(messages=messages, tools=tools)
            turn = response.output
            if not isinstance(turn, ToolTurn):
                raise TypeError("Tool agent must return ToolTurn")
            responses.append(response)
            output = turn.text
            event = response.trajectory[-1]
            assistant = {key: event[key] for key in ("role", "content", "tool_calls") if key in event}
            messages.append(assistant)
            trajectory.append(event)
            if turn.error:
                feedback = {"role": "user", "content": f"Observation: {turn.error}"}
                messages.append(feedback)
                trajectory.append(feedback)
                continue
            if not turn.calls:
                break
            for call in turn.calls:
                if call.name not in {tool["function"]["name"] for tool in tools}:
                    observation = f"Error: unknown tool {call.name}"
                else:
                    observation = call.error or run_tool(call.name, call.arguments, directory, config.execution)
                if call.name == "skill" and not call.error and not observation.startswith("Error:"):
                    loaded = True
                    feedback = [
                        agent.observation(call, "Result of 'skill' delivered in the following user message."),
                        {"role": "user", "content": observation},
                    ]
                else:
                    feedback = [agent.observation(call, observation)]
                messages.extend(feedback)
                trajectory.extend(feedback)
        skill_loaded_all = skill_loaded_all and loaded

    return SpreadsheetInteractionResult(
        output=output,
        trajectory=trajectory,
        usage=total_usage(responses),
        diagnostics={
            "n_turns": len(responses),
            "skill_loaded": skill_loaded_all,
            "skill_loading_mode": config.skill_loading_mode,
            "skill_tool_exposed": any(tool["function"]["name"] == "skill" for tool in tools),
        },
    )
