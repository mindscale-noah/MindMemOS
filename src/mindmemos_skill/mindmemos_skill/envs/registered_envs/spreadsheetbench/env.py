"""Attempt-isolated SpreadsheetBench environment."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import time
import weakref
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from pydantic import Field

from ....agents.base import Agent
from ....agents.react.tool import Tool
from ....registry import ComponentType, register
from ....typing import EnvConfig, Reward, Skill, Task, Trajectory
from ...base import BaseEnv, EnvRolloutContext, PreparedRollout
from .evaluator import compare_workbooks
from .prompts import build_messages

_INSTALL_RE = re.compile(
    r"^(?:sudo\s+)?(?:\w+=\S+\s+)*(?:(?:python[\d.]*\s+-m\s+)?pip[\d.]*\s+(?:install|uninstall)|"
    r"uv\s+(?:pip\s+(?:install|uninstall|sync)|add|remove)|(?:conda|mamba)\s+(?:install|remove|uninstall)|"
    r"poetry\s+(?:add|remove)|pipenv\s+(?:install|uninstall)|easy_install)\b",
    re.IGNORECASE,
)
_SPREADSHEETBENCH_THREAD_POOL_WORKERS = 32
_configured_event_loops: weakref.WeakSet[asyncio.AbstractEventLoop] = weakref.WeakSet()


def _configure_default_executor() -> None:
    loop = asyncio.get_running_loop()
    if loop in _configured_event_loops:
        return
    loop.set_default_executor(
        ThreadPoolExecutor(
            max_workers=_SPREADSHEETBENCH_THREAD_POOL_WORKERS,
            thread_name_prefix="spreadsheetbench",
        )
    )
    _configured_event_loops.add(loop)


def _is_utf8(value: bytes) -> bool:
    try:
        value.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


class SpreadsheetBenchEnvConfig(EnvConfig):
    max_turns: int = Field(default=15, ge=1)
    shell_timeout_seconds: int = Field(default=120, ge=1)


@register(type=ComponentType.ENV, name="spreadsheetbench")
class SpreadsheetBenchEnv(BaseEnv[SpreadsheetBenchEnvConfig]):
    """Run one workbook edit with request-scoped tools and runtime state."""

    config_type = SpreadsheetBenchEnvConfig

    async def _prepare(
        self,
        *,
        task: Task,
        skills: Sequence[Skill],
        context: EnvRolloutContext,
    ) -> PreparedRollout:
        prepared = await super()._prepare(task=task, skills=skills, context=context)
        if prepared.environment.running_dir is None:
            raise ValueError("SpreadsheetBench requires rollout.workspace_root")
        workspace = Path(prepared.environment.running_dir)
        source_dir = Path(str(task.metadata["src_dir"]))
        init_workbook = self._workbook(source_dir, "init")
        golden_workbook = self._workbook(source_dir, "golden")
        shutil.copyfile(init_workbook, workspace / "input.xlsx")
        prepared.agent_request.options["skill_injection_mode"] = "tool"
        tools = SpreadsheetTools(workspace, timeout_seconds=self.config.shell_timeout_seconds).as_tools()
        prepared.runtime_state = {
            "workspace": workspace,
            "golden_workbook": golden_workbook,
            "messages": build_messages(task=task, skill_names=[]),
            "initial_messages": [],
            "tools": tools,
            "skill_runtime": {},
            "error": None,
            "finished": False,
            "turns": 0,
            "evaluation": {},
        }
        return prepared

    async def _execute(self, *, agent: Agent[Any], prepared: PreparedRollout) -> Trajectory:
        _configure_default_executor()
        state = prepared.runtime_state
        messages = list(state["messages"])
        started_at = time.time()
        error: str | None = None
        finished = False
        turns = 0
        try:
            async with agent.on_skill_runtime_task(prepared.agent_request) as injection:
                messages = agent.apply_skill_injection(state["messages"], injection)
                state["initial_messages"] = list(messages)
                tools: list[Tool] = [*state["tools"], *injection.tools]
                tools_by_name = {tool.name: tool for tool in tools}
                if len(tools_by_name) != len(tools):
                    raise ValueError("Skill Runtime tool conflicts with a SpreadsheetBench tool")
                schemas = [tool.to_openai_schema() for tool in tools]
                for turns in range(1, self.config.max_turns + 1):
                    response = await agent.respond(prepared.agent_request, messages, tools=schemas)
                    assistant = response.to_assistant_message()
                    messages.append(assistant)
                    calls = assistant.get("tool_calls") or []
                    if not calls:
                        finished = True
                        break
                    for call in calls:
                        messages.extend(await self._call_tool(call, tools_by_name))
                state["skill_runtime"] = injection.metadata.get("skill_runtime", {})
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        ended_at = time.time()
        state.update({"messages": messages, "error": error, "finished": finished, "turns": turns})
        self._write_artifacts(prepared)
        return agent.build_trajectory(
            request=prepared.agent_request,
            messages=messages,
            started_at=started_at,
            ended_at=ended_at,
            n_turn=turns,
            is_success=error is None,
            error_info=error,
            metadata={
                "finished": finished,
                "turns": turns,
                "error": error,
                "instruction_type": prepared.agent_request.task.metadata.get("instruction_type"),
                "skill_runtime": state["skill_runtime"],
            },
        )

    async def _evaluate(self, *, trajectory: Trajectory, prepared: PreparedRollout) -> Reward:
        state = prepared.runtime_state
        task = prepared.agent_request.task
        position = str(task.metadata.get("answer_position") or "")
        answer_sheet = task.metadata.get("answer_sheet")
        if position and answer_sheet and "!" not in position:
            position = f"{answer_sheet}!{position}"
        try:
            correct, detail = compare_workbooks(
                state["golden_workbook"],
                state["workspace"] / "output.xlsx",
                position,
            )
        except Exception as exc:
            correct, detail = False, f"score error: {type(exc).__name__}: {exc}"
        state["evaluation"] = {"correct": correct, "detail": detail}
        if not correct and not (trajectory.metadata.get("error") or trajectory.execution.error_info):
            trajectory.metadata["error"] = detail
        return Reward(score=1.0 if correct else 0.0, detail=detail or None, metadata={"correct": correct})

    @staticmethod
    async def _call_tool(call: Any, tools: Mapping[str, Tool]) -> list[dict[str, Any]]:
        call = call if isinstance(call, Mapping) else {}
        function = call.get("function") if isinstance(call.get("function"), Mapping) else {}
        name = str(function.get("name") or "")
        call_id = str(call.get("id") or "")
        raw_arguments = function.get("arguments") or "{}"
        try:
            arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
        except json.JSONDecodeError:
            arguments = {"__raw__": raw_arguments}
        tool = tools.get(name)
        if tool is None:
            content = f"Error: unknown tool {name!r}"
        else:
            try:
                result = await tool.call(arguments or {})
                content = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
            except Exception as exc:
                content = f"Error: {type(exc).__name__}: {exc}"
        message = {"role": "tool", "tool_call_id": call_id, "name": name, "content": content}
        if tool is not None and tool.deliver_result_as_user:
            return [
                {**message, "content": f"Result of '{name}' delivered in the following user message."},
                {"role": "user", "content": content},
            ]
        return [message]

    @staticmethod
    def _workbook(source_dir: Path, kind: str) -> Path:
        marker = "init" if kind == "init" else "golden"
        hits = sorted(source_dir.glob(f"*{marker}*.xlsx"))
        if not hits:
            raise FileNotFoundError(f"No {kind} workbook in {source_dir}")
        return hits[0]

    @staticmethod
    def _write_artifacts(prepared: PreparedRollout) -> None:
        state = prepared.runtime_state
        workspace: Path = state["workspace"]
        messages = state["messages"]
        (workspace / "conversation.json").write_text(
            json.dumps(messages, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        initial = state["initial_messages"] or build_messages(task=prepared.agent_request.task, skill_names=[])
        (workspace / "target_system_prompt.txt").write_text(initial[0]["content"], encoding="utf-8")
        (workspace / "target_user_prompt.txt").write_text(initial[1]["content"], encoding="utf-8")


class SpreadsheetTools:
    def __init__(self, workspace: Path, *, timeout_seconds: int) -> None:
        self.workspace = workspace.resolve()
        self.timeout_seconds = timeout_seconds

    def _path(self, value: str) -> Path | None:
        raw = Path(value)
        path = (self.workspace / raw if not raw.is_absolute() else raw).resolve()
        return path if path == self.workspace or self.workspace in path.parents else None

    @staticmethod
    def _readable(path: Path) -> bool:
        if path.suffix in {".py", ".txt", ".json", ".csv", ".md"}:
            return True
        if path.suffix in {".doc", ".docx", ".pdf", ".xlsx", ".pptx", ".ppt", ".xls"}:
            return False
        try:
            chunk = path.read_bytes()[:1024]
            return b"\x00" not in chunk and _is_utf8(chunk)
        except Exception:
            return False

    def read(self, path: str) -> str:
        target = self._path(path)
        if target is None:
            return f"Error: access denied, {path} is outside the working directory"
        if not target.exists():
            return f"Error: File {target} not found"
        if not self._readable(target):
            return f"Error: {target.suffix} file cannot be read as plain text. "
        try:
            return target.read_text(encoding="utf-8")
        except Exception as exc:
            return f"Error: File {target} read error: {exc}"

    def write(self, path: str, content: str) -> str:
        target = self._path(path)
        if target is None:
            return f"Error: access denied, {path} is outside the working directory"
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return f"Successfully wrote {len(content)} characters to {target}"
        except Exception as exc:
            return f"Error writing file: {exc}"

    def edit(self, path: str, original_text: str, replacement_text: str) -> str:
        target = self._path(path)
        if target is None:
            return f"Error: access denied, {path} is outside the working directory"
        if not target.exists():
            return f"Error: File {target} not found"
        try:
            content = target.read_text(encoding="utf-8")
        except Exception as exc:
            return f"Error: File {target} read error: {exc}"
        count = content.count(original_text)
        if count == 0:
            return f"Error: original_text not found in {target}"
        if count > 1:
            return f"Error: original_text matched {count} times in {target}, must be unique"
        try:
            target.write_text(content.replace(original_text, replacement_text), encoding="utf-8")
            return f"Successfully edited {target}"
        except Exception as exc:
            return f"Error writing file: {exc}"

    def shell(self, commands: list[str], timeout_ms: int | None = None) -> str:
        timeout = timeout_ms / 1000 if timeout_ms else self.timeout_seconds
        environment = {
            **os.environ,
            "PIP_NO_INDEX": "1",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "UV_OFFLINE": "1",
        }
        outputs: list[str] = []
        for command in commands:
            block = [f"Command: {command}"]
            if any(_INSTALL_RE.search(part.strip()) for part in re.split(r"&&|\|\||[;\n|]", command)):
                block.append(
                    "Error: installing packages is disabled in this environment. Required packages are preinstalled; "
                    "import and use them directly. Do not run pip/uv/conda install."
                )
            else:
                try:
                    result = subprocess.run(
                        command,
                        shell=True,
                        cwd=self.workspace,
                        env=environment,
                        capture_output=True,
                        text=True,
                        timeout=timeout,
                        check=False,
                    )
                    if result.stdout:
                        block.append(f"stdout:\n{result.stdout}")
                    if result.stderr:
                        block.append(f"stderr:\n{result.stderr}")
                    block.append(f"exit_code: {result.returncode}")
                except subprocess.TimeoutExpired:
                    block.append(f"Error: timed out after {int(timeout)}s")
                    outputs.append("\n".join(block))
                    break
                except Exception as exc:
                    block.append(f"Error: {exc}")
            outputs.append("\n".join(block))
        return "\n\n".join(outputs)

    def as_tools(self) -> list[Tool]:
        return [
            Tool(
                "read",
                "Read a text file's full contents. Returns an error for missing or non-text files.",
                self.read,
                {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path, relative to the working directory."}
                    },
                    "required": ["path"],
                },
            ),
            Tool(
                "write",
                "Write (overwrite) a text file, creating parent directories as needed.",
                self.write,
                {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path, relative to the working directory."},
                        "content": {"type": "string", "description": "Full text content to write."},
                    },
                    "required": ["path", "content"],
                },
            ),
            Tool(
                "edit",
                "Replace an exact, unique snippet of text in a file. Errors if the snippet is missing or matches more "
                "than once.",
                self.edit,
                {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path, relative to the working directory."},
                        "original_text": {
                            "type": "string",
                            "description": "Exact text to replace; must occur exactly once.",
                        },
                        "replacement_text": {"type": "string", "description": "Text to substitute in."},
                    },
                    "required": ["path", "original_text", "replacement_text"],
                },
            ),
            Tool(
                "shell",
                "Run shell commands sequentially in the working directory. Returns stdout/stderr/exit_code per command.",
                self.shell,
                {
                    "type": "object",
                    "properties": {
                        "commands": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Shell commands to run in order.",
                        },
                        "timeout_ms": {
                            "type": "integer",
                            "description": "Per-command timeout in milliseconds (default 120000).",
                        },
                    },
                    "required": ["commands"],
                },
            ),
        ]


class SpreadsheetSkillSet:
    def __init__(self, directories: dict[str, Path]) -> None:
        self.directories = directories

    def load(self, name: str) -> str:
        directory = self.directories.get(name)
        if directory is None:
            return f"Error: unknown skill '{name}'. Available skills: {', '.join(self.directories) or '(none)'}"
        instructions = (directory / "SKILL.md").read_text(encoding="utf-8")
        return (
            f"Loaded skill '{name}'.\n"
            f"Skill directory (absolute path): {directory}\n"
            "Reference files live under that directory; read or run them with the read/shell tools as needed.\n\n"
            f"----- {name}/SKILL.md -----\n{instructions}"
        )

    def as_tool(self) -> Tool:
        available = ", ".join(self.directories) or "(none)"
        return Tool(
            "skill",
            "Load an expert skill to get detailed instructions and the absolute path to its reusable reference "
            f"scripts. Call this before starting the task. Available skills: {available}.",
            self.load,
            {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": f"Skill to load. One of: {available}.",
                    }
                },
                "required": ["name"],
            },
            deliver_result_as_user=True,
        )


__all__ = ["SpreadsheetBenchEnv", "SpreadsheetBenchEnvConfig", "SpreadsheetSkillSet", "SpreadsheetTools"]
