"""Workspace tools for the workspace spreadsheet protocol."""

import re
import subprocess
from pathlib import Path

from ....infra.execution import ExecutionConfig, execute
from ....infra.execution.constants import PIP_NO_INDEX_ENV, UV_OFFLINE_ENV


def _tool(name, description, properties, required):
    """Declare an environment-owned function schema."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


TOOLS = [
    _tool("read", "Read a text file in the working directory.", {"path": {"type": "string"}}, ["path"]),
    _tool(
        "write",
        "Write or overwrite a text file in the working directory.",
        {"path": {"type": "string"}, "content": {"type": "string"}},
        ["path", "content"],
    ),
    _tool(
        "edit",
        "Replace one exact, unique text snippet in a file.",
        {"path": {"type": "string"}, "original_text": {"type": "string"}, "replacement_text": {"type": "string"}},
        ["path", "original_text", "replacement_text"],
    ),
    _tool(
        "shell",
        "Run shell commands sequentially in the working directory. Use Python/openpyxl to inspect and edit xlsx files.",
        {"commands": {"type": "array", "items": {"type": "string"}}, "timeout_ms": {"type": "integer"}},
        ["commands"],
    ),
    _tool(
        "skill",
        "Load an expert skill to get detailed instructions and the absolute path to its reusable reference scripts. Call this before starting the task. Available skills: xlsx.",
        {"name": {"type": "string", "description": "Skill to load. One of: xlsx."}},
        ["name"],
    ),
]
"""Released tool names, arguments, and descriptions."""

_INSTALL_RE = re.compile(
    r"^(?:sudo\s+)?(?:\w+=\S+\s+)*(?:(?:python[\d.]*\s+-m\s+)?pip[\d.]*\s+"
    r"(?:install|uninstall)|uv\s+(?:pip\s+(?:install|uninstall|sync)|add|remove)|"
    r"(?:conda|mamba)\s+(?:install|remove|uninstall))\b",
    re.IGNORECASE,
)
"""Recognize package installation commands rejected by the reference protocol."""


def _resolve(workspace: Path, value: str) -> Path:
    """Resolve local paths and reject paths outside the public workspace."""
    relative = Path(value)
    path = (workspace / relative).resolve()
    if not path.is_relative_to(workspace.resolve()):
        raise ValueError("outside workspace")
    return path


def run_tool(name: str, args: dict, workspace: Path, config: ExecutionConfig) -> str:
    """Execute one public tool without exposing private reference workbooks.

    Args:
        name: Environment tool name.
        args: Decoded model arguments.
        workspace: Public workbook directory.
        config: Local shell runtime settings.

    Returns:
        Released observation text, with local workspace paths.
    """
    if name == "skill":
        path = _resolve(workspace, "skills/xlsx/SKILL.md")
        if args.get("name") != "xlsx" or not path.is_file():
            return "Error: unknown skill 'xlsx'. Available skills: xlsx"
        return (
            f"Loaded skill 'xlsx'.\nSkill directory (absolute path): {path.parent}\n"
            "Reference files live under that directory; read or run them with the "
            "read/shell tools as needed.\n\n"
            f"----- xlsx/SKILL.md -----\n{path.read_text(encoding='utf-8')}"
        )
    if name in {"read", "write", "edit"}:
        try:
            path = _resolve(workspace, str(args.get("path", "")))
        except ValueError:
            return "Error: access denied; path is outside the working directory"
    try:
        if name == "read":
            return path.read_text(encoding="utf-8")
        if name == "write":
            path.parent.mkdir(parents=True, exist_ok=True)
            content = str(args.get("content", ""))
            path.write_text(content, encoding="utf-8")
            return f"Successfully wrote {len(content)} characters to {path}"
        if name == "edit":
            content = path.read_text(encoding="utf-8")
            old = str(args.get("original_text", ""))
            if content.count(old) != 1:
                return f"Error: original_text matched {content.count(old)} times; expected exactly once"
            path.write_text(content.replace(old, str(args.get("replacement_text", ""))), encoding="utf-8")
            return f"Successfully edited {path}"
        if name == "shell":
            commands = args.get("commands")
            if isinstance(commands, str):
                commands = [commands]
            if not isinstance(commands, list):
                return (
                    "Error: protocol error: shell.commands must be a string or list of strings; "
                    + f"got {type(commands).__name__}"
                )
            for index, command in enumerate(commands):
                if not isinstance(command, str):
                    return (
                        "Error: protocol error: shell.commands must contain only strings; "
                        + f"item {index} is {type(command).__name__}"
                    )
            timeout = max(1, int(args.get("timeout_ms", config.timeout_s * 1000))) / 1000
            runtime = config.model_copy(
                update={
                    "timeout_s": timeout,
                    "environment": {**config.environment, PIP_NO_INDEX_ENV: "1", UV_OFFLINE_ENV: "1"},
                }
            )
            blocks = []
            for command in commands:
                if any(_INSTALL_RE.search(part.strip()) for part in re.split(r"&&|\|\||[;\n|]", command)):
                    blocks.append(f"Command: {command}\nError: installing packages is disabled")
                    continue
                result = execute([*config.shell_command, command], workspace, runtime)
                block = [f"Command: {command}"]
                if result.stdout:
                    block.append(f"stdout:\n{result.stdout}")
                if result.stderr:
                    block.append(f"stderr:\n{result.stderr}")
                block.append(f"exit_code: {result.returncode}")
                blocks.append("\n".join(block))
            return "\n\n".join(blocks)
    except subprocess.TimeoutExpired:
        return "Error: command timed out"
    except (OSError, ValueError, TypeError) as exc:
        return f"Error: {type(exc).__name__}: {exc}"
    return f"Error: unknown tool {name}"
