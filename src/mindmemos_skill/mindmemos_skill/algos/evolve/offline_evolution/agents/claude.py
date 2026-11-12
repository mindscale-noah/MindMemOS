"""Express offline stage permissions through the existing Claude Agent entry."""

import json
import os
from pathlib import Path
from uuid import uuid4

from .....agents.buitin_agents.claude.agent import ClaudeAgent
from .....agents.buitin_agents.claude.config import ClaudeConfig
from .....agents.typing import AgentResult
from .execution import run_cli
from .typing import SessionOptions

CLAUDE_CONFIG_DIR_ENV = "CLAUDE_CONFIG_DIR"
"""Run-local native configuration and transcript directory."""

CLAUDE_DISABLE_MEMORY_ENV = "CLAUDE_CODE_DISABLE_AUTO_MEMORY"
"""Disable automatic memory for historical evidence sessions."""

EXCLUDED_ENVIRONMENT = ("CLAUDECODE", "CLAUDE_CODE_USE_POWERSHELL_TOOL")
"""Parent-session and shell switches omitted from offline child processes."""

ANTHROPIC_ENV_PREFIX = "ANTHROPIC_"
"""Ambient connection and model aliases replaced by explicit Agent settings."""

READ_TOOLS = ("Read", "Glob", "Grep")
"""Native tools permitted to inspect selected directories."""

WRITE_TOOLS = ("Write", "Edit")
"""Native tools permitted to change selected directories."""


class _OfflineClaudeAgent(ClaudeAgent):
    """Specialize only native invocation while reusing the shared command and result handling."""

    session: SessionOptions
    """Scopes assigned to this private per-invocation instance by run_claude."""

    async def _run_cli(self, command: list[str], workspace: Path, overrides: dict[str, str], prompt: str) -> str:
        """Persist the offline session and keep its environment and logs local to the run."""
        command.remove("--no-session-persistence")
        excluded = [key for key in os.environ if key in EXCLUDED_ENVIRONMENT or key.startswith(ANTHROPIC_ENV_PREFIX)]
        return await run_cli(
            command,
            workspace,
            self.config,
            {
                **overrides,
                CLAUDE_CONFIG_DIR_ENV: str(self.session.config_dir.resolve()),
                CLAUDE_DISABLE_MEMORY_ENV: "1",
            },
            prompt,
            excluded_environment=excluded,
            log_path=self.session.log_path,
        )


async def run_claude(config: ClaudeConfig, *, user: str, workspace: Path, session: SessionOptions) -> AgentResult:
    """Invoke the shared Agent with offline-owned arguments and child environment.

    Args:
        config: Shared Claude connection, model, limits and optional native arguments.
        user: Stage instruction or gate feedback sent through stdin.
        workspace: Isolated run directory used as the native working directory.
        session: This stage's file scopes, observed identity and raw event path.

    Returns:
        Shared Agent output, native events and reported usage.

    Raises:
        ValueError: Preset arguments conflict with this stage's managed native flags.
        RuntimeError: The shared Agent cannot complete the invocation.
    """
    managed = {
        "--resume",
        "-r",
        "--continue",
        "-c",
        "--session-id",
        "--no-session-persistence",
        "--permission-mode",
        "--allowedTools",
        "--allowed-tools",
        "--disallowedTools",
        "--disallowed-tools",
        "--tools",
        "--settings",
        "--setting-sources",
        "--strict-mcp-config",
        "--mcp-config",
        "--dangerously-skip-permissions",
    }
    if any(argument.split("=", 1)[0] in managed for argument in config.extra_args):
        raise ValueError("Claude extra_args conflict with offline stage permissions or session identity")
    options = config.model_copy(update={"extra_args": [*config.extra_args, *session_arguments(session, workspace)]})
    session.config_dir.mkdir(parents=True, exist_ok=True)
    agent = _OfflineClaudeAgent(options)
    agent.session = session
    return await agent.run(system="", user=user, workspace=workspace)


def session_arguments(session: SessionOptions, workspace: Path) -> list[str]:
    """Build stage-specific arguments without duplicating the shared command builder.

    Args:
        session: Native file scopes and optional observed identity.
        workspace: Working directory used to deny relative Host paths.

    Returns:
        Native arguments appended through ClaudeConfig.extra_args.
    """
    rules = ["Bash"]
    rules += [_path_rule(tool, path) for tool in READ_TOOLS for path in (*session.read_dirs, *session.write_dirs)]
    rules += [_path_rule(tool, path) for tool in WRITE_TOOLS for path in session.write_dirs]
    arguments = [
        "--strict-mcp-config",
        "--tools",
        ",".join(("Bash", *READ_TOOLS, *WRITE_TOOLS)),
        "--allowedTools",
        *rules,
    ]
    denied = _deny_rules(session, workspace)
    denied += [_path_rule(tool, path) for tool in WRITE_TOOLS for path in session.readonly_dirs]
    if denied:
        arguments += ["--settings", json.dumps({"permissions": {"deny": denied}})]
    arguments += ["--resume", session.session_id] if session.session_id else ["--session-id", str(uuid4())]
    return arguments


def _rule_path(path: Path) -> str:
    """Convert an absolute path to Claude's cross-platform permission spelling."""
    absolute = path.resolve()
    value = absolute.as_posix()
    return "/" + absolute.drive[0].lower() + value[len(absolute.drive) :] if absolute.drive else value


def _path_rule(tool: str, path: Path) -> str:
    """Describe one file or directory tree in a native permission rule."""
    value = _rule_path(path)
    return f"{tool}(/{value})" if path.is_file() else f"{tool}(/{value}/**)"


def _deny_rules(session: SessionOptions, workspace: Path) -> list[str]:
    """Deny file tools and literal shell references to Host-owned paths."""
    rules = []
    root = workspace.resolve()
    for path in session.deny_paths:
        absolute = path.resolve()
        rules += [_path_rule(tool, path) for tool in (*READ_TOOLS, *WRITE_TOOLS)]
        spellings = {absolute.as_posix(), str(absolute), _rule_path(path)}
        if absolute.is_relative_to(root):
            spellings.add(absolute.relative_to(root).as_posix())
        rules += [f"Bash(*{value}*)" for value in sorted(spellings)]
    return rules
