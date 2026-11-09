"""Map offline stage scopes to the existing Codex Agent and native permission profiles."""

import json
import os
import sys
from pathlib import Path

from .....agents.buitin_agents.codex.agent import CodexAgent
from .....agents.buitin_agents.codex.config import CodexConfig
from .....agents.typing import AgentResult
from ..constants import HOST_DIR
from .execution import run_cli
from .typing import SessionOptions

CODEX_HOME_ENV = "CODEX_HOME"
"""Child-only home containing this run's Codex configuration and transcripts."""

PATH_ENV = "PATH"
"""Executable search path forwarded to sandboxed tools."""

PYTHONPATH_ENV = "PYTHONPATH"
"""Package roots required by the installed internal preflight command."""

TEMP_ENV_NAMES = ("TMPDIR", "TMP", "TEMP")
"""Platform temporary-directory variables confined to this execution."""

PYTHON_UTF8_ENV = "PYTHONUTF8"
"""Enable consistent Unicode handling for preview scripts."""

PYTHON_ENCODING_ENV = "PYTHONIOENCODING"
"""Use UTF-8 for script stdout and stdin."""

PERMISSION_PROFILE = "offline_evolution"
"""Native permission profile generated for this invocation only."""

MANAGED_SETTINGS = {
    "approval_policy",
    "default_permissions",
    "permissions",
    "sandbox_mode",
    "sandbox_workspace_write",
    "shell_environment_policy",
    "project_doc_max_bytes",
    "project_root_markers",
    "skills",
    "mcp_servers",
    "hooks",
    "features",
    "web_search",
    "tools",
    "sqlite_home",
    "history",
    "model_provider",
    "model_providers",
}
"""Settings whose overrides would widen stage scopes or replace explicit connections."""


class _OfflineCodexAgent(CodexAgent):
    """Reuse the shared Codex implementation while specializing this invocation's execution."""

    session: SessionOptions
    """Stage scopes assigned by run_codex to this invocation's private instance."""

    async def _run_cli(self, command: list[str], workspace: Path, overrides: dict[str, str], prompt: str) -> str:
        """Apply offline persistence, native permissions, child home and event logging."""
        command.remove("--ephemeral")
        sandbox = command.index("--sandbox")
        del command[sandbox : sandbox + 2]
        if self.session.session_id:
            command[2:2] = ["resume", self.session.session_id]
        return await run_cli(
            command,
            workspace,
            self.config,
            {**overrides, CODEX_HOME_ENV: str(self.session.config_dir.resolve())},
            prompt,
            log_path=self.session.log_path,
        )


async def run_codex(config: CodexConfig, *, user: str, workspace: Path, session: SessionOptions) -> AgentResult:
    """Invoke the shared Codex implementation with persistent, permission-bound threads.

    Args:
        config: Shared Codex connection, model and optional native arguments.
        user: Stage instruction or gate feedback.
        workspace: Algorithm execution root.
        session: Allowed paths, isolated native home, thread ID and event log.

    Returns:
        Shared output with native events and the algorithm's cache-token field names.

    Raises:
        ValueError: Extra arguments override stage policy or token counts are inconsistent.
        RuntimeError: Codex cannot complete the invocation or enforce its sandbox.
    """
    _validate_arguments(config.extra_args)
    session.config_dir.mkdir(parents=True, exist_ok=True)
    scratch = workspace / ".codex-tmp"
    scratch.mkdir(parents=True, exist_ok=True)
    options = config.model_copy(update={"extra_args": [*config.extra_args, *session_arguments(session, scratch)]})
    agent = _OfflineCodexAgent(options)
    agent.session = session
    result = await agent.run(system="", user=user, workspace=workspace)
    usage = dict(result.usage)
    cached = usage.get("cached_input_tokens") or 0
    total = usage.get("input_tokens") or 0
    if not isinstance(cached, int) or not isinstance(total, int) or cached < 0 or cached > total:
        raise ValueError("Codex cached input usage is inconsistent with total input tokens")
    usage.update(input_tokens=total - cached, cache_read_input_tokens=cached)
    return result.model_copy(update={"usage": usage})


def session_arguments(session: SessionOptions, scratch: Path) -> list[str]:
    """Build native policy overrides through the existing extra_args field.

    Args:
        session: Read, write, read-only and denied paths for this stage.
        scratch: Run-local temporary directory for sandboxed Python helpers.

    Returns:
        Codex options containing no model credentials.
    """
    import mindmemos_sdk

    package = Path(__file__).resolve().parents[4]
    sdk = Path(mindmemos_sdk.__file__).resolve().parent
    permissions = {":minimal": "read"}
    for path in (Path(sys.prefix), package, sdk, *session.read_dirs):
        permissions[str(path.resolve())] = "read"
    for path in (*session.write_dirs, scratch):
        permissions[str(path.resolve())] = "write"
    for path in session.readonly_dirs:
        permissions[str(path.resolve())] = "read"
    for path in session.deny_paths:
        permissions[str(path.resolve())] = "deny"
    tool_env = {
        PATH_ENV: str(Path(sys.executable).parent) + os.pathsep + os.environ.get(PATH_ENV, os.defpath),
        PYTHONPATH_ENV: os.pathsep.join((str(package.parent), str(sdk.parent))),
        PYTHON_UTF8_ENV: "1",
        PYTHON_ENCODING_ENV: "utf-8",
        **{name: str(scratch.resolve()) for name in TEMP_ENV_NAMES},
    }
    settings = {
        "default_permissions": PERMISSION_PROFILE,
        f"permissions.{PERMISSION_PROFILE}.filesystem": permissions,
        f"permissions.{PERMISSION_PROFILE}.network.enabled": False,
        "shell_environment_policy.inherit": "none",
        "shell_environment_policy.set": tool_env,
        "project_doc_max_bytes": 0,
        "project_root_markers": [HOST_DIR],
        "mcp_servers": {},
        "features.multi_agent": False,
        "features.memories": False,
        "features.plugins": False,
        "features.hooks": False,
        "web_search": "disabled",
    }
    arguments = ["--ignore-user-config", "--ignore-rules", "--strict-config"]
    for key, value in settings.items():
        arguments += ["--config", f"{key}={_toml(value)}"]
    return arguments


def _toml(value) -> str:
    """Encode the small native settings subset as TOML rather than JSON objects."""
    if isinstance(value, dict):
        return "{" + ", ".join(f"{json.dumps(key)}={_toml(item)}" for key, item in value.items()) + "}"
    return json.dumps(value)


def _validate_arguments(arguments: list[str]) -> None:
    """Reject native options that conflict with managed scopes or thread persistence."""
    flags = {
        "resume",
        "fork",
        "review",
        "--ephemeral",
        "--sandbox",
        "-s",
        "--add-dir",
        "--cd",
        "-C",
        "--dangerously-bypass-approvals-and-sandbox",
        "--dangerously-bypass-hook-trust",
        "--yolo",
        "--approve-for-me",
        "--worktree",
        "--profile",
        "-p",
        "--enable",
        "--disable",
        "--ignore-user-config",
        "--ignore-rules",
        "--strict-config",
    }
    for index, argument in enumerate(arguments):
        if argument.split("=", 1)[0] in flags or (len(argument) > 2 and argument.startswith(("-s", "-C", "-p"))):
            raise ValueError("Codex extra_args conflict with offline permissions or session identity")
        setting = None
        if argument in {"--config", "-c"} and index + 1 < len(arguments):
            setting = arguments[index + 1]
        elif argument.startswith("--config="):
            setting = argument[len("--config=") :]
        elif argument.startswith("-c") and len(argument) > 2:
            setting = argument[2:].lstrip("=")
        if setting and setting.split("=", 1)[0].strip().split(".", 1)[0] in MANAGED_SETTINGS:
            raise ValueError("Codex extra_args override managed offline settings")
