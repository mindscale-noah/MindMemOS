"""Claude Code CLI agent implementation.

Wraps ``claude -p`` as a MindMemOS agent.  Skills are written to a temporary
workspace's ``.claude/skills/`` directory so Claude Code discovers and loads
them natively, rather than injecting them as plain text in the system prompt.
"""

from __future__ import annotations

import asyncio
import shutil
import time
from collections.abc import Mapping
from typing import Any

from ...registry import ComponentType, register
from ...typing import AgentExecutionRequest, AgentType, SkillInjectionMode, Trajectory
from ..base import Agent
from .config import ClaudeAgentConfig
from .skill_runtime import ClaudeSkillRuntime
from .support import (
    extract_cli_num_turns,
    extract_cli_session_id,
    extract_cli_trajectory_messages,
    parse_cli_events,
)


@register(type=ComponentType.AGENT, name=AgentType.CLAUDE.value, capabilities={"execute"})
class ClaudeAgent(Agent[ClaudeAgentConfig]):
    agent_type = AgentType.CLAUDE
    config_type = ClaudeAgentConfig
    skill_runtime_types = {SkillInjectionMode.FILESYSTEM: ClaudeSkillRuntime}

    def __init__(self, config: ClaudeAgentConfig | Mapping[str, Any]) -> None:
        super().__init__(config)
        self._cli_path: str | None = None

    async def execute(
        self,
        request: AgentExecutionRequest,
    ) -> Trajectory:
        started_at = time.time()
        config = self.config.with_overrides(request.options)

        trajectory_messages: list[dict[str, Any]] = []
        if request.task.system_prompt:
            trajectory_messages.append({"role": "system", "content": request.task.system_prompt})
        trajectory_messages.append({"role": "user", "content": request.task.instruction})

        # Resolve CLI path early so we don't create temp dirs on failure.
        try:
            cli = self._resolve_cli()
        except RuntimeError as exc:
            return self._error_result(request, started_at, trajectory_messages, str(exc))

        timeout = config.timeout_seconds
        try:
            with self.inject_skills(request.skills, mode=config.skill_injection_mode) as injection:
                returncode, stdout, stderr = await self._run_cli(
                    request=request,
                    config=config,
                    cli=cli,
                    skill_workspace=injection.workspace,
                )
        except asyncio.TimeoutError:
            return self._error_result(
                request,
                started_at,
                trajectory_messages,
                f"Claude CLI timed out after {timeout}s",
            )
        except Exception as e:
            return self._error_result(
                request,
                started_at,
                trajectory_messages,
                f"Claude CLI execution failed: {e}",
            )

        stdout_text = stdout.decode("utf-8", errors="replace") if stdout else ""
        stderr_text = (stderr.decode("utf-8", errors="replace") or "").strip() if stderr else ""

        events = parse_cli_events(stdout_text)
        session_id = extract_cli_session_id(events)
        num_turns = extract_cli_num_turns(events)
        stream_messages = extract_cli_trajectory_messages(events)
        ended_at = time.time()
        is_success = returncode == 0

        # Append all stream messages (assistant + user) for a faithful trace.
        trajectory_messages.extend(stream_messages)

        return self._build_trajectory(
            request=request,
            config=config,
            messages=trajectory_messages,
            started_at=started_at,
            ended_at=ended_at,
            n_turn=num_turns,
            is_success=is_success,
            error_info=stderr_text if not is_success else None,
            metadata={"session_id": session_id} if session_id else None,
        )

    async def _run_cli(
        self,
        *,
        request: AgentExecutionRequest,
        config: ClaudeAgentConfig,
        cli: str,
        skill_workspace: str | None,
    ) -> tuple[int, bytes, bytes]:
        """Run Claude while the Agent-owned Skill injection scope is active."""
        cmd = [cli, "-p", request.task.instruction]
        if config.model:
            cmd += ["--model", config.model]
        if config.max_turns:
            cmd += ["--max-turns", str(config.max_turns)]
        if request.task.system_prompt:
            cmd += ["--system-prompt", request.task.system_prompt]
        if skill_workspace:
            cmd += ["--add-dir", skill_workspace]
        cmd += ["--output-format", "stream-json", "--verbose"]
        if config.dangerously_skip_permissions:
            cmd += ["--dangerously-skip-permissions"]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=request.environment.running_dir,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=config.timeout_seconds)
        return proc.returncode, stdout, stderr

    def _resolve_cli(self) -> str:
        if self._cli_path is not None:
            return self._cli_path
        executable = self.config.cli_path or "claude"
        path = shutil.which(executable)
        if not path:
            raise RuntimeError(f"Claude CLI executable {executable!r} was not found.")
        self._cli_path = path
        return path

    def _error_result(
        self,
        request: AgentExecutionRequest,
        started_at: float,
        messages: list[dict[str, Any]],
        error_info: str,
    ) -> Trajectory:
        ended_at = time.time()
        return self._build_trajectory(
            request=request,
            config=self.config.with_overrides(request.options),
            messages=messages,
            started_at=started_at,
            ended_at=ended_at,
            n_turn=0,
            is_success=False,
            error_info=error_info,
        )
