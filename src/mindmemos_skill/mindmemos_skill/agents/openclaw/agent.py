"""OpenClaw CLI agent using native workspace Skill discovery."""

from __future__ import annotations

import asyncio
import json
import math
import os
import shutil
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ...registry import ComponentType, register
from ...typing import AgentExecutionRequest, AgentType, SkillInjectionMode, Trajectory
from ..base import Agent
from .config import OpenClawAgentConfig
from .skill_runtime import OpenClawSkillRuntime
from .support import (
    convert_session_events,
    count_assistant_turns,
    extract_error,
    extract_final_text,
    extract_session_file,
    extract_session_id,
    extract_transport,
    load_cli_result,
    read_session_events,
)


@register(type=ComponentType.AGENT, name=AgentType.OPENCLAW.value, capabilities={"execute"})
class OpenClawAgent(Agent[OpenClawAgentConfig]):
    """Run one local OpenClaw turn with an ephemeral Skill/config overlay."""

    agent_type = AgentType.OPENCLAW
    config_type = OpenClawAgentConfig
    skill_runtime_types = {SkillInjectionMode.FILESYSTEM: OpenClawSkillRuntime}

    def __init__(self, config: OpenClawAgentConfig | Mapping[str, Any]) -> None:
        super().__init__(config)
        self._cli_path: str | None = None

    async def execute(self, request: AgentExecutionRequest) -> Trajectory:
        started_at = time.time()
        config = self.config.with_overrides(request.options)
        fallback_messages = [{"role": "user", "content": self._compose_message(request)}]
        try:
            cli = self._resolve_cli(config)
            with self.inject_skills(request.skills, mode=config.skill_injection_mode) as injection:
                config_path = self._write_overlay_config(request, config, injection.workspace, injection.skill_names)
                command = self._build_command(cli, request, config)
                process = await asyncio.create_subprocess_exec(
                    *command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=request.environment.running_dir,
                    env=self._build_env(config, config_path),
                )
                try:
                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(),
                        timeout=config.timeout_seconds + 30,
                    )
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
                    raise
                stdout_text = stdout.decode("utf-8", errors="replace") if stdout else ""
                stderr_text = stderr.decode("utf-8", errors="replace").strip() if stderr else ""
                result = load_cli_result(stdout_text)
                session_file = extract_session_file(result) if result is not None else None
                native_events = read_session_events(session_file) if session_file is not None else []
                messages = convert_session_events(native_events)
                if not messages:
                    messages = fallback_messages
                    final_text = extract_final_text(result) if result is not None else ""
                    if final_text:
                        messages.append({"role": "assistant", "content": final_text})
                structured_error = (
                    extract_error(result) if result is not None else "OpenClaw CLI did not return a valid JSON result"
                )
                is_success = process.returncode == 0 and structured_error is None
                error_info = None
                if not is_success:
                    error_info = (
                        structured_error or stderr_text or f"OpenClaw CLI exited with code {process.returncode}"
                    )
                metadata = self._metadata(result, session_file)
                n_turn = count_assistant_turns(native_events)
                if n_turn == 0:
                    n_turn = sum(message.get("role") == "assistant" for message in messages)
                return self._build_trajectory(
                    request=request,
                    config=config,
                    messages=messages,
                    started_at=started_at,
                    ended_at=time.time(),
                    n_turn=n_turn,
                    is_success=is_success,
                    error_info=error_info,
                    metadata=metadata,
                )
        except asyncio.TimeoutError:
            error = f"OpenClaw CLI timed out after {config.timeout_seconds}s"
        except Exception as exc:
            error = f"OpenClaw CLI execution failed: {exc}"
        return self._build_trajectory(
            request=request,
            config=config,
            messages=fallback_messages,
            started_at=started_at,
            ended_at=time.time(),
            n_turn=0,
            is_success=False,
            error_info=error,
        )

    def _resolve_cli(self, config: OpenClawAgentConfig) -> str:
        if self._cli_path is not None:
            return self._cli_path
        executable = config.cli_path or "openclaw"
        path = shutil.which(executable)
        if path is None:
            raise RuntimeError(f"OpenClaw CLI executable {executable!r} was not found.")
        self._cli_path = path
        return path

    @staticmethod
    def _compose_message(request: AgentExecutionRequest) -> str:
        parts: list[str] = []
        if request.environment.running_dir:
            parts.append(f"Working directory: {request.environment.running_dir}")
        if request.task.system_prompt:
            parts.append(f"System instructions:\n{request.task.system_prompt}")
        parts.append(request.task.instruction)
        return "\n\n".join(parts)

    def _build_command(
        self,
        cli: str,
        request: AgentExecutionRequest,
        config: OpenClawAgentConfig,
    ) -> list[str]:
        command = [
            cli,
            "--no-color",
            "agent",
            "--local",
            "--json",
            "--agent",
            config.agent_id,
            "--session-id",
            request.trajectory_id,
            "--timeout",
            str(math.ceil(config.timeout_seconds)),
            "--message",
            self._compose_message(request),
        ]
        if config.model:
            command.extend(["--model", config.model])
        if config.thinking:
            command.extend(["--thinking", config.thinking])
        if config.verbose is not None:
            command.extend(["--verbose", "on" if config.verbose else "off"])
        return command

    def _write_overlay_config(
        self,
        request: AgentExecutionRequest,
        config: OpenClawAgentConfig,
        injection_workspace: str | None,
        skill_names: set[str],
    ) -> Path:
        if injection_workspace is None:
            raise RuntimeError("OpenClaw filesystem injection did not provide a workspace")
        source_path = self._source_config_path(config)
        payload: dict[str, Any] = {}
        if source_path.is_file():
            try:
                loaded = json.loads(source_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"OpenClaw config is not valid JSON: {source_path}") from exc
            if not isinstance(loaded, dict):
                raise RuntimeError(f"OpenClaw config root must be an object: {source_path}")
            payload = loaded

        workspace = request.environment.running_dir or injection_workspace
        agents = _object_field(payload, "agents")
        defaults = _object_field(agents, "defaults")
        defaults["workspace"] = workspace
        defaults["skills"] = sorted(skill_names)
        if config.model:
            defaults["model"] = {"primary": config.model}
        agent_list = agents.setdefault("list", [])
        if not isinstance(agent_list, list):
            raise RuntimeError("OpenClaw agents.list must be an array")
        target = next(
            (item for item in agent_list if isinstance(item, dict) and item.get("id") == config.agent_id), None
        )
        if target is None:
            target = {"id": config.agent_id}
            agent_list.append(target)
        target["workspace"] = workspace
        target["skills"] = sorted(skill_names)
        if config.model:
            target["model"] = {"primary": config.model}

        skills = _object_field(payload, "skills")
        skills["allowBundled"] = []
        _object_field(skills, "load")["extraDirs"] = [str(Path(injection_workspace) / "skills")]
        tools = _object_field(payload, "tools")
        tools["allow"] = list(config.allowed_tools)
        _object_field(payload, "gateway")["mode"] = "local"

        destination = Path(injection_workspace) / "openclaw.json"
        destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return destination

    @staticmethod
    def _source_config_path(config: OpenClawAgentConfig) -> Path:
        if config.config_path is not None:
            return config.config_path.expanduser()
        configured = os.environ.get("OPENCLAW_CONFIG_PATH")
        if configured:
            return Path(configured).expanduser()
        state_dir = OpenClawAgent._source_state_dir(config)
        return state_dir / "openclaw.json"

    @staticmethod
    def _source_state_dir(config: OpenClawAgentConfig) -> Path:
        if config.state_dir is not None:
            return config.state_dir.expanduser()
        configured = os.environ.get("OPENCLAW_STATE_DIR")
        return Path(configured).expanduser() if configured else Path.home() / ".openclaw"

    def _build_env(self, config: OpenClawAgentConfig, config_path: Path) -> dict[str, str]:
        environment = dict(os.environ)
        environment["OPENCLAW_CONFIG_PATH"] = str(config_path)
        environment["OPENCLAW_STATE_DIR"] = str(self._source_state_dir(config))
        environment.pop("OPENCLAW_PROFILE", None)
        return environment

    @staticmethod
    def _metadata(result: Mapping[str, Any] | None, session_file: Path | None) -> dict[str, Any] | None:
        metadata: dict[str, Any] = {}
        if result is not None:
            session_id = extract_session_id(result)
            transport = extract_transport(result)
            if session_id:
                metadata["session_id"] = session_id
            if transport:
                metadata["transport"] = transport
        if session_file is not None:
            metadata["session_file"] = str(session_file)
        return metadata or None


def _object_field(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.setdefault(key, {})
    if not isinstance(value, dict):
        raise RuntimeError(f"OpenClaw config {key!r} must be an object")
    return value


__all__ = ["OpenClawAgent"]
