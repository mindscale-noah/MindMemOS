"""Claude Agent SDK agent implementation.

Uses ``claude_agent_sdk`` (``query`` + streaming events) instead of the
``claude -p`` subprocess.  Skills are written to a temporary workspace's
``.claude/skills/`` directory and discovered natively by the SDK.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

from ...errors import SkillCapabilityUnavailableError
from ...registry import ComponentType, register
from ...typing import AgentExecutionRequest, AgentType, SkillInjectionMode, Trajectory
from ..base import Agent
from .config import ClaudeSDKAgentConfig
from .skill_runtime import ClaudeSkillRuntime
from .support import (
    convert_sdk_message,
    load_claude_agent_sdk,
)


@register(type=ComponentType.AGENT, name=AgentType.CLAUDE_SDK.value, capabilities={"execute"})
class ClaudeSDKAgent(Agent[ClaudeSDKAgentConfig]):
    """Agent that uses ``claude_agent_sdk`` to execute tasks with skill support."""

    agent_type = AgentType.CLAUDE_SDK
    config_type = ClaudeSDKAgentConfig
    skill_runtime_types = {SkillInjectionMode.FILESYSTEM: ClaudeSkillRuntime}

    def __init__(self, config: ClaudeSDKAgentConfig | Mapping[str, Any]) -> None:
        super().__init__(config)

    async def execute(
        self,
        request: AgentExecutionRequest,
    ) -> Trajectory:
        started_at = time.time()
        config = self.config.with_overrides(request.options)

        # Record input trajectory.
        trajectory_messages: list[dict[str, Any]] = []
        if request.task.system_prompt:
            trajectory_messages.append({"role": "system", "content": request.task.system_prompt})
        trajectory_messages.append({"role": "user", "content": request.task.instruction})

        try:
            ClaudeAgentOptions, query, AssistantMessage, ResultMessage, UserMessage = load_claude_agent_sdk()
        except SkillCapabilityUnavailableError as exc:
            return self._error_result(
                request=request,
                started_at=started_at,
                messages=trajectory_messages,
                error_info=f"{type(exc).__name__}: {exc}",
            )

        # Track result metadata from the stream.
        session_id: str | None = None
        num_turns: int = 0
        result_text: str = ""
        is_success = False
        error_info: str | None = None

        try:
            with self.inject_skills(request.skills, mode=config.skill_injection_mode) as injection:
                session_id, num_turns, result_text, is_success, error_info = await self._run_query(
                    request=request,
                    config=config,
                    skill_workspace=injection.workspace,
                    options_type=ClaudeAgentOptions,
                    query=query,
                    assistant_message_type=AssistantMessage,
                    result_message_type=ResultMessage,
                    user_message_type=UserMessage,
                    trajectory_messages=trajectory_messages,
                )
        except Exception as exc:
            is_success = False
            error_info = f"Claude Agent SDK query failed: {exc}"

        ended_at = time.time()

        # Ensure at least one assistant message if we have a result.
        if result_text and not any(m.get("role") == "assistant" for m in trajectory_messages):
            trajectory_messages.append({"role": "assistant", "content": result_text})

        return self._build_trajectory(
            request=request,
            config=config,
            messages=trajectory_messages,
            started_at=started_at,
            ended_at=ended_at,
            n_turn=num_turns or 1,
            is_success=is_success,
            error_info=error_info if not is_success else None,
            metadata={"session_id": session_id} if session_id else None,
        )

    async def _run_query(
        self,
        *,
        request: AgentExecutionRequest,
        config: ClaudeSDKAgentConfig,
        skill_workspace: str | None,
        options_type: Any,
        query: Any,
        assistant_message_type: type[Any],
        result_message_type: type[Any],
        user_message_type: type[Any],
        trajectory_messages: list[dict[str, Any]],
    ) -> tuple[str | None, int, str, bool, str | None]:
        """Run the SDK query while the Agent-owned Skill scope is active."""
        options = options_type(
            system_prompt=request.task.system_prompt,
            add_dirs=[skill_workspace] if skill_workspace else None,
            permission_mode=config.permission_mode,
            cwd=request.environment.running_dir,
            max_turns=config.max_turns,
            model=config.model,
        )
        session_id: str | None = None
        num_turns = 0
        result_text = ""
        is_success = False
        error_info: str | None = None
        async for message in query(prompt=request.task.instruction, options=options):
            if isinstance(message, result_message_type):
                session_id = getattr(message, "session_id", None) or session_id
                num_turns = getattr(message, "num_turns", 0) or 0
                result_text = message.result or ""
                is_success = not message.is_error
                if message.is_error:
                    error_info = result_text or "Claude Agent SDK returned an error result"
            elif isinstance(message, assistant_message_type):
                trajectory_messages.append(
                    convert_sdk_message(
                        message,
                        assistant_message_type=assistant_message_type,
                        user_message_type=user_message_type,
                    )
                )
            elif isinstance(message, user_message_type):
                trajectory_messages.extend(
                    convert_sdk_message(
                        message,
                        assistant_message_type=assistant_message_type,
                        user_message_type=user_message_type,
                    )
                )
        return session_id, num_turns, result_text, is_success, error_info

    def _error_result(
        self,
        *,
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
