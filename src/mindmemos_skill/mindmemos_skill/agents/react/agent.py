"""OpenAI-compatible, tool-calling ReAct agent."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from ...llm import ChatResponse
from ...registry import ComponentRequirements, ComponentType, register
from ...typing import (
    AgentExecutionRequest,
    AgentType,
    SkillInjectionMode,
    Trajectory,
)
from ..base import Agent
from .config import ReactAgentConfig
from .skill_runtime import ReactSkillRuntime
from .tool import Tool

_SKILL_TOOL_NAME = "skill"


class ChatClient(Protocol):
    """The subset of :class:`LLMClient` required by the ReAct loop."""

    async def chat(
        self,
        task: str,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        **kwargs: Any,
    ) -> ChatResponse | None: ...


@register(
    type=ComponentType.AGENT,
    name=AgentType.REACT.value,
    capabilities={"execute"},
    requirements=ComponentRequirements(requires_model_ref=True),
)
class ReactAgent(Agent[ReactAgentConfig]):
    """A minimal ReAct loop over native OpenAI ``tool_calls`` messages.

    The model client and callable tools are runtime dependencies, not persisted
    configuration. Injected Skills are exposed through a reserved ``skill``
    tool so the trajectory records exactly which persisted version was loaded.
    """

    agent_type = AgentType.REACT
    config_type = ReactAgentConfig
    skill_runtime_types = {
        SkillInjectionMode.TOOL: ReactSkillRuntime,
        SkillInjectionMode.SYSTEM_PROMPT: ReactSkillRuntime,
    }

    def __init__(
        self,
        config: ReactAgentConfig | Mapping[str, Any],
        *,
        llm: ChatClient,
        tools: Sequence[Tool] = (),
    ) -> None:
        super().__init__(config)
        self._llm = llm
        self._tools: dict[str, Tool] = {}
        for candidate in tools:
            if candidate.name == _SKILL_TOOL_NAME:
                raise ValueError(f"{_SKILL_TOOL_NAME!r} is reserved for injected MindMemOS Skills")
            if candidate.name in self._tools:
                raise ValueError(f"duplicate tool name: {candidate.name!r}")
            self._tools[candidate.name] = candidate

    async def execute(self, request: AgentExecutionRequest) -> Trajectory:
        started_at = time.time()
        config = self.config.with_overrides(request.options)
        messages = self._initial_messages(request, config)
        turns = 0
        response_metadata: dict[str, Any] = {}

        try:
            async with self.on_skill_runtime_task(request, mode=config.skill_injection_mode) as injection:
                messages = self.apply_skill_injection(messages, injection)
                tools = dict(self._tools)
                tools.update({tool.name: tool for tool in injection.tools})

                schemas = [candidate.to_openai_schema() for candidate in tools.values()]
                for turns in range(1, config.max_turns + 1):
                    response = await self._call_llm(request, config, messages, schemas)
                    assistant = response.to_assistant_message()
                    messages.append(assistant)
                    response_metadata = {
                        "finish_reason": response.finish_reason,
                        "response_model": response.model,
                        **injection.metadata,
                    }

                    tool_calls = assistant.get("tool_calls") or []
                    if not tool_calls:
                        return self._trajectory(
                            request=request,
                            config=config,
                            messages=messages,
                            started_at=started_at,
                            turns=turns,
                            is_success=True,
                            error_info=None,
                            metadata=response_metadata,
                        )

                    for tool_call in tool_calls:
                        messages.extend(await self._execute_tool_call(tool_call, tools))

            error_info = f"ReAct agent reached max_turns={config.max_turns} before producing a final response"
        except Exception as exc:
            error_info = f"{type(exc).__name__}: {exc}"

        return self._trajectory(
            request=request,
            config=config,
            messages=messages,
            started_at=started_at,
            turns=turns,
            is_success=False,
            error_info=error_info,
            metadata=response_metadata,
        )

    async def _call_llm(
        self,
        request: AgentExecutionRequest,
        config: ReactAgentConfig,
        messages: list[dict[str, Any]],
        schemas: list[dict[str, Any]],
    ) -> ChatResponse:
        kwargs: dict[str, Any] = dict(config.model_kwargs)
        for name in ("temperature", "top_p", "max_tokens", "reasoning_effort"):
            value = getattr(config, name)
            if value is not None:
                kwargs[name] = value
        if schemas:
            kwargs["tools"] = schemas

        response = await self._llm.chat(
            task=request.task.task_id,
            messages=list(messages),
            model=config.model,
            **kwargs,
        )
        if response is None:
            raise RuntimeError("Chat model returned no response")
        return response

    async def respond(
        self,
        request: AgentExecutionRequest,
        messages: list[dict[str, Any]],
        *,
        tools: Sequence[dict[str, Any]] | None = (),
    ) -> ChatResponse:
        """Generate one benchmark-controlled turn without altering its messages."""

        config = self.config.with_overrides(request.options)
        kwargs: dict[str, Any] = dict(config.model_kwargs)
        for name in ("temperature", "top_p", "max_tokens", "reasoning_effort"):
            value = getattr(config, name)
            if value is not None:
                kwargs[name] = value
        if tools is not None:
            kwargs["tools"] = list(tools)

        response = await self._llm.chat(
            task=request.task.task_id,
            messages=list(messages),
            model=config.model,
            **kwargs,
        )
        if response is None:
            raise RuntimeError("Chat model returned no response")
        return response

    async def _execute_tool_call(
        self,
        tool_call: Any,
        tools: Mapping[str, Tool],
    ) -> list[dict[str, Any]]:
        call = tool_call if isinstance(tool_call, Mapping) else {}
        function = call.get("function")
        function = function if isinstance(function, Mapping) else {}
        name = function.get("name") if isinstance(function.get("name"), str) else ""
        call_id = call.get("id") if isinstance(call.get("id"), str) else ""

        raw_arguments = function.get("arguments") or "{}"
        try:
            arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
        except json.JSONDecodeError:
            arguments = {"__raw__": raw_arguments}
        arguments = arguments or {}

        candidate = tools.get(name)
        if candidate is None:
            return [self._tool_message(call_id, name, f"Error: unknown tool {name!r}")]

        try:
            result = await candidate.call(arguments)
            content = self._stringify_tool_result(result)
        except Exception as exc:
            content = f"Error: {type(exc).__name__}: {exc}"

        if candidate.deliver_result_as_user:
            return [
                self._tool_message(call_id, name, f"Result of '{name}' delivered in the following user message."),
                {"role": "user", "content": content},
            ]
        return [self._tool_message(call_id, name, content)]

    @staticmethod
    def _tool_message(call_id: str, name: str, content: str) -> dict[str, Any]:
        return {
            "role": "tool",
            "tool_call_id": call_id,
            "name": name,
            "content": content,
        }

    @staticmethod
    def _stringify_tool_result(result: Any) -> str:
        if isinstance(result, str):
            return result
        try:
            return json.dumps(result, ensure_ascii=False)
        except TypeError:
            return str(result)

    @staticmethod
    def _initial_messages(
        request: AgentExecutionRequest,
        config: ReactAgentConfig,
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        system_prompt = "\n\n".join(prompt for prompt in (config.system_prompt, request.task.system_prompt) if prompt)
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": request.task.instruction})
        return messages

    def _trajectory(
        self,
        *,
        request: AgentExecutionRequest,
        config: ReactAgentConfig,
        messages: list[dict[str, Any]],
        started_at: float,
        turns: int,
        is_success: bool,
        error_info: str | None,
        metadata: dict[str, Any],
    ) -> Trajectory:
        return self._build_trajectory(
            request=request,
            config=config,
            messages=messages,
            started_at=started_at,
            ended_at=time.time(),
            n_turn=turns,
            is_success=is_success,
            error_info=error_info,
            metadata={key: value for key, value in metadata.items() if value},
        )


__all__ = ["ReactAgent"]
