"""OpenAI-compatible chat calls with optional tool protocol callbacks."""

import random
from collections.abc import Callable

from pydantic import JsonValue

from ....logging import get_logger
from ...base import Agent
from ...constants import AGENT_API_KEY_ENV
from ...typing import AgentResult, ToolCall, ToolTurn
from .config import OpenAIChatConfig


@Agent.register("openai_chat")
class OpenAIChatAgent(Agent[OpenAIChatConfig]):
    """Send only explicitly supplied messages to the model."""

    config_type = OpenAIChatConfig
    """Configuration validated by the shared registry."""

    def __init__(
        self,
        config: OpenAIChatConfig,
        *,
        model: "OpenAIChatAgent | None" = None,
        parse_response: Callable[[AgentResult, OpenAIChatConfig], ToolTurn] | None = None,
        format_error: Callable[[ValueError], str] = str,
        observation: Callable[[ToolCall, str], dict[str, JsonValue]] | None = None,
        native_tools: bool = True,
    ) -> None:
        """Configure a shared model call with optional protocol callbacks.

        Args:
            config: Model transport settings.
            model: Optional injected transport with matching settings.
            parse_response: Decode raw results; None preserves plain text output.
            format_error: Convert parser ValueError into retry feedback.
            observation: Encode environment results for the next turn.
            native_tools: Whether to send tool schemas to the provider.
        """
        super().__init__(config)
        if model is not None and model.config != config:
            raise ValueError("Injected model config must match the agent config")
        self.model = model
        self.parse_response = parse_response
        self.format_error = format_error
        self.encode_observation = observation
        self.native_tools = native_tools

    def run(
        self,
        *,
        system: str | None = None,
        user: str | None = None,
        messages: list[dict[str, JsonValue]] | None = None,
        tools: list[dict[str, JsonValue]] | None = None,
    ) -> AgentResult:
        """Call the model once and optionally decode the response.

        Args:
            system: Public instructions, required with user when messages is omitted.
            user: Public task text without private scoring inputs.
            messages: Conversation history, mutually exclusive with system/user.
            tools: Environment tool schemas; omitted from textual protocol requests.

        Returns:
            Text or a decoded ToolTurn with the original trace and usage.

        Raises:
            ValueError: The message arguments are incomplete or conflicting.

        Provider failures propagate unchanged after transport retries.
        """
        if messages is None:
            if system is None or user is None:
                raise ValueError("Provide messages or both system and user")
            messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        elif system is not None or user is not None:
            raise ValueError("messages cannot be combined with system or user")
        transport = self.model if self.model is not None else self
        result = transport.run_messages(messages=messages, tools=tools if self.native_tools else None)
        if self.parse_response is None:
            return result
        try:
            turn = self.parse_response(result, self.config)
        except ValueError as exc:
            turn = ToolTurn(text=result.output, error=self.format_error(exc))
        return AgentResult(output=turn, trajectory=result.trajectory, usage=result.usage)

    def observation(self, call: ToolCall, content: str) -> dict[str, JsonValue]:
        """Encode tool feedback through the configured callback.

        Args:
            call: Invocation being answered.
            content: Verbatim environment result.

        Returns:
            A protocol-specific conversation message.

        Raises:
            ValueError: No observation callback was supplied.
        """
        if self.encode_observation is None:
            raise ValueError("Tool observations require an observation callback")
        return self.encode_observation(call, content)

    def run_messages(
        self, *, messages: list[dict[str, JsonValue]], tools: list[dict[str, JsonValue]] | None = None
    ) -> AgentResult:
        """Execute one conversation turn, optionally requesting declared tools.

        Args:
            messages: Public conversation history in OpenAI chat format.
            tools: Explicit tool schemas; None disables tool calls.

        Returns:
            Text output and full trace, including tool calls in the last event.

        Raises:
            RuntimeError: The response is refused, truncated, empty, or violates the requested protocol.
            ValueError: The configured credential is unavailable.
        """
        from openai import APIConnectionError, APIStatusError, DefaultHttpxClient, OpenAI
        from tenacity import RetryCallState, Retrying, retry_if_exception, stop_after_attempt, wait_exponential

        if self.config.api_key is None:
            raise ValueError(f"Provide api_key or set {AGENT_API_KEY_ENV}")
        api_key = self.config.api_key.get_secret_value()
        options = {}
        if self.config.max_completion_tokens is not None:
            options["max_completion_tokens"] = self.config.max_completion_tokens
        if self.config.reasoning_effort is not None:
            options["reasoning_effort"] = self.config.reasoning_effort
        if self.config.top_p is not None:
            options["top_p"] = self.config.top_p
        if self.config.presence_penalty is not None:
            options["presence_penalty"] = self.config.presence_penalty
        if self.config.seed is not None:
            options["seed"] = self.config.seed
        if self.config.stop:
            options["stop"] = self.config.stop
        if self.config.extra_body:
            options["extra_body"] = self.config.extra_body
        if self.config.temperature is not None:
            options["temperature"] = self.config.temperature
        if tools:
            options["tools"] = tools
            options["tool_choice"] = "auto"

        def is_retryable(error: BaseException) -> bool:
            """Retry transport failures and transient provider status codes."""
            if self.config.retry_policy == "all":
                return True
            return isinstance(error, APIConnectionError) or (
                isinstance(error, APIStatusError)
                and (error.status_code in {408, 409, 425, 429} or error.status_code >= 500)
            )

        def log_retry(state: RetryCallState) -> None:
            """Log the scheduled retry and its cause immediately before sleeping."""
            assert state.outcome is not None and state.next_action is not None
            error = state.outcome.exception()
            get_logger(__name__).warning(
                "OpenAI chat retry %d/%d; sleep %.3f seconds; reason: %s: %s",
                state.attempt_number,
                self.config.max_retries,
                state.next_action.sleep,
                type(error).__name__,
                error,
            )

        def retry_delay(state: RetryCallState) -> float:
            """Select the configured wait without changing request attempt counts."""
            if self.config.retry_wait == "schedule":
                index = min(state.attempt_number - 1, len(self.config.retry_waits_s) - 1)
                return self.config.retry_waits_s[index]
            nominal = wait_exponential(multiplier=self.config.retry_initial_wait_s, max=self.config.retry_max_wait_s)(
                state
            )
            if self.config.retry_wait == "equal_jitter":
                return nominal * (0.5 + 0.5 * random.random())
            return nominal

        with OpenAI(
            api_key=api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout_s,
            max_retries=self.config.sdk_max_retries,
            http_client=DefaultHttpxClient(trust_env=False),
        ) as client:
            response = Retrying(
                stop=stop_after_attempt(self.config.max_retries + 1),
                wait=retry_delay,
                retry=retry_if_exception(is_retryable),
                before_sleep=log_retry,
                reraise=True,
            )(
                client.chat.completions.create,
                model=self.config.model,
                messages=messages,
                **options,
            )
        if not response.choices:
            raise RuntimeError("The model returned no choices")
        choice = response.choices[0]
        tool_calls = choice.message.tool_calls or []
        valid_finish = choice.finish_reason == "stop" or (tools and tool_calls and choice.finish_reason == "tool_calls")
        if self.config.response_validation == "strict" and (
            not valid_finish or choice.message.refusal or (tool_calls and not tools)
        ):
            raise RuntimeError(f"The model did not return a complete text response: {choice.finish_reason}")
        output = choice.message.content or ""
        if self.config.strip_thinking and "</think>" in output:
            output = output.split("</think>")[1].strip()
        if self.config.response_validation == "strict" and not output.strip() and not tool_calls:
            raise RuntimeError("The model returned empty text")
        usage = response.usage.model_dump(mode="json") if response.usage is not None else {}
        if usage:
            usage["input_tokens"] = usage.pop("prompt_tokens")
            usage["output_tokens"] = usage.pop("completion_tokens")
        return AgentResult(
            output=output,
            trajectory=[
                *messages,
                {
                    "role": "assistant",
                    "content": output,
                    "model": response.model,
                    "response_id": response.id,
                    **({"tool_calls": [call.model_dump(mode="json") for call in tool_calls]} if tool_calls else {}),
                },
            ],
            usage=usage,
        )
