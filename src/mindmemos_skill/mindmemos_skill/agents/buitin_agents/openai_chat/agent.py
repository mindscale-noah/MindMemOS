"""OpenAI-compatible chat calls with optional tool protocol callbacks."""

import random
from collections.abc import Callable, Mapping
from copy import deepcopy
from functools import partial

from pydantic import BaseModel, JsonValue

from ....common.json import parse_model
from ....logging import get_logger
from ....logging.otel import debug_fields
from ....persistence.observability import get_log_context
from ...base import Agent
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
        output_parser: Callable[[str], str | BaseModel] | None = None,
        output_model: type[BaseModel] | None = None,
        parse_feedback: Callable[[Exception], str] | None = None,
    ) -> None:
        """Configure a shared model call with optional protocol callbacks.

        Args:
            config: Model transport settings.
            model: Optional injected transport with matching settings.
            parse_response: Decode raw results; None preserves plain text output.
            format_error: Convert parser ValueError into retry feedback.
            observation: Encode environment results for the next turn.
            native_tools: Whether to send tool schemas to the provider.
            output_parser: Validate or decode final text; parser exceptions trigger bounded retries.
            output_model: Extract, repair and validate JSON with this model; mutually
                exclusive with output_parser and tool parse_response.
            parse_feedback: Optional correction message renderer for feedback retries.
        """
        super().__init__(config)
        if model is not None and model.config != config:
            raise ValueError("Injected model config must match the agent config")
        self.model = model
        self.parse_response = parse_response
        self.format_error = format_error
        self.encode_observation = observation
        if output_model is not None:
            if output_parser is not None or parse_response is not None:
                raise ValueError("output_model cannot be combined with output_parser or tool parse_response")
            output_parser = partial(parse_model, model_type=output_model)
        if output_parser is not None and parse_response is not None:
            raise ValueError("output_parser and tool parse_response are mutually exclusive")
        self.output_parser = output_parser
        self.parse_feedback = parse_feedback or _parse_feedback
        self.native_tools = native_tools

    def run(
        self,
        *,
        system: str | None = None,
        user: str | None = None,
        messages: list[dict[str, JsonValue]] | None = None,
        tools: list[dict[str, JsonValue]] | None = None,
        response_format: dict[str, JsonValue] | None = None,
    ) -> AgentResult:
        """Invoke this Agent with run-scoped tracing during algorithm training.

        Args:
            system: Public system instruction when messages is omitted.
            user: Public task instruction when messages is omitted.
            messages: Explicit conversation history.
            tools: Optional tool schemas supplied to the model.
            response_format: Optional per-call output constraint.

        Returns:
            The original Agent result with its trajectory and usage.
        """

        def call():
            """Invoke the existing Agent behavior exactly once."""
            return self._run_untraced(
                system=system, user=user, messages=messages, tools=tools, response_format=response_format
            )

        session = get_log_context()
        if session is None:
            return call()
        input_payload = {
            "system": system,
            "user": user,
            "messages": messages,
            "tools": tools,
            "response_format": response_format,
        }
        with session.span(
            "algorithm.agent.call",
            {"algorithm.capture_level": session.level, "gen_ai.request.model": self.config.model},
        ) as span:
            try:
                result = call()
            except BaseException as exc:
                get_logger(__name__).error(
                    "agent.call",
                    extra={
                        "event_body": {
                            "model": self.config.model,
                            "status": "failed",
                            "error_type": type(exc).__name__,
                        }
                    },
                )
                if session.level == "debug":
                    get_logger(__name__).debug(
                        "agent.io",
                        extra={"event_body": debug_fields({"input": input_payload, "error_type": type(exc).__name__})},
                    )
                raise
            usage = result.usage
            span.set_attribute("algorithm.usage_reported", bool(usage))
            if isinstance(usage, Mapping):
                for key in ("input_tokens", "output_tokens", "total_tokens"):
                    value = usage.get(key)
                    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                        span.set_attribute(f"gen_ai.usage.{key}", value)
            get_logger(__name__).info(
                "agent.call",
                extra={"event_body": {"model": self.config.model, "status": "completed", "usage": usage}},
            )
            if session.level == "debug":
                get_logger(__name__).debug(
                    "agent.io",
                    extra={
                        "event_body": debug_fields({"input": input_payload, "output": result.model_dump(mode="json")})
                    },
                )
            return result

    def _run_untraced(
        self,
        *,
        system: str | None = None,
        user: str | None = None,
        messages: list[dict[str, JsonValue]] | None = None,
        tools: list[dict[str, JsonValue]] | None = None,
        response_format: dict[str, JsonValue] | None = None,
    ) -> AgentResult:
        """Call the model once and optionally decode the response.

        Args:
            system: Public instructions, required with user when messages is omitted.
            user: Public task text without private scoring inputs.
            messages: Conversation history, mutually exclusive with system/user.
            tools: Environment tool schemas; omitted from textual protocol requests.
            response_format: Per-call structured output constraint; overrides configuration.

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
        conversation = deepcopy(messages)
        attempts = []
        for attempt in range(self.config.parse_max_attempts if self.output_parser else 1):
            options = {"messages": conversation, "tools": tools if self.native_tools else None}
            if response_format is not None:
                options["response_format"] = response_format
            result = transport.run_messages(**options)
            attempts.append(result)
            if self.output_parser is None:
                break
            try:
                if not isinstance(result.output, str):
                    raise TypeError("Output parser requires text")
                parsed = self.output_parser(result.output)
                if not isinstance(parsed, (str, BaseModel)):
                    raise TypeError("Output parser must return str or a Pydantic model")
            except Exception as exc:
                if attempt + 1 == self.config.parse_max_attempts:
                    raise
                if self.config.parse_retry_mode == "feedback":
                    conversation.extend(
                        [
                            {"role": "assistant", "content": result.output},
                            {"role": "user", "content": self.parse_feedback(exc)},
                        ]
                    )
                continue
            usage = dict(result.usage)
            for key in ("input_tokens", "output_tokens", "total_tokens"):
                usage[key] = sum(int(item.usage.get(key, 0)) for item in attempts)
            usage["parse_attempts"] = len(attempts)
            usage["attempt_usage"] = [item.usage for item in attempts]
            return AgentResult(
                output=parsed,
                trajectory=[event for item in attempts for event in item.trajectory],
                usage=usage,
            )
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
        self,
        *,
        messages: list[dict[str, JsonValue]],
        tools: list[dict[str, JsonValue]] | None = None,
        response_format: dict[str, JsonValue] | None = None,
    ) -> AgentResult:
        """Execute one conversation turn, optionally requesting declared tools.

        Args:
            messages: Public conversation history in OpenAI chat format.
            tools: Explicit tool schemas; None disables tool calls.
            response_format: Override the configured JSON output constraint.

        Returns:
            Text output and full trace, including tool calls in the last event.

        Raises:
            RuntimeError: The response is refused, truncated, empty, or violates the requested protocol.
            ValueError: The configured credential is unavailable.
        """
        from openai import APIConnectionError, APIStatusError, DefaultHttpxClient, OpenAI
        from tenacity import RetryCallState, Retrying, retry_if_exception, stop_after_attempt, wait_exponential

        self.config.require_connection()
        api_key = self.config.api_key
        options = {}
        effective_format = response_format if response_format is not None else self.config.response_format
        if effective_format is not None:
            options["response_format"] = effective_format
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
            # Tenacity owns retries so the configured limit counts physical requests.
            max_retries=0,
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


def _parse_feedback(error: Exception) -> str:
    """Render the reference format-correction message."""
    return (
        f"Your previous reply could not be applied:\n{error}\n\n"
        "Fix exactly that problem and resend the COMPLETE corrected output in the same "
        "format as before. Do not apologize or add commentary."
    )
