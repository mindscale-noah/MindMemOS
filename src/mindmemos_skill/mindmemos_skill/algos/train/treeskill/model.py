"""Adapt recorded OpenAIChatAgent calls to the reference algorithm protocol."""

import asyncio
from collections.abc import Callable
from pathlib import Path
from threading import Lock
from time import perf_counter

from ....agents import AgentResult, OpenAIChatAgent, OpenAIChatConfig
from ....common.json import write_json
from .typing import ChatResponse


class RecordedChatModel:
    """Keep every physical model attempt, including parser-rejected responses."""

    def __init__(self, config: OpenAIChatConfig, directory: Path, *, factory=OpenAIChatAgent):
        """Configure transport and audit storage.

        Args:
            config: Default provider and generation settings.
            directory: Experiment-owned directory for individual calls.
            factory: Agent constructor, injectable for offline tests.
        """
        self.config = config
        self.directory = directory
        self.factory = factory
        self._lock = Lock()
        self._counter = 0

    async def chat(
        self, task, messages, format_parser: Callable | None = None, *, feedback_on_parse_error=False, **kwargs
    ):
        """Execute the upstream chat protocol using the shared agent implementation.

        Args:
            task: Algorithm stage label saved with each call.
            messages: Exact upstream conversation payload.
            format_parser: Optional validator returning a Pydantic model.
            feedback_on_parse_error: Append the reference correction message on parse failure.
            **kwargs: Stage-specific temperature, token budget and response format.

        Returns:
            Raw text and the optional parsed output.
        """
        options = dict(kwargs)
        if "max_tokens" in options:
            options["max_completion_tokens"] = options.pop("max_tokens")
        options["parse_retry_mode"] = "feedback" if feedback_on_parse_error else "direct"
        config = self.config.model_copy(update=options)
        with self._lock:
            self._counter += 1
            directory = self.directory / f"{self._counter:06d}"
        transport = _RecordedTransport(config, self.factory(config), directory, task)
        agent = OpenAIChatAgent(config, model=transport, output_parser=format_parser)
        result = await asyncio.to_thread(agent.run, messages=messages)
        if transport.last_result is None or not isinstance(transport.last_result.output, str):
            raise TypeError("Algorithm model must return text")
        return ChatResponse(content=transport.last_result.output, parsed=result.output if format_parser else None)


class _RecordedTransport(OpenAIChatAgent):
    """Record provider boundaries without changing request or retry behavior."""

    def __init__(self, config, transport, directory, task):
        """init  .

        Args:
            config: Validated execution and generation settings.
            transport: Explicit transport for this operation.
            directory: Experiment-owned directory for runtime checks and their artifacts.
            task: Stage audit label or public task supplied by the caller.
        """
        super().__init__(config)
        self.transport = transport
        self.directory = directory
        self.task = task
        self.attempt = 0
        self.last_result: AgentResult | None = None

    def run_messages(self, *, messages, tools=None, response_format=None):
        """Run messages.

        Args:
            messages: Explicit messages for this operation.
            tools: Explicit tools for this operation.
            response_format: Explicit response format for this operation.
        """
        self.attempt += 1
        path = self.directory / f"attempt-{self.attempt:02d}.json"
        payload = {
            "stage": self.task,
            "config": self.config.model_dump(mode="json", exclude={"api_key"}),
            "messages": messages,
            "response_format": response_format or self.config.response_format,
        }
        write_json(path, {**payload, "status": "running"})
        start = perf_counter()
        try:
            options = {"messages": messages, "tools": tools}
            if response_format is not None:
                options["response_format"] = response_format
            result = self.transport.run_messages(**options)
        except Exception as exc:
            write_json(
                path,
                {
                    **payload,
                    "status": "failed",
                    "duration_s": perf_counter() - start,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            raise
        self.last_result = result
        write_json(
            path,
            {
                **payload,
                "status": "completed",
                "duration_s": perf_counter() - start,
                "response": result.model_dump(mode="json"),
            },
        )
        return result


class RecordedRouter(OpenAIChatAgent):
    """Record task-time routing calls under the same experiment root."""

    def __init__(self, config, directory, *, factory=OpenAIChatAgent):
        """Configure the router without making a provider call.

        Args:
            config: Router generation settings.
            directory: Root for distinct routing call records.
            factory: Optional offline transport constructor.
        """
        super().__init__(config)
        self.directory = directory
        self.factory = factory
        self._counter = 0
        self._lock = Lock()

    def run(self, **kwargs):
        """Run and record one routing request.

        Args:
            **kwargs: Standard OpenAIChatAgent request arguments.

        Returns:
            Original agent response with trace and usage.
        """
        with self._lock:
            self._counter += 1
            directory = self.directory / f"{self._counter:06d}"
        transport = _RecordedTransport(self.config, self.factory(self.config), directory, "treeskill_subtree_routing")
        return OpenAIChatAgent(self.config, model=transport).run(**kwargs)
