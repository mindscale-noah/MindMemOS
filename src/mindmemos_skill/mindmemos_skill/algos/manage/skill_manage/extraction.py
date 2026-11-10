"""In-memory SSL extraction using the shared structured chat retry implementation."""

import json
from time import perf_counter

from openai import APIStatusError
from opentelemetry.trace import get_current_span
from pydantic import ValidationError

from ....agents import AgentResult, OpenAIChatAgent
from ....common.json import parse_model
from ....logging import get_logger
from ....logging.tracing import traced
from ....scene.typing import ProvidedSceneRecord, SceneExtractionResponse
from .errors import SceneGraphError, SceneGraphFailure
from .prompts import SSL_SYSTEM
from .typing import ChatModel, ExtractionSummary


class SceneExtractor(OpenAIChatAgent):
    """Adapt the existing graph chat port to shared parsing and bounded repair retries."""

    def __init__(self, client: ChatModel | OpenAIChatAgent):
        """Borrow a chat client and retain only allowlisted usage accounting.

        Args:
            client: Configured chat transport; transport retries remain its responsibility.
        """
        super().__init__(
            client.config,
            output_parser=self._parse,
            parse_feedback=lambda error: (
                "The response violates the supplied schema or ID/ownership/transition/reachability constraints. "
                "Return the complete corrected JSON using only the original document."
            ),
        )
        self.client = client
        self.summary = ExtractionSummary()

    def _parse(self, text):
        """Sanitize structural failures before shared repair handling sees an exception."""
        try:
            return parse_model(text, SceneExtractionResponse)
        except (ValueError, TypeError) as error:
            self.summary = self.summary.model_copy(update={"error_type": "invalid_output"})
            details = {"error_type": type(error).__name__, "attempt": self.summary.attempts}
            if isinstance(error, ValidationError):
                details["validation_types"] = sorted(
                    {item["type"] for item in error.errors(include_input=False, include_context=False)}
                )
            get_logger(__name__).warning("scene.extraction.invalid_output", extra={"event_body": details})
        raise SceneGraphError(SceneGraphFailure.MODEL_FAILED)

    @traced("skill_manage.scene.model_call")
    async def run_messages(self, *, messages, tools=None, response_format=None) -> AgentResult:
        """Execute one structural attempt without retaining transport traces.

        Args:
            messages: In-memory initial request and optional repair exchange.
            tools: Unused; SSL extraction has no tool calls.
            response_format: JSON constraint forwarded to the injected client.

        Returns:
            Text and allowlisted numeric usage, without a trajectory.
        """
        self.summary = self.summary.model_copy(update={"attempts": self.summary.attempts + 1, "error_type": None})
        try:
            if isinstance(self.client, OpenAIChatAgent):
                transport = self.client.model if self.client.model is not None else self.client
                result = await transport.run_messages(messages=messages, response_format=response_format)
            else:
                user = messages[1]["content"]
                if len(messages) > 2:
                    user += "\nRepair exchange:\n" + json.dumps(messages[2:], ensure_ascii=False)
                result = await self.client.run(
                    system=messages[0]["content"], user=user, response_format=response_format
                )
        except Exception as error:
            self.summary = self.summary.model_copy(update={"error_type": "transport"})
            details = {"error_type": type(error).__name__, "attempt": self.summary.attempts}
            if isinstance(error, APIStatusError):
                details["http_status"] = error.status_code
            elif isinstance(error, RuntimeError):
                # Match only messages owned by our chat adapter, never arbitrary provider text.
                reasons = {
                    "The model returned no choices": "no_choices",
                }
                if str(error) in reasons:
                    details["reason"] = reasons[str(error)]
            for key, value in details.items():
                get_current_span().set_attribute(key, value)
            # Do not retain provider bodies, document text, or credential-bearing exceptions.
            get_logger(__name__).error("scene.extraction.transport_failed", extra={"event_body": details})
            raise SceneGraphError(SceneGraphFailure.MODEL_FAILED) from None
        usage = {
            key: value
            for key in ("input_tokens", "output_tokens", "total_tokens")
            if type(value := result.usage.get(key)) is int and value >= 0
        }
        totals = self.summary.model_dump(include={"input_tokens", "output_tokens", "total_tokens"})
        self.summary = self.summary.model_copy(update={key: value + usage.get(key, 0) for key, value in totals.items()})
        # Reject structured objects through the parser so malformed responses consume the same budget.
        output = result.output if isinstance(result.output, str) else ""
        return AgentResult(output=output, usage=usage)

    async def extract(self, document: str) -> ProvidedSceneRecord:
        """Extract one transient record; never return or persist a raw AgentResult.

        Args:
            document: Complete public SKILL.md after document-size validation.

        Returns:
            The same strict typed record accepted by the provided-input seam.

        Raises:
            SceneGraphError: Insufficient source or exhausted structural/transport attempts.
        """
        response = await self._extract_response(document)
        if response.record is None:
            raise SceneGraphError(SceneGraphFailure.INSUFFICIENT_SOURCE)
        return response.record

    @traced("skill_manage.scene.extract")
    async def _extract_response(self, document: str) -> SceneExtractionResponse:
        """Trace validated outcomes before translating insufficient source for the caller."""
        self.summary = ExtractionSummary()
        start = perf_counter()
        try:
            result = await self.run(
                system=SSL_SYSTEM,
                user=json.dumps(
                    {"skill_md": document, "schema": SceneExtractionResponse.model_json_schema()}, ensure_ascii=False
                ),
                response_format={"type": "json_object"},
            )
            response = result.output
            if not isinstance(response, SceneExtractionResponse):
                raise SceneGraphError(SceneGraphFailure.MODEL_FAILED)
            if response.record is None:
                self.summary = self.summary.model_copy(update={"error_type": "insufficient_source"})
            get_current_span().set_attribute("extraction.outcome", response.status)
            return response
        finally:
            self.summary = self.summary.model_copy(update={"elapsed_seconds": perf_counter() - start})
            for key, value in self.summary.model_dump(exclude_none=True).items():
                get_current_span().set_attribute(f"extraction.{key}", value)
