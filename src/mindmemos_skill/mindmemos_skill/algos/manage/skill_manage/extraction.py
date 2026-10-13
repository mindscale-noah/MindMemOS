"""In-memory SSL extraction using the shared structured chat retry implementation."""

import json
from time import perf_counter

from ....agents import AgentResult, OpenAIChatAgent
from ....common.json import parse_model
from ....scene.typing import ProvidedSceneRecord, SceneExtractionResponse
from .errors import SceneGraphError, SceneGraphFailure
from .prompts import SSL_SYSTEM
from .typing import ChatModel, ExtractionSummary


class SceneExtractor(OpenAIChatAgent):
    """Adapt the existing graph chat port to shared parsing and bounded repair retries."""

    def __init__(self, client: ChatModel):
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
        except (ValueError, TypeError):
            self.summary = self.summary.model_copy(update={"error_type": "invalid_output"})
        raise SceneGraphError(SceneGraphFailure.MODEL_FAILED)

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
        except Exception:
            self.summary = self.summary.model_copy(update={"error_type": "transport"})
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
                raise SceneGraphError(SceneGraphFailure.INSUFFICIENT_SOURCE)
            return response.record
        finally:
            self.summary = self.summary.model_copy(update={"elapsed_seconds": perf_counter() - start})
