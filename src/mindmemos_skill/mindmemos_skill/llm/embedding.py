"""Embedding client backed by :class:`litellm.Router`."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from time import perf_counter
from typing import TYPE_CHECKING, Any

from ..errors import EmbeddingDimensionError, ModelEndpointNotConfiguredError
from .recording import LLMCallSink, write_llm_call
from .router import Usage, dump_response, get_response_value, litellm_response_headers, usage_tokens

if TYPE_CHECKING:
    from litellm import Router

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class EmbeddingResponse:
    """Normalized embedding response returned by :class:`EmbedClient`."""

    embeddings: list[list[float]]
    model: str = ""
    usage: Usage = field(default_factory=Usage)
    raw_response: dict[str, Any] = field(default_factory=dict)


class EmbedClient:
    """Thin embedding wrapper around a pre-built LiteLLM router."""

    ALIAS = "embedding"

    def __init__(
        self,
        router: Router,
        *,
        default_model: str | None = ALIAS,
        call_sink: LLMCallSink | None = None,
    ) -> None:
        self._router = router
        self._default_model = default_model
        self._call_sink = call_sink

    async def embed(
        self,
        task: str,
        text: str | list[str],
        *,
        model: str | None = None,
        expected_dim: int | None = None,
        **kwargs: Any,
    ) -> EmbeddingResponse:
        """Embed one or more strings, optionally validating vector dimensions."""
        target = model or self._default_model
        if target is None:
            raise ModelEndpointNotConfiguredError("embedding")

        request_payload = {"model": target, "input": text, **kwargs}
        started_at = datetime.now(UTC)
        started_counter = perf_counter()
        try:
            response = await self._router.aembedding(model=target, input=text, **kwargs)
        except Exception as exc:
            finished_at = datetime.now(UTC)
            latency_ms = (perf_counter() - started_counter) * 1000
            await write_llm_call(
                self._call_sink,
                call_type="embedding",
                task=task,
                request=request_payload,
                response=None,
                model=target,
                input_tokens=None,
                output_tokens=None,
                total_tokens=None,
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
                started_at=started_at,
                finished_at=finished_at,
                latency_ms=latency_ms,
            )
            logger.exception(
                "LiteLLM embedding call failed for task=%s model=%s after %.2fms",
                task,
                target,
                latency_ms,
            )
            raise

        finished_at = datetime.now(UTC)
        latency_ms = (perf_counter() - started_counter) * 1000
        raw_response = dump_response(response)
        usage = usage_tokens(getattr(response, "usage", None))
        model_name = get_response_value(response, "model", target) or target
        await write_llm_call(
            self._call_sink,
            call_type="embedding",
            task=task,
            request=request_payload,
            response=raw_response,
            model=model_name,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            status="succeeded",
            error=None,
            started_at=started_at,
            finished_at=finished_at,
            latency_ms=latency_ms,
        )

        embeddings: list[list[float]] = []
        for item in getattr(response, "data", []) or []:
            vector = item.get("embedding") if isinstance(item, dict) else getattr(item, "embedding", None)
            embeddings.append(vector or [])

        if expected_dim is not None:
            for vector in embeddings:
                if len(vector) != expected_dim:
                    raise EmbeddingDimensionError(
                        expected=expected_dim,
                        actual=len(vector),
                        model=target,
                        task=task,
                    )

        headers = litellm_response_headers(response)
        logger.info(
            "LiteLLM embedding call completed for task=%s model=%s in %.2fms retries=%s/%s",
            task,
            target,
            latency_ms,
            headers.get("x-litellm-attempted-retries"),
            headers.get("x-litellm-max-retries"),
        )
        return EmbeddingResponse(
            embeddings=embeddings,
            model=model_name,
            usage=usage,
            raw_response=raw_response,
        )
