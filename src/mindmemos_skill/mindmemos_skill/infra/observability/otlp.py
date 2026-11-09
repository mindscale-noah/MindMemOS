"""Export OpenTelemetry batches over native asynchronous HTTP."""

import asyncio
from typing import Literal

import httpx
from opentelemetry.exporter.otlp.proto.common._log_encoder import encode_logs
from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans
from opentelemetry.sdk._logs import ReadableLogRecord
from opentelemetry.sdk.trace import ReadableSpan
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_random_exponential

from ...logging import get_logger

logger = get_logger(__name__)


def _retryable(error: BaseException) -> bool:
    """Retry transport failures and transient collector responses."""
    return isinstance(error, httpx.TransportError) or (
        isinstance(error, httpx.HTTPStatusError)
        and (error.response.status_code in {408, 429} or error.response.status_code >= 500)
    )


class AsyncOTLPExporter:
    """Batch completed records without an SDK exporter thread."""

    def __init__(self, endpoint: str, timeout: float):
        """Configure an owned connection pool without making a request.

        Args:
            endpoint: OTLP/HTTP collector base URL.
            timeout: Total seconds available for one batch, including retries.
        """
        self.timeout = timeout
        self.client = httpx.AsyncClient(base_url=endpoint.rstrip("/") + "/", timeout=timeout)
        self.pending: list[tuple[Literal["logs", "traces"], ReadableLogRecord | ReadableSpan]] = []
        self.worker: asyncio.Task[None] | None = None
        self.failed_batches = 0
        self._failed_since_success = False

    def enqueue(self, kind: Literal["logs", "traces"], record: ReadableLogRecord | ReadableSpan) -> None:
        """Queue a record from a synchronous OpenTelemetry callback.

        Args:
            kind: OTLP signal name, either logs or traces.
            record: SDK record accepted by the corresponding protobuf encoder.
        """
        self.pending.append((kind, record))
        if self.worker is None or self.worker.done():
            self.worker = asyncio.create_task(self._drain())

    async def _send(self, kind: str, payload: bytes) -> None:
        """Retry transient HTTP failures within the configured batch deadline."""
        async with asyncio.timeout(self.timeout):
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(6),
                wait=wait_random_exponential(multiplier=1, max=32),
                retry=retry_if_exception(_retryable),
                reraise=True,
            ):
                with attempt:
                    response = await self.client.post(
                        f"v1/{kind}", content=payload, headers={"Content-Type": "application/x-protobuf"}
                    )
                    response.raise_for_status()

    async def _drain(self) -> None:
        """Export bounded batches without affecting local audit persistence."""
        while self.pending:
            batch, self.pending = self.pending[:512], self.pending[512:]
            for kind, encoder in (("logs", encode_logs), ("traces", encode_spans)):
                selected = [record for signal, record in batch if signal == kind]
                if not selected:
                    continue
                try:
                    await self._send(kind, encoder(selected).SerializeToString())
                except Exception as exc:
                    self.failed_batches += 1
                    self._failed_since_success = True
                    logger.exception(
                        "otel.export.failed",
                        extra={
                            "skip_otlp": True,
                            "event_body": {
                                "signal": kind,
                                "error_type": type(exc).__name__,
                                "failed_batches": self.failed_batches,
                            },
                        },
                    )
                else:
                    if self._failed_since_success:
                        self._failed_since_success = False
                        logger.info(
                            "otel.export.recovered",
                            extra={
                                "event_body": {
                                    "signal": kind,
                                    "failed_batches": self.failed_batches,
                                }
                            },
                        )

    async def close(self) -> None:
        """Drain queued exports and release the HTTP connection pool."""
        try:
            if self.worker is not None:
                await self.worker
        finally:
            await self.client.aclose()
