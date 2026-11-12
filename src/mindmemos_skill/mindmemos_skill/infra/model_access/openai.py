"""OpenAI-compatible requests through a direct provider or the Platform gateway."""

from __future__ import annotations

import random
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from contextlib import asynccontextmanager, contextmanager, nullcontext
from contextvars import ContextVar
from typing import TYPE_CHECKING, Literal

from ...logging import get_logger
from .platform import (
    PLATFORM_CHAT_MODEL,
    PLATFORM_EMBEDDING_MODEL,
    PlatformModelError,
    fetch_embedding_space,
    platform_connection,
)
from .typing import EmbeddingSpace

if TYPE_CHECKING:
    from openai import AsyncOpenAI
    from openai.types import CreateEmbeddingResponse
    from openai.types.chat import ChatCompletion
    from pydantic import JsonValue

    from ...agents.buitin_agents.openai_chat.config import OpenAIChatConfig
    from ..embedding.config import OpenAIEmbeddingConfig


_provider_models: ContextVar[list[str] | None] = ContextVar("skill_provider_models", default=None)


@contextmanager
def capture_provider_models() -> Iterator[list[str]]:
    """Collect actual model names from successful gateway chat calls."""
    models: list[str] = []
    token = _provider_models.set(models)
    try:
        yield models
    finally:
        _provider_models.reset(token)


def record_provider_model(model: str) -> None:
    """Record the provider model in the active trace, if one is bound.

    Args:
        model: Model name returned by the provider.
    """
    models = _provider_models.get()
    if models is not None:
        models.append(model)


def has_fixed_chat_model(config: OpenAIChatConfig) -> bool:
    """Report whether the configured model is known before sending a request.

    Args:
        config: Chat model settings.

    Returns:
        True when the configuration selects a single provider model.
    """
    return config.transport == "direct"


async def resolve_embedding_space(config: OpenAIEmbeddingConfig) -> EmbeddingSpace | None:
    """Resolve a vector space for a new build when routing selects one remotely.

    Args:
        config: Deployment embedding settings.

    Returns:
        Authorized space, or None when the configured endpoint already fixes the space.
    """
    if config.transport == "direct":
        return None
    return await fetch_embedding_space()


@asynccontextmanager
async def _client(
    config: OpenAIChatConfig | OpenAIEmbeddingConfig,
    *,
    capability: Literal["chat", "embedding"],
    sdk_retries: int,
    borrowed: AsyncOpenAI | None = None,
) -> AsyncIterator[tuple[AsyncOpenAI, str]]:
    """Resolve the model and open an owned SDK client or yield a borrowed one."""
    from openai import AsyncOpenAI, DefaultAsyncHttpxClient

    via_platform = config.transport == "platform_gateway"
    model = config.model
    if via_platform:
        model = PLATFORM_CHAT_MODEL if capability == "chat" else PLATFORM_EMBEDDING_MODEL
    if borrowed is not None:
        yield borrowed, model
        return

    if via_platform:
        base_url, api_key = platform_connection()
    else:
        base_url, api_key = config.base_url, config.api_key or "unused"

    http_client = None
    if via_platform or capability == "chat":
        http_client = DefaultAsyncHttpxClient(trust_env=False, follow_redirects=not via_platform)

    async with AsyncOpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=config.timeout_s,
        max_retries=0 if via_platform else sdk_retries,
        http_client=http_client,
    ) as client:
        yield client, model


@contextmanager
def _platform_errors() -> Iterator[None]:
    """Translate gateway SDK failures without exposing the upstream response."""
    from openai import APIConnectionError, APIStatusError

    try:
        yield
    except APIStatusError as exc:
        raise PlatformModelError(exc.status_code) from None
    except APIConnectionError:
        raise PlatformModelError(503) from None


async def complete_chat(
    config: OpenAIChatConfig, *, messages: list[dict[str, JsonValue]], **options: JsonValue
) -> ChatCompletion:
    """Send one chat request using the route and retry policy from the configuration.

    Args:
        config: Model connection, timeout and retry settings.
        messages: Public conversation history in OpenAI chat format.
        **options: Request parameters already assembled by the caller.

    Returns:
        Provider completion with the actual model name and usage.
    """
    fixed_model = has_fixed_chat_model(config)
    if fixed_model:
        config.require_connection()

    async with _client(config, capability="chat", sdk_retries=config.sdk_max_retries) as (client, model):

        async def request() -> ChatCompletion:
            """Send the prepared chat request on the selected connection."""
            return await client.chat.completions.create(model=model, messages=messages, **options)

        if fixed_model:
            response = await _retry_chat(config, request)
        else:
            with _platform_errors():
                response = await request()

    if not fixed_model and response.choices and response.model:
        record_provider_model(response.model)
    return response


async def _retry_chat(config: OpenAIChatConfig, request: Callable[[], Awaitable[ChatCompletion]]) -> ChatCompletion:
    """Preserve the direct chat retry schedule and retry diagnostics."""
    from openai import APIConnectionError, APIStatusError
    from tenacity import AsyncRetrying, RetryCallState, retry_if_exception, stop_after_attempt, wait_exponential

    def is_retryable(error: BaseException) -> bool:
        """Retry transport failures and transient provider status codes."""
        if config.retry_policy == "all":
            return True
        return isinstance(error, APIConnectionError) or (
            isinstance(error, APIStatusError)
            and (error.status_code in {408, 409, 425, 429} or error.status_code >= 500)
        )

    def log_retry(state: RetryCallState) -> None:
        """Log the scheduled retry and its cause before sleeping."""
        assert state.outcome is not None and state.next_action is not None
        error = state.outcome.exception()
        get_logger(__name__).warning(
            f"agent.retry | attempt={state.attempt_number}/{config.max_retries}"
            f" sleep={state.next_action.sleep:.3f}s | error={type(error).__name__}: {error}"
        )

    def retry_delay(state: RetryCallState) -> float:
        """Select the configured wait without changing the attempt count."""
        if config.retry_wait == "schedule":
            index = min(state.attempt_number - 1, len(config.retry_waits_s) - 1)
            return config.retry_waits_s[index]
        nominal = wait_exponential(multiplier=config.retry_initial_wait_s, max=config.retry_max_wait_s)(state)
        if config.retry_wait == "equal_jitter":
            return nominal * (0.5 + 0.5 * random.random())
        return nominal

    return await AsyncRetrying(
        stop=stop_after_attempt(config.max_retries + 1),
        wait=retry_delay,
        retry=retry_if_exception(is_retryable),
        before_sleep=log_retry,
        reraise=True,
    )(request)


@asynccontextmanager
async def open_embedding_batches(
    config: OpenAIEmbeddingConfig, *, client: AsyncOpenAI | None = None
) -> AsyncIterator[Callable[[list[str]], Awaitable[CreateEmbeddingResponse]]]:
    """Open a batch sender bound to the configured model and vector space.

    Args:
        config: Embedding connection and pinned vector settings.
        client: Optional caller-owned client.

    Yields:
        Async sender to share across all batches of an embedding call.
    """
    via_platform = config.transport == "platform_gateway"
    async with _client(config, capability="embedding", sdk_retries=config.max_retries, borrowed=client) as (sdk, model):
        options = {"dimensions": config.dimensions} if config.send_dimensions else {}
        if via_platform:
            if config.space_id is None:
                raise ValueError("Platform embeddings require a pinned vector space")
            options = {"dimensions": config.dimensions, "extra_body": {"space_id": config.space_id}}

        async def send(batch: list[str]) -> CreateEmbeddingResponse:
            """Send one batch and normalize provider errors."""
            try:
                with _platform_errors() if via_platform else nullcontext():
                    return await sdk.embeddings.create(model=model, input=batch, encoding_format="float", **options)
            except PlatformModelError:
                raise
            except Exception as exc:
                raise RuntimeError(f"Embedding request failed: {type(exc).__name__}") from None

        yield send
