"""FastAPI request dependencies: authentication and AuthContext assembly.

The flow per request:

1. The bearer credential is read from the ``Authorization`` header.
2. The configured auth provider resolves it into account/project scope.
3. Config overrides are injected via ``update_config()`` (ContextVar — request-
   scoped, does not touch the global config).
4. A security-only :class:`~mindmemos.api.schemas.AuthContext` is returned.

Actor identity (user_id / app_id / session_id / agent_id) is supplied in the
request body for endpoints that need it and merged into a
:class:`~mindmemos.typing.memory.MemoryRequestContext` in the service layer
(``api.mappers.to_memory_request_context``).

The concrete credential type is selected by ``auth.mode``. Standalone memory
API deployments use ``api_key``; product deployments use ``gateway_jwt``.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Sequence

from fastapi import Depends, Header, Query
from opentelemetry import trace

from ..config import get_config, update_config
from ..errors import AuthenticationError, PermissionDeniedError
from ..infra.blob.signed_url import resolve_signing_secret, verify_image_url_signature
from ..logging import get_logger, traced
from ..pipelines.memory_db import MemoryDbReader, MemoryDbWriter
from .auth import GatewayJwtAuthProvider, get_auth_provider
from .schemas import AuthContext, ResolvedKey

logger = get_logger(__name__)
STANDALONE_ACCOUNT_ID = "memory_standalone"


@traced("auth.resolve_api_key", record_args=False, record_result=False)
def resolve_api_key(api_key: str) -> ResolvedKey:
    """Resolve a bearer credential into its account/project scope."""

    identity = get_auth_provider().resolve_api_key(api_key)
    return ResolvedKey(
        account_id=identity.account_id or STANDALONE_ACCOUNT_ID,
        project_id=identity.project_id,
        api_key_uuid=identity.key_id,
        memory_algorithm=identity.memory_algorithm,
        scopes=identity.scopes,
        user_override_config=identity.user_override_config,
        project_override_config=identity.project_override_config,
    )


def _extract_api_key(authorization: str | None) -> str:
    if not authorization:
        raise AuthenticationError("missing Authorization header", code="auth.missing_authorization")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise AuthenticationError(
            "expected 'Authorization: Bearer <api_key>'",
            code="auth.invalid_authorization_scheme",
        )
    return token.strip()


@traced("auth.get_request_context", record_args=False, record_result=False)
async def get_request_context(
    authorization: str | None = Header(default=None),
) -> AuthContext:
    """Resolve the bearer credential into a security-only :class:`AuthContext`.

    Config overrides from the resolved key are applied via ``update_config()``
    before business logic runs; they are not carried in the context.
    """
    api_key = _extract_api_key(authorization)
    resolved = resolve_api_key(api_key)

    update_config(resolved.user_override_config, resolved.project_override_config)

    ctx = AuthContext(
        request_id=str(uuid.uuid4()),
        account_id=resolved.account_id,
        project_id=resolved.project_id,
        api_key_uuid=resolved.api_key_uuid,
        memory_algorithm=resolved.memory_algorithm,
        scopes=resolved.scopes,
    )
    logger.debug(
        "auth context resolved",
        request_id=ctx.request_id,
        account_id=ctx.account_id,
        project_id=ctx.project_id,
    )
    annotate_request_trace(ctx)
    return ctx


@traced("auth.get_internal_request_context", record_args=False, record_result=False)
async def get_internal_request_context(
    authorization: str | None = Header(default=None),
) -> AuthContext:
    """Resolve the Gateway internal token into an :class:`AuthContext`."""
    token = _extract_api_key(authorization)
    provider = get_auth_provider()
    if not isinstance(provider, GatewayJwtAuthProvider):
        raise AuthenticationError("internal endpoints require gateway_jwt auth mode", code="auth.unsupported_mode")
    identity = provider.resolve_api_key(token)

    request_id = identity.request_id or str(uuid.uuid4())
    ctx = AuthContext(
        request_id=request_id,
        account_id=identity.account_id or STANDALONE_ACCOUNT_ID,
        project_id=identity.project_id,
        api_key_uuid=identity.key_id,
        memory_algorithm=identity.memory_algorithm,
        scopes=identity.scopes,
    )
    annotate_request_trace(ctx)
    return ctx


def annotate_request_trace(ctx: AuthContext) -> None:
    """Attach request identity once to the active authentication span.

    Stamps request identity (request_id, project_id, api_key_uuid, account_id) onto
    the active trace span so that all downstream operations (including LLM calls) can
    be attributed to the originating request.

    Args:
        ctx: Authentication context containing request identity information.
    """

    span = trace.get_current_span()
    if not span.get_span_context().is_valid:
        return
    span.set_attribute("request_id", ctx.request_id)
    span.set_attribute("project_id", ctx.project_id)
    span.set_attribute("api_key_uuid", ctx.api_key_uuid)
    span.set_attribute("account_id", ctx.account_id)


def require_scopes(*required_scopes: str):
    async def dependency(ctx: AuthContext = Depends(get_request_context)) -> AuthContext:
        ensure_scopes(ctx, required_scopes)
        return ctx

    return dependency


async def get_optional_request_context(authorization: str | None = Header(default=None)) -> AuthContext | None:
    """``get_request_context`` that yields None instead of raising when no header is sent."""

    if not authorization:
        return None
    return await get_request_context(authorization=authorization)


async def get_image_access_context(
    blob_id: str,
    user_id: str = Query(min_length=1),
    project_id: str | None = Query(default=None),
    expires: int | None = Query(default=None),
    signature: str | None = Query(default=None),
    bearer_ctx: AuthContext | None = Depends(get_optional_request_context),
) -> AuthContext:
    """Authenticate ``GET /v1/memory/images/{blob_id}`` by Bearer key **or** URL signature.

    An ``Authorization: Bearer`` header takes the standard path (api key, ``memory:read``
    scope); without a header the query parameters ``project_id`` / ``expires`` /
    ``signature`` must form a valid unexpired signature minted by ``_attach_image_urls``.
    The signed branch synthesizes a minimal context — the signature already binds
    ``project_id``/``blob_id``/``user_id``/``expires``, and the service still re-verifies
    blob ownership, so the placeholder identity fields are never consulted for access.
    """

    if bearer_ctx is not None:
        ensure_scopes(bearer_ctx, ("memory:read",))
        return bearer_ctx
    if signature is None or expires is None or project_id is None:
        raise AuthenticationError(
            "missing Authorization header or image url signature",
            code="auth.missing_authorization",
        )
    cfg = get_config()
    secret = resolve_signing_secret(cfg.blob_store.url_signing_secret, cfg.auth.gateway_jwt_secret)
    if not verify_image_url_signature(
        project_id, blob_id, user_id, expires, signature, secret, now=int(time.time())
    ):
        raise AuthenticationError("invalid or expired image url signature", code="image.signature_invalid")
    ctx = AuthContext(
        request_id=str(uuid.uuid4()),
        account_id=STANDALONE_ACCOUNT_ID,
        project_id=project_id,
        api_key_uuid="image-url-signature",
        memory_algorithm="vanilla",
        scopes=["memory:read"],
    )
    annotate_request_trace(ctx)
    return ctx


def ensure_scopes(ctx: AuthContext, required_scopes: Sequence[str]) -> None:
    scopes = set(ctx.scopes)
    if "memory:*" in scopes:
        return
    missing = [scope for scope in required_scopes if scope not in scopes]
    if missing:
        raise PermissionDeniedError("insufficient scope", code="auth.insufficient_scope")


def get_memory_db_reader() -> MemoryDbReader:
    """Return a lightweight memory DB reader orchestration dependency."""

    return MemoryDbReader()


def get_memory_db_writer() -> MemoryDbWriter:
    """Return a lightweight memory DB writer orchestration dependency."""

    return MemoryDbWriter()
