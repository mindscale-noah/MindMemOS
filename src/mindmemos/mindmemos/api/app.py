"""FastAPI application factory and entry point.

``make dev`` launches uvicorn against the module-level ``app`` below. All
process-level initialization (config, logging, tracing, DB schema) happens in
the lifespan handler so importing this module stays cheap and side-effect free.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from ..config import get_config, init_config_from_env
from ..errors import ApiError, InvalidConfigError
from ..infra import shutdown_tracer_provider
from ..infra.blob import get_blob_store
from ..infra.blob.signed_url import resolve_signing_secret
from ..infra.db import close_database_clients, ensure_database_schema
from ..infra.kafka import start_kafka, stop_kafka
from ..llm import (
    close_llm_clients,
    get_mm_embed_client,
    init_embed_client,
    init_llm_client,
    validate_embedding_dimension,
)
from ..logging import configure_logging, configure_tracing, get_logger
from ..pipelines.memory_db import ImageMemoryStore
from ..workers import register_workers
from .internal_routes import router as internal_router
from .routes import router as memory_router
from .skill_routes import router as skill_router

logger = get_logger(__name__)

# Generous on purpose: a cold local SigLIP2 server may need tens of seconds for its first
# forward pass; the bound only exists so a dead endpoint (litellm retries for minutes) boots.
_MM_EMBED_PROBE_TIMEOUT_SECONDS = 60.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize config, logging, tracing and DB on startup; tear down on exit."""
    config_source = init_config_from_env()
    cfg = get_config()

    configure_logging(level=cfg.telemetry.log_level)
    configure_tracing(cfg.telemetry)
    logger.info("starting mindmemos api", config_source=config_source)

    # Initialize databases (Qdrant + Neo4j): connect and ensure schema exists.
    await ensure_database_schema()
    logger.info("memory databases ready")

    # Blob-store misconfiguration must fail the boot, not the first image request.
    _check_blob_store()

    # startup instead of on the first request.
    init_llm_client()
    init_embed_client()
    await validate_embedding_dimension()
    await _check_multimodal_embedding()
    logger.info("llm clients ready")

    # Register all Kafka consumer handlers before starting; start_kafka is a
    # no-op when kafka.enabled is false, so this stays safe without a broker.
    register_workers()
    await start_kafka()

    try:
        yield
    finally:
        await stop_kafka()
        await _close_llm_clients()
        await _close_databases()
        logger.info("mindmemos api stopped")
        shutdown_tracer_provider()


def _check_blob_store() -> None:
    """Construct the configured blob store once at startup.

    The S3 backend imports ``minio`` lazily, so a missing ``mindmemos[s3]`` extra would
    otherwise pass validation and only surface as an ImportError -> HTTP 500 on the first
    image add or get-image call. The minio client also rejects some endpoints validation
    cannot (e.g. a non-numeric port), as a bare ``ValueError``.
    """

    try:
        get_blob_store()
    except ImportError as exc:
        raise InvalidConfigError(
            "blob_store.s3_endpoint",
            support=f"install the s3 extra (pip install 'mindmemos[s3]'): {exc}",
        ) from exc
    except ValueError as exc:
        raise InvalidConfigError("blob_store.s3_endpoint", support=str(exc)) from exc
    cfg = get_config()
    if cfg.algo_config.multimodal.mode != "off" and not resolve_signing_secret(
        cfg.blob_store.url_signing_secret, cfg.auth.gateway_jwt_secret
    ):
        # not fatal: presigned/public-mirror image URLs still work, but the signed
        # proxy tier is disabled rather than minted with a forgeable empty-key HMAC
        logger.warning(
            "image url signing disabled: set blob_store.url_signing_secret (or auth.gateway_jwt_secret); "
            "search hits fall back to presigned/public-mirror links only"
        )


async def _check_multimodal_embedding() -> None:
    """Probe the multimodal embedding endpoint once and check dimensions against config and Qdrant.

    A dimension mismatch is fatal: vectors would not fit the image collection. An unreachable
    endpoint only warns, so text memory keeps working; image indexing fails until it recovers.
    """

    client = get_mm_embed_client()
    if client is None:
        mode = get_config().algo_config.multimodal.mode
        if str(mode) != "off":
            logger.warning(
                "multimodal mode has no mm_embed_model_router endpoints: images are stored but never"
                " indexed, so multimodal searches return no image hits (enhanced also drops per-image"
                " descriptions and visual facts after VLM extraction)",
                mode=str(mode),
            )
        else:
            logger.info("multimodal embedding not configured; multimodal vanilla stores images without indexing them")
        return
    expected = client.dimensions
    store = ImageMemoryStore()
    collection_size = await store.collection_vector_size()
    if collection_size is not None and collection_size != expected:
        raise InvalidConfigError(
            "mm_embed_model_router.endpoints[].dimensions",
            support=f"{collection_size} to match the existing image collection (or use a new image_collection)",
        )
    if collection_size is not None and await store.collection_vector_size("text") != store.text_vector_size:
        # Collections created before images had a text vector (or for another text embedding size).
        raise InvalidConfigError(
            "database.qdrant.image_collection",
            support=f"a collection with a {store.text_vector_size}-d 'text' vector: use a new name or drop it",
        )
    try:
        vector = (
            await asyncio.wait_for(
                client.embed_texts(["dimension check"], task="startup.dimension_check"),
                timeout=_MM_EMBED_PROBE_TIMEOUT_SECONDS,
            )
        )[0]
    except InvalidConfigError:
        raise
    except TimeoutError:
        # The probe goes out under the router's full retry policy; an unbounded wait here
        # would stall the boot for minutes when the endpoint is down (same as unreachable).
        logger.warning(
            "multimodal embedding probe timed out",
            timeout_seconds=_MM_EMBED_PROBE_TIMEOUT_SECONDS,
        )
        return
    except Exception as exc:  # noqa: BLE001 - reachability problems must not block text memory
        if getattr(exc, "code", None) == "embedding.dimension_mismatch":
            raise InvalidConfigError("mm_embed_model_router.endpoints[].dimensions", support=str(exc)) from exc
        logger.warning("multimodal embedding endpoint unreachable at startup", error=str(exc)[:300])
        return
    logger.info("multimodal embedding ready", model=client.model, dimensions=len(vector))


async def _close_llm_clients() -> None:
    """Close LiteLLM-managed async HTTP clients."""
    try:
        await close_llm_clients()
    except Exception:  # noqa: BLE001 - shutdown best-effort
        logger.warning("failed to close llm clients", exc_info=True)


async def _close_databases() -> None:
    """Close Qdrant/Neo4j clients for the current event loop."""
    try:
        await close_database_clients()
    except Exception:  # noqa: BLE001 - shutdown best-effort
        logger.warning("failed to close database clients", exc_info=True)


def create_app() -> FastAPI:
    app = FastAPI(title="MindMemOS API", version="0.1.0", lifespan=lifespan)
    register_exception_handlers(app)

    @app.get("/healthz", tags=["meta"])
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(memory_router)
    app.include_router(skill_router)
    app.include_router(internal_router)

    return app


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _handle_api_error(_request: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"code": exc.code, "message": exc.message, "data": None},
        )

    @app.exception_handler(RequestValidationError)
    async def _handle_request_validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "code": "invalid_request",
                "message": _format_validation_errors(exc.errors()),
                "data": None,
            },
        )


def _format_validation_errors(errors: list[dict]) -> str:
    return "; ".join(_format_validation_error(error) for error in errors) or "request validation failed"


def _format_validation_error(error: dict) -> str:
    field = ".".join(str(part) for part in error.get("loc", ()))
    message = str(error.get("msg", "invalid value"))
    return f"{field}: {message}" if field else message


app = create_app()
