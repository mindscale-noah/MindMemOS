"""Multimodal infrastructure config: the image-capable embedding router and the image blob store."""

from dataclasses import dataclass, field

from .base import secret_field

MM_IMAGE_TRANSPORTS = frozenset({"base64", "url"})


@dataclass
class MmEmbedEndpointConfig:
    """One multimodal embedding endpoint: text and images are encoded into one vector space."""

    model: str
    """LiteLLM model identifier with a provider prefix, e.g. ``openai/siglip2-base-patch16-384``;
    litellm strips the prefix on the wire, so the endpoint sees the bare model name."""

    api_base: str
    """Endpoint base URL, e.g. ``http://127.0.0.1:8090/v1``; litellm appends the embeddings path."""

    api_key: str | None = secret_field(default=None)
    """Bearer token; may be empty for a local endpoint."""

    dimensions: int | None = None
    """Vector dimension; sizes the Qdrant image collection and is checked at startup."""

    image_transport: str = "base64"
    """How images are sent: ``base64`` data URIs, or ``url`` when the image has a public URL."""

    timeout: int = 60
    """Per-request timeout in seconds."""

    num_retries: int = 3
    """Retries for transient failures before failing over to the next endpoint (LiteLLM field)."""

    max_batch_size: int = 16
    """Maximum inputs per request; larger batches are split. Shared by all endpoints."""

    rpm: int | None = None
    """Allowed requests per minute (LiteLLM field)."""

    tpm: int | None = None
    """Allowed tokens per minute (LiteLLM field)."""

    extra_body: dict = field(default_factory=dict)
    """Extra request body fields passed through to the endpoint."""


@dataclass
class MmEmbedRouterConfig:
    endpoints: list[MmEmbedEndpointConfig] = field(default_factory=list)
    """Endpoint deployments behind one LiteLLM alias; an empty list disables image
    embedding (images are stored only)."""

    routing_strategy: str = "simple-shuffle"
    """Routing strategy across deployments, matching the other model routers."""

    allowed_fails: int | None = None
    """Cooldown a deployment after this many consecutive failures (LiteLLM field)."""

    cool_down: int | float | None = None
    """Cooldown window in seconds for a failed deployment (LiteLLM field)."""

    retry_after: int | float = 0
    """Minimum delay in seconds before LiteLLM retries a failed endpoint."""

    dimensions_supported_models: list[str] = field(default_factory=list)
    """Model name prefixes (provider prefix stripped) allowed to receive the ``dimensions``
    request param. litellm otherwise drops ``dimensions`` for any openai-compatible model
    whose name lacks the literal ``text-embedding-3``; list ``siglip2`` here to keep it."""


@dataclass
class BlobStoreConfig:
    """Where raw images are kept and which image sources the server may read.

    Storage is S3-compatible only (MinIO, R2, OSS, AWS S3, …); the server talks
    to it through the minio client shipped with the ``s3`` extra.
    """

    s3_endpoint: str = ""
    """S3-compatible endpoint as an ``http://`` or ``https://`` URL (the scheme decides TLS), e.g. ``http://localhost:9100``."""

    s3_region: str = "us-east-1"
    """S3 region/location used when creating the bucket."""

    s3_bucket: str = ""
    """S3 bucket name; created automatically when missing."""

    s3_access_key: str | None = secret_field(default=None)
    """S3 access key; every request is signed with it."""

    s3_secret_key: str | None = secret_field(default=None)
    """S3 secret key; every request is signed with it."""

    s3_public_base_url: str | None = None
    """Base URL of a public path-style mirror (e.g. a CDN); enables ``BlobStore.public_url``."""

    s3_presign_endpoint: str | None = None
    """Endpoint baked into presigned image URLs when clients cannot reach ``s3_endpoint``
    (e.g. the server uses an internal MinIO address but clients fetch from a public gateway).
    Unset signs against ``s3_endpoint`` itself."""

    s3_presign_expiry_seconds: int = 3600
    """Validity window of presigned image URLs returned in search results (1..604800 seconds)."""

    proxy_url_expiry_seconds: int | None = None
    """Validity window of signed proxy image URLs (1..604800 seconds); None reuses
    ``s3_presign_expiry_seconds``."""

    proxy_public_base_url: str | None = None
    """Trusted absolute base (e.g. ``https://memory.example.com``) prepended to signed proxy
    image URLs; unset keeps them relative. Never derived from the request Host header."""

    url_signing_secret: str | None = secret_field(default=None)
    """HMAC secret for signed proxy image URLs; unset derives from ``auth.gateway_jwt_secret``."""

    max_image_bytes: int = 20 * 1024 * 1024
    """Images larger than this are rejected."""

    allowed_path_roots: list[str] = field(default_factory=list)
    """Directories an ``ImageRef.path`` may point into; empty disables path inputs."""

    url_fetch_enabled: bool = True
    """Whether ``ImageRef.url`` inputs are downloaded."""

    url_fetch_timeout: float = 20.0
    """Download timeout in seconds."""

    url_allow_private_hosts: bool = False
    """Allow URLs resolving to loopback/private/link-local addresses (off to prevent SSRF)."""
