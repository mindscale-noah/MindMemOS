"""Multimodal infrastructure config: the image-capable embedding router and the image blob store."""

from dataclasses import dataclass, field

from .base import secret_field

MM_EMBED_PROTOCOLS = frozenset({"openai"})
"""Wire protocols implemented by ``mindmemos.llm.mm_embedding`` adapters."""

MM_IMAGE_TRANSPORTS = frozenset({"base64", "url"})
BLOB_STORE_BACKENDS = frozenset({"local"})


@dataclass
class MmEmbedEndpointConfig:
    """One multimodal embedding endpoint: text and images are encoded into one vector space."""

    model: str
    """Model name sent to the endpoint."""

    api_base: str
    """Endpoint base URL, e.g. ``http://127.0.0.1:8090/v1``; the protocol appends its path."""

    api_key: str | None = secret_field(default=None)
    """Bearer token; may be empty for a local endpoint."""

    dimensions: int | None = None
    """Vector dimension; sizes the Qdrant image collection and is checked at startup."""

    protocol: str = "openai"
    """Wire protocol adapter, one of ``MM_EMBED_PROTOCOLS``."""

    image_transport: str = "base64"
    """How images are sent: ``base64`` data URIs, or ``url`` when the image has a public URL."""

    timeout: int = 60
    """Per-request timeout in seconds."""

    num_retries: int = 3
    """Retries for transient failures before failing over to the next endpoint."""

    max_batch_size: int = 16
    """Maximum inputs per request; larger batches are split."""

    extra_body: dict = field(default_factory=dict)
    """Extra request body fields passed through to the endpoint."""


@dataclass
class MmEmbedRouterConfig:
    endpoints: list[MmEmbedEndpointConfig] = field(default_factory=list)
    """Endpoints tried in order; an empty list disables image embedding (images are stored only)."""


@dataclass
class BlobStoreConfig:
    """Where raw images are kept and which image sources the server may read."""

    backend: str = "local"
    """Storage backend, one of ``BLOB_STORE_BACKENDS``; the interface is ready for object storage."""

    local_root: str = "data/blobs"
    """Root directory of the local backend; relative paths resolve against the repository root."""

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
