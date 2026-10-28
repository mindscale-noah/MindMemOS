"""Lazy resolution of the multimodal collaborators, shared by the add pipeline and the search engine.

Both accept explicit overrides (tests, request-scoped wiring) and otherwise resolve the process
singletons the same way; one implementation keeps the two resolution rules from drifting apart.
"""

from __future__ import annotations

from ..config import MultimodalConfig, get_config
from ..errors import ConfigNotInitializedError
from ..infra.blob import BlobStore, get_blob_store
from ..llm import MmEmbedClient, get_mm_embed_client
from .memory_db import ImageMemoryStore

MM_CLIENT_UNSET: object = object()
"""Constructor default meaning "no explicit mm-embed client": resolve the process singleton."""


class MultimodalDepsMixin:
    """Resolve the multimodal config, mm-embed client, blob store and image store on first use.

    Subclasses assign ``_explicit_multimodal_config``, ``_explicit_mm_client``, ``_blob_store``
    and ``_image_store`` in ``__init__``; nothing is touched at import or construction time, so
    a pipeline can exist before any external service is configured.
    """

    _explicit_multimodal_config: MultimodalConfig | None
    _explicit_mm_client: MmEmbedClient | None | object
    _blob_store: BlobStore | None
    _image_store: ImageMemoryStore | None

    def multimodal_config(self) -> MultimodalConfig:
        if self._explicit_multimodal_config is not None:
            return self._explicit_multimodal_config
        try:
            return get_config().algo_config.multimodal
        except ConfigNotInitializedError:
            return MultimodalConfig()

    def mm_client(self) -> MmEmbedClient | None:
        if self._explicit_mm_client is not MM_CLIENT_UNSET:
            return self._explicit_mm_client  # type: ignore[return-value]
        return get_mm_embed_client()

    def blobs(self) -> BlobStore:
        if self._blob_store is None:
            self._blob_store = get_blob_store()
        return self._blob_store

    def images(self) -> ImageMemoryStore:
        if self._image_store is None:
            self._image_store = ImageMemoryStore()
        return self._image_store
