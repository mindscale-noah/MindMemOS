"""Resolve ``ImageRef`` inputs (path / url / base64 / blob_id) to validated image bytes.

Every source is size-limited (before decoding or streaming, not after) and must decode to a
known raster format (checked by magic bytes), so an image reference can never be used to read
or relay arbitrary files. Paths must lie under ``blob_store.allowed_path_roots``; URLs are
fetched without following redirects to hosts that resolve to private addresses unless
``blob_store.url_allow_private_hosts`` is set. URL hosts are resolved once and the connection
is dialed to the resolved IP (Host header and TLS SNI keep the original hostname), so a
rebinding DNS cannot pass the private-address check and then answer the connection from
inside the network.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import ipaddress
import socket
import weakref
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import httpx

from ...config import BlobStoreConfig
from ...errors import BadRequestError
from ...infra.blob import BlobStore
from ...typing import ImageRef

_MAX_REDIRECTS = 3


class ImageLoadError(BadRequestError):
    code = "image.invalid"


@dataclass(frozen=True)
class _DialTarget:
    """Where to send one URL hop: the (IP-pinned) URL to dial plus the original hostname."""

    url: str
    host_header: str | None
    hostname: str | None


@dataclass(frozen=True)
class LoadedImage:
    data: bytes
    mime_type: str
    source_url: str | None = None


def sniff_image_mime(data: bytes) -> str | None:
    """Return the MIME type of a supported raster image, or None."""

    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:2] == b"BM" and len(data) > 26:
        return "image/bmp"
    return None


class ImageLoader:
    """Load image bytes for one project from any supported ``ImageRef`` source."""

    def __init__(
        self,
        cfg: BlobStoreConfig,
        blob_store: BlobStore,
        *,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._cfg = cfg
        self._blobs = blob_store
        self._http = http_client
        self._owned: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, httpx.AsyncClient] = (
            weakref.WeakKeyDictionary()
        )
        self._roots = [Path(root).expanduser().resolve() for root in cfg.allowed_path_roots]

    async def load(self, ref: ImageRef, *, namespace: str) -> LoadedImage:
        if ref.blob_id:
            found = await self._blobs.get(namespace, ref.blob_id)
            if found is None:
                raise ImageLoadError(f"unknown blob_id {ref.blob_id!r}", code="image.not_found")
            blob, data = found
            return LoadedImage(data=data, mime_type=blob.mime_type)
        if ref.data:
            return self._validated(_decode_base64(ref.data, max_bytes=self._cfg.max_image_bytes))
        if ref.path:
            return self._validated(await asyncio.to_thread(self._read_path, ref.path))
        if ref.url:
            data = await self._fetch(ref.url)
            return self._validated(data, source_url=ref.url)
        raise ImageLoadError("image has no source")

    def _validated(self, data: bytes, *, source_url: str | None = None) -> LoadedImage:
        if len(data) > self._cfg.max_image_bytes:
            raise ImageLoadError(f"image exceeds {self._cfg.max_image_bytes} bytes", code="image.too_large")
        mime = sniff_image_mime(data)
        if mime is None:
            raise ImageLoadError("unsupported image format (png, jpeg, gif, webp, bmp)", code="image.unsupported")
        return LoadedImage(data=data, mime_type=mime, source_url=source_url)

    def _read_path(self, raw: str) -> bytes:
        if not self._roots:
            raise ImageLoadError(
                "image paths are disabled (blob_store.allowed_path_roots is empty)", code="image.path_disabled"
            )
        path = Path(raw).expanduser().resolve()
        if not any(path.is_relative_to(root) for root in self._roots):
            raise ImageLoadError("image path is outside blob_store.allowed_path_roots", code="image.path_denied")
        if not path.is_file():
            raise ImageLoadError(f"image not found: {raw}", code="image.not_found")
        if path.stat().st_size > self._cfg.max_image_bytes:
            raise ImageLoadError(f"image exceeds {self._cfg.max_image_bytes} bytes", code="image.too_large")
        return path.read_bytes()

    async def _fetch(self, url: str) -> bytes:
        if not self._cfg.url_fetch_enabled:
            raise ImageLoadError("image URLs are disabled (blob_store.url_fetch_enabled)", code="image.url_disabled")
        client = self._http or self._client_for()
        try:
            current = url
            for _ in range(_MAX_REDIRECTS + 1):
                target = await self._resolve_url(current)
                headers = {"Host": target.host_header} if target.host_header else None
                extensions = (
                    {"sni_hostname": target.hostname}
                    if target.hostname and urlsplit(current).scheme == "https"
                    else None
                )
                async with client.stream(
                    "GET",
                    target.url,
                    headers=headers,
                    extensions=extensions,
                    timeout=self._cfg.url_fetch_timeout,
                    follow_redirects=False,
                ) as response:
                    if response.is_redirect:
                        current = urljoin(current, response.headers.get("location", ""))
                        continue
                    if response.status_code >= 400:
                        raise ImageLoadError(
                            f"image URL returned HTTP {response.status_code}", code="image.fetch_failed"
                        )
                    chunks: list[bytes] = []
                    total = 0
                    async for chunk in response.aiter_bytes():
                        total += len(chunk)
                        if total > self._cfg.max_image_bytes:
                            raise ImageLoadError(
                                f"image exceeds {self._cfg.max_image_bytes} bytes", code="image.too_large"
                            )
                        chunks.append(chunk)
                    return b"".join(chunks)
            raise ImageLoadError("too many redirects", code="image.fetch_failed")
        except httpx.HTTPError as exc:
            raise ImageLoadError(f"image URL fetch failed: {type(exc).__name__}", code="image.fetch_failed") from exc

    def _client_for(self) -> httpx.AsyncClient:
        """One shared client per event loop, so fetches of one request reuse their connections.

        Cached per loop on purpose: a client binds to the loop it first runs on, and test
        suites run a fresh loop per test; the server process has a single long-lived loop.
        """

        loop = asyncio.get_running_loop()
        client = self._owned.get(loop)
        if client is None:
            client = self._owned[loop] = httpx.AsyncClient()
        return client

    async def _resolve_url(self, url: str) -> _DialTarget:
        """Validate a URL hop and say where to dial it.

        The hostname is resolved once and the resolved IP is pinned into the dialed URL: httpx
        would otherwise resolve the hostname a second time when connecting, and a rebinding DNS
        could pass the private-address check here and serve the actual connection from a
        private address. The Host header and TLS SNI keep the original hostname; with
        ``url_allow_private_hosts`` there is nothing to enforce, so the URL is dialed as-is.
        """

        parts = urlsplit(url)
        if parts.scheme not in {"http", "https"} or not parts.hostname:
            raise ImageLoadError("image URL must be http(s)", code="image.url_denied")
        if self._cfg.url_allow_private_hosts:
            return _DialTarget(url=url, host_header=None, hostname=None)
        try:
            infos = await asyncio.get_running_loop().getaddrinfo(
                parts.hostname, parts.port or 443, type=socket.SOCK_STREAM
            )
        except OSError as exc:
            raise ImageLoadError("image URL host does not resolve", code="image.fetch_failed") from exc
        for info in infos:
            address = ipaddress.ip_address(info[4][0])
            if not address.is_global:
                raise ImageLoadError("image URL resolves to a non-public address", code="image.url_denied")
        pinned_ip = infos[0][4][0]
        netloc = f"[{pinned_ip}]" if ":" in pinned_ip else pinned_ip
        if parts.port is not None:
            netloc = f"{netloc}:{parts.port}"
        return _DialTarget(
            url=parts._replace(netloc=netloc).geturl(),
            host_header=parts.netloc.rpartition("@")[2] or parts.hostname,
            hostname=parts.hostname,
        )


def _decode_base64(raw: str, *, max_bytes: int) -> bytes:
    text = raw.strip()
    if text.startswith("data:"):
        header, _, text = text.partition(",")
        if ";base64" not in header:
            raise ImageLoadError("data URI must be base64 encoded")
    # Reject before decoding: a huge payload would otherwise be fully expanded in memory
    # before the size check sees it. Four base64 chars encode three bytes.
    if len(text) > (max_bytes // 3 + 1) * 4:
        raise ImageLoadError(f"image exceeds {max_bytes} bytes", code="image.too_large")
    try:
        return base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ImageLoadError("image data is not valid base64") from exc
