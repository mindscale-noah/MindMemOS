import asyncio
import base64
import socket

import pytest
from mindmemos.components.multimodal import ImageLoader, ImageLoadError, sniff_image_mime
from mindmemos.config import BlobStoreConfig
from mindmemos.infra.blob import LocalBlobStore
from mindmemos.typing import ImageRef

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 32


def _loader(tmp_path, **cfg) -> ImageLoader:
    return ImageLoader(BlobStoreConfig(**cfg), LocalBlobStore(tmp_path / "blobs"))


def test_sniff_recognizes_supported_formats_only() -> None:
    assert sniff_image_mime(PNG) == "image/png"
    assert sniff_image_mime(JPEG) == "image/jpeg"
    assert sniff_image_mime(b"RIFF\x00\x00\x00\x00WEBPVP8 ") == "image/webp"
    assert sniff_image_mime(b"GIF89a....") == "image/gif"
    assert sniff_image_mime(b"%PDF-1.7") is None
    assert sniff_image_mime(b"root:x:0:0:root") is None


@pytest.mark.asyncio
async def test_base64_and_data_uri(tmp_path) -> None:
    loader = _loader(tmp_path)
    raw = base64.b64encode(PNG).decode()
    assert (await loader.load(ImageRef(data=raw), namespace="p")).data == PNG
    loaded = await loader.load(ImageRef(data=f"data:image/png;base64,{raw}"), namespace="p")
    assert loaded.mime_type == "image/png"
    with pytest.raises(ImageLoadError):
        await loader.load(ImageRef(data="not base64!!"), namespace="p")
    with pytest.raises(ImageLoadError, match="unsupported image format"):
        await loader.load(ImageRef(data=base64.b64encode(b"hello world").decode()), namespace="p")


@pytest.mark.asyncio
async def test_paths_need_an_allowed_root(tmp_path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    (images / "a.png").write_bytes(PNG)
    secret = tmp_path / "secret.png"
    secret.write_bytes(PNG)

    with pytest.raises(ImageLoadError, match="disabled"):
        await _loader(tmp_path).load(ImageRef(path=str(images / "a.png")), namespace="p")
    loader = _loader(tmp_path, allowed_path_roots=[str(images)])
    assert (await loader.load(ImageRef(path=str(images / "a.png")), namespace="p")).data == PNG
    with pytest.raises(ImageLoadError, match="outside"):
        await loader.load(ImageRef(path=str(images / ".." / "secret.png")), namespace="p")


@pytest.mark.asyncio
async def test_size_limit_and_private_urls(tmp_path) -> None:
    loader = _loader(tmp_path, max_image_bytes=16)
    with pytest.raises(ImageLoadError, match="exceeds"):
        await loader.load(ImageRef(data=base64.b64encode(PNG).decode()), namespace="p")
    for url in ("http://127.0.0.1/x.png", "http://169.254.169.254/latest", "file:///etc/passwd"):
        with pytest.raises(ImageLoadError):
            await _loader(tmp_path).load(ImageRef(url=url), namespace="p")


@pytest.mark.asyncio
async def test_base64_is_rejected_before_decoding_when_oversized(tmp_path) -> None:
    # The check must fire on the encoded length: a payload far beyond the limit is refused
    # without ever being expanded in memory.
    loader = _loader(tmp_path, max_image_bytes=64)
    huge = base64.b64encode(PNG + b"\x00" * 1_000_000).decode()
    assert len(huge) > 1_300_000
    with pytest.raises(ImageLoadError, match="exceeds"):
        await loader.load(ImageRef(data=huge), namespace="p")
    with pytest.raises(ImageLoadError, match="exceeds"):  # data-URI form takes the same path
        await loader.load(ImageRef(data=f"data:image/png;base64,{huge}"), namespace="p")


@pytest.mark.asyncio
async def test_url_is_dialed_on_the_resolved_ip_with_original_host(monkeypatch, tmp_path) -> None:
    # DNS rebinding: the private-address check resolves the hostname once and the connection
    # must dial that same IP (Host header / TLS SNI keep the original hostname), so a hostname
    # that re-resolves to a private address for the actual connection cannot slip through.
    import httpx

    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["host"] = request.headers.get("host")
        seen["sni"] = request.extensions.get("sni_hostname")
        return httpx.Response(200, content=PNG)

    loader = ImageLoader(
        BlobStoreConfig(),
        LocalBlobStore(tmp_path / "blobs"),
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    async def fake_getaddrinfo(host, port, **kwargs):  # noqa: ARG001 - (host, port, ...) per getaddrinfo
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))]

    monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", fake_getaddrinfo)

    loaded = await loader.load(ImageRef(url="https://images.example.com:8443/pics/a.png"), namespace="p")

    assert loaded.data == PNG and loaded.source_url == "https://images.example.com:8443/pics/a.png"
    assert seen["url"] == "https://93.184.216.34:8443/pics/a.png"  # dialed on the pinned IP
    assert seen["host"] == "images.example.com:8443"
    assert seen["sni"] == "images.example.com"


@pytest.mark.asyncio
async def test_url_with_private_hosts_allowed_dials_the_hostname(monkeypatch, tmp_path) -> None:
    import httpx

    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["host"] = request.headers.get("host")
        return httpx.Response(200, content=PNG)

    loader = ImageLoader(
        BlobStoreConfig(url_allow_private_hosts=True),
        LocalBlobStore(tmp_path / "blobs"),
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    async def fail_getaddrinfo(*args, **kwargs):  # nothing may resolve: private hosts are allowed
        raise AssertionError("getaddrinfo must not be called when private hosts are allowed")

    monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", fail_getaddrinfo)

    loaded = await loader.load(ImageRef(url="http://localhost:9910/a.png"), namespace="p")

    assert loaded.data == PNG
    assert seen["url"] == "http://localhost:9910/a.png"
    assert seen["host"] == "localhost:9910"  # no override: httpx derives Host from the URL itself


@pytest.mark.asyncio
async def test_blob_ids_resolve_within_the_project_only(tmp_path) -> None:
    loader = _loader(tmp_path)
    blob = await LocalBlobStore(tmp_path / "blobs").put("p", PNG, mime_type="image/png")
    assert (await loader.load(ImageRef(blob_id=blob.blob_id), namespace="p")).data == PNG
    with pytest.raises(ImageLoadError, match="unknown blob_id"):
        await loader.load(ImageRef(blob_id=blob.blob_id), namespace="other")


def test_image_ref_needs_exactly_one_source() -> None:
    with pytest.raises(ValueError):
        ImageRef()
    with pytest.raises(ValueError):
        ImageRef(path="/a.png", url="http://x/a.png")
