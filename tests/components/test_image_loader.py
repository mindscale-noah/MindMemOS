import asyncio
import base64
import socket
from pathlib import Path

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

    assert loaded.data == PNG and loaded.mime_type == "image/png"
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


def test_owned_http_client_is_one_per_event_loop() -> None:
    # Fetches of one request reuse their connections; a client binds to the loop it first ran
    # on, so a fresh loop (every test, or a restarted server) must get its own client.
    loader = _loader(Path("nowhere"))

    async def twice() -> tuple[object, object]:
        return loader._client_for(), loader._client_for()

    loop = asyncio.new_event_loop()
    try:
        first, second = loop.run_until_complete(twice())
    finally:
        loop.close()
    assert first is second  # one shared pool within the loop

    def client_in_fresh_loop() -> object:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(twice())[0]
        finally:
            loop.close()

    assert client_in_fresh_loop() is not first  # never shared across loops


@pytest.mark.asyncio
async def test_out_of_range_url_port_is_a_400_not_a_500(tmp_path) -> None:
    # urlsplit defers port parsing to first access: an out-of-range port raises ValueError
    # there, not an httpx error — it must surface as the image.* 400 envelope like any
    # other invalid image input, not escape as a 500.
    loader = _loader(tmp_path)
    for url in ("http://example.com:70000/a.png", "https://h:99999/x"):
        with pytest.raises(ImageLoadError, match="invalid port") as error:
            await loader.load(ImageRef(url=url), namespace="p")
        assert error.value.code == "image.invalid"


@pytest.mark.asyncio
async def test_malformed_urls_and_hostnames_are_400s_not_500s(monkeypatch, tmp_path) -> None:
    loader = _loader(tmp_path)

    # An unterminated IPv6 literal makes urlsplit itself raise ValueError.
    with pytest.raises(ImageLoadError, match="not valid") as broken:
        await loader.load(ImageRef(url="http://[::1/x.png"), namespace="p")
    assert broken.value.code == "image.invalid"

    # A non-ASCII hostname makes getaddrinfo raise UnicodeError before any lookup.
    async def unicode_getaddrinfo(host, port, **kwargs):  # noqa: ARG001 - signature per getaddrinfo
        raise UnicodeEncodeError("idna", "café", 0, 1, "not an ASCII hostname")

    monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", unicode_getaddrinfo)
    with pytest.raises(ImageLoadError, match="does not resolve") as unresolvable:
        await loader.load(ImageRef(url="http://café.example/x.png"), namespace="p")
    assert unresolvable.value.code == "image.fetch_failed"


@pytest.mark.asyncio
async def test_url_follows_one_redirect_and_stops_loops(tmp_path) -> None:
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/hop":
            return httpx.Response(302, headers={"location": "/final.png"})
        if request.url.path == "/loop":
            return httpx.Response(302, headers={"location": "/loop"})
        return httpx.Response(200, content=PNG)

    loader = ImageLoader(
        BlobStoreConfig(url_allow_private_hosts=True),
        LocalBlobStore(tmp_path / "blobs"),
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    loaded = await loader.load(ImageRef(url="http://images.test/hop"), namespace="p")
    assert loaded.data == PNG  # followed the single redirect

    with pytest.raises(ImageLoadError, match="too many redirects") as loop_error:
        await loader.load(ImageRef(url="http://images.test/loop"), namespace="p")
    assert loop_error.value.code == "image.fetch_failed"


@pytest.mark.asyncio
async def test_url_fetch_failures_are_reported_as_image_errors(tmp_path) -> None:
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/missing":
            return httpx.Response(404)
        raise httpx.ConnectError("refused", request=request)

    loader = ImageLoader(
        BlobStoreConfig(url_allow_private_hosts=True),
        LocalBlobStore(tmp_path / "blobs"),
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(ImageLoadError, match="HTTP 404") as missing:
        await loader.load(ImageRef(url="http://images.test/missing"), namespace="p")
    assert missing.value.code == "image.fetch_failed"
    with pytest.raises(ImageLoadError, match="fetch failed") as unreachable:
        await loader.load(ImageRef(url="http://images.test/down"), namespace="p")
    assert unreachable.value.code == "image.fetch_failed"
