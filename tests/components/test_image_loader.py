import base64

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
