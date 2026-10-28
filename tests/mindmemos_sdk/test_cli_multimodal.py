"""Tests for the multimodal CLI wiring: ``--image``, ``--query-image``/``--multimodal``,
and ``memory get-image`` — argument parsing, local file inlining, and error paths,
all against a fake client (no network, no local SDK config)."""

from __future__ import annotations

import base64
import json

import pytest
from mindmemos_sdk.errors import ApiError
from mindmemos_sdk.memory import AddResult, MemorySearchHit, SearchResult

from mindmemos_sdk import cli

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"fake-png-body"
PNG_B64 = base64.b64encode(PNG_BYTES).decode("ascii")
BLOB_ID = "a" * 64


class _FakeMemory:
    def __init__(self, result: object) -> None:
        self._result = result
        self.calls: list[tuple[str, tuple, dict]] = []

    def _record(self, method: str, *args, **kwargs):
        self.calls.append((method, args, kwargs))
        return self._result

    def add(self, *args, **kwargs):
        return self._record("add", *args, **kwargs)

    def search(self, *args, **kwargs):
        return self._record("search", *args, **kwargs)

    def get_image(self, *args, **kwargs):
        result = self._record("get_image", *args, **kwargs)
        if isinstance(result, Exception):
            raise result  # canned error results simulate server failures
        return result


class _FakeClient:
    def __init__(self, result: object) -> None:
        self.memory = _FakeMemory(result)

    def __enter__(self) -> _FakeClient:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


@pytest.fixture
def fake_client(monkeypatch):
    holder: dict[str, _FakeClient] = {}

    def factory(result: object) -> _FakeClient:
        client = _FakeClient(result)
        holder["client"] = client
        monkeypatch.setattr(cli, "_build_client", lambda: client)
        return client

    factory.holder = holder  # type: ignore[attr-defined]
    return factory


def _run(argv: list[str]) -> int:
    return cli.main(argv)


def _last_kwargs(fake_client) -> dict:
    _name, _args, kwargs = fake_client.holder["client"].memory.calls[0]
    return kwargs


# ---- memory add --image ----


def test_memory_add_image_inlines_local_file_as_base64(fake_client, tmp_path):
    image = tmp_path / "photo.png"
    image.write_bytes(PNG_BYTES)
    fake_client(AddResult(code="ok"))

    rc = _run(["memory", "add", "--content", "look at this", "--image", str(image)])

    assert rc == 0
    (message,) = _last_kwargs(fake_client)["messages"]
    (ref,) = message.images
    assert ref.data == PNG_B64 and ref.mime_type == "image/png"
    assert ref.url is None and ref.path is None and ref.blob_id is None


def test_memory_add_image_accepts_urls_among_files(fake_client, tmp_path):
    image = tmp_path / "photo.png"
    image.write_bytes(PNG_BYTES)
    fake_client(AddResult(code="ok"))

    rc = _run(
        [
            "memory",
            "add",
            "--content",
            "mixed sources",
            "--image",
            "https://example.com/cat.jpg",
            "--image",
            str(image),
        ]
    )

    assert rc == 0
    (message,) = _last_kwargs(fake_client)["messages"]
    url_ref, data_ref = message.images
    assert url_ref.url == "https://example.com/cat.jpg" and url_ref.data is None
    assert data_ref.data == PNG_B64 and data_ref.mime_type == "image/png"


def test_memory_add_image_rejects_messages_json_combination(fake_client, capsys):
    rc = _run(
        [
            "memory",
            "add",
            "--messages-json",
            '[{"role": "user", "content": "x"}]',
            "--image",
            "https://example.com/cat.jpg",
        ]
    )

    assert rc == 2
    assert "embed images in --messages-json" in capsys.readouterr().out
    assert not fake_client.holder  # failed validation never reaches the client


def test_memory_add_image_missing_file_is_a_usage_error(capsys, tmp_path):
    rc = _run(["memory", "add", "--content", "x", "--image", str(tmp_path / "missing.png")])

    assert rc == 2
    assert "failed to read" in capsys.readouterr().out


def test_memory_add_image_unrecognized_format_is_a_usage_error(capsys, tmp_path):
    mystery = tmp_path / "mystery.bin"
    mystery.write_bytes(b"definitely not an image")

    rc = _run(["memory", "add", "--content", "x", "--image", str(mystery)])

    assert rc == 2
    assert "unrecognized image format" in capsys.readouterr().out


# ---- memory search --query-image / --multimodal ----


def test_memory_search_passes_query_images_and_multimodal(fake_client, tmp_path):
    image = tmp_path / "query.png"
    image.write_bytes(PNG_BYTES)
    fake_client(SearchResult(memories=[MemorySearchHit(id="m1", memory="a cat photo")]))

    rc = _run(
        [
            "memory",
            "search",
            "cats",
            "--query-image",
            str(image),
            "--query-image",
            "http://example.com/cat.jpg",
            "--multimodal",
        ]
    )

    assert rc == 0
    kwargs = _last_kwargs(fake_client)
    file_ref, url_ref = kwargs["query_images"]
    assert file_ref.data == PNG_B64 and file_ref.mime_type == "image/png"
    assert url_ref.url == "http://example.com/cat.jpg"
    assert kwargs["multimodal"] is True


def test_memory_search_text_only_kwargs_unchanged(fake_client):
    fake_client(SearchResult(memories=[]))

    rc = _run(["memory", "search", "tea"])

    assert rc == 0
    kwargs = _last_kwargs(fake_client)
    assert "query_images" not in kwargs and "multimodal" not in kwargs


def test_memory_search_query_image_missing_file_is_a_usage_error(capsys, tmp_path):
    rc = _run(["memory", "search", "cats", "--query-image", str(tmp_path / "missing.png")])

    assert rc == 2
    assert "failed to read" in capsys.readouterr().out


# ---- memory search image_url rendering ----


def test_memory_search_renders_image_url_line(fake_client, capsys):
    url = "https://minio.example.com/blobs/p1/aa/aabb?X-Amz-Signature=fake"
    fake_client(
        SearchResult(
            memories=[
                MemorySearchHit(id="m1", memory="a cat photo", metadata={"blob_id": BLOB_ID, "image_url": url}),
                MemorySearchHit(id="m2", memory="a text fact"),
            ]
        )
    )

    rc = _run(["memory", "search", "cats"])

    assert rc == 0
    out = capsys.readouterr().out
    assert f"image: {url}" in out  # the presigned link rides along the hit it belongs to
    assert out.count("image:") == 1  # the text-only hit renders no image line


def test_memory_search_json_includes_image_url_metadata(fake_client, capsys):
    url = f"http://testserver/v1/memory/images/{BLOB_ID}?user_id=user-1"
    fake_client(SearchResult(memories=[MemorySearchHit(id="m1", memory="photo", metadata={"image_url": url})]))

    rc = _run(["memory", "search", "cats", "--json"])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["memories"][0]["metadata"]["image_url"] == url


# ---- memory get-image ----


def test_memory_get_image_writes_default_named_file(fake_client, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    fake_client((PNG_BYTES, "image/png"))

    rc = _run(["memory", "get-image", BLOB_ID, "--user-id", "user-1"])

    assert rc == 0
    assert (tmp_path / f"{BLOB_ID}.png").read_bytes() == PNG_BYTES
    assert f"Saved {len(PNG_BYTES)} bytes" in capsys.readouterr().out
    assert fake_client.holder["client"].memory.calls[0][1] == (BLOB_ID, "user-1")


def test_memory_get_image_honors_output_path_with_nested_dirs(fake_client, tmp_path):
    fake_client((PNG_BYTES, "image/png"))
    output = tmp_path / "nested" / "dir" / "out.png"

    rc = _run(["memory", "get-image", BLOB_ID, "--user-id", "user-1", "-o", str(output)])

    assert rc == 0
    assert output.read_bytes() == PNG_BYTES


def test_memory_get_image_requires_user_id(fake_client):
    with pytest.raises(SystemExit) as excinfo:
        _run(["memory", "get-image", BLOB_ID])

    assert excinfo.value.code == 2  # argparse rejects the missing --user-id


def test_memory_get_image_reports_api_errors(fake_client, capsys):
    fake_client(ApiError("image not found", code="image.not_found", request_id="req-1"))

    rc = _run(["memory", "get-image", BLOB_ID, "--user-id", "user-1"])

    assert rc == 1
    out = capsys.readouterr().out
    assert "image not found" in out
    assert "request_id=req-1" in out
