from __future__ import annotations

import importlib.util
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest


def _load_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "mm_embed_server.py"
    spec = importlib.util.spec_from_file_location("mm_embed_server", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class SlowEncoder:
    """Stands in for Siglip2 (make_handler only needs dimensions + the two embed methods)."""

    dimensions = 3

    def __init__(self) -> None:
        self.calls = 0
        self.done = threading.Event()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        self.done.wait(timeout=10)
        return [[0.0, 0.0, 1.0] for _ in texts]

    def embed_images(self, images: list) -> list[list[float]]:
        raise AssertionError("no image input in this test")


@pytest.fixture()
def server():
    module = _load_module()
    encoder = SlowEncoder()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), module.make_handler(encoder, "siglip2-test", max_concurrent=1))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield module, encoder, f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def _post(base: str, payload: dict) -> tuple[int, dict, dict]:
    request = urllib.request.Request(
        f"{base}/v1/embeddings", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as resp:
            return resp.status, json.loads(resp.read()), dict(resp.headers)
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read()), dict(exc.headers)


def test_overload_returns_503_with_retry_after(server) -> None:
    # max_concurrent=1 with a blocked encoder: the first request holds the slot, the second
    # must be rejected immediately as overload (503 + Retry-After), not queued behind it.
    module, encoder, base = server
    result: dict[str, tuple[int, dict, dict]] = {}
    first = threading.Thread(target=lambda: result.setdefault("first", _post(base, {"input": ["a"], "model": "m"})))
    first.start()
    deadline = time.monotonic() + 5
    while encoder.calls == 0 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert encoder.calls == 1, "first request never reached the encoder"

    status, body, headers = _post(base, {"input": ["b"], "model": "m"})
    assert status == 503
    assert body["error"]["type"] == "overloaded"
    assert "already running" in body["error"]["message"]
    assert headers.get("Retry-After") == "2"

    encoder.done.set()
    first.join(timeout=10)
    status, body, _ = result["first"]
    assert status == 200
    assert body["object"] == "list"
    assert body["data"][0]["embedding"] == [0.0, 0.0, 1.0]


def test_slot_is_released_and_health_reports_the_model(server) -> None:
    # After the blocked request finishes, the slot is free again: the next request is served.
    module, encoder, base = server
    encoder.done.set()
    status, body, _ = _post(base, {"input": ["hello"]})
    assert status == 200
    assert body["data"][0]["index"] == 0

    with urllib.request.urlopen(f"{base}/health", timeout=10) as resp:
        assert resp.status == 200
        assert json.loads(resp.read()) == {"status": "ok", "model": "siglip2-test", "dimensions": 3}


def test_bad_input_shapes_are_400s(server) -> None:
    _, encoder, base = server
    encoder.done.set()
    status, body, _ = _post(base, {"input": ["a"], "dimensions": 999})
    assert status == 400
    assert body["error"]["message"] == "dimensions must be 3"

    status, body, _ = _post(base, {"input": [{"video": "x"}]})
    assert status == 400
    assert "input items must be" in body["error"]["message"]


def test_unknown_paths_are_404(server) -> None:
    _, _, base = server
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        urllib.request.urlopen(f"{base}/nope", timeout=10)
    assert excinfo.value.code == 404
