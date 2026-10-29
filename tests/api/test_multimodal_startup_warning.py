"""Startup multimodal check: a mode beyond off without mm embedding warns, off stays INFO."""

import asyncio

import pytest
from mindmemos.api import app as app_module
from mindmemos.config import MemoryConfig, build
from structlog.testing import capture_logs


def _run_check(monkeypatch, mode: str) -> None:
    cfg = build(MemoryConfig, {"algo_config": {"multimodal": {"mode": mode}}})
    monkeypatch.setattr(app_module, "get_config", lambda: cfg)
    monkeypatch.setattr(app_module, "get_mm_embed_client", lambda: None)
    asyncio.run(app_module._check_multimodal_embedding())


def test_basic_mode_without_mm_endpoints_warns(monkeypatch) -> None:
    with capture_logs() as logs:
        _run_check(monkeypatch, "basic")
    warnings = [event for event in logs if event.get("log_level") == "warning"]
    assert len(warnings) == 1
    assert "no mm_embed_model_router endpoints" in warnings[0]["event"]
    assert warnings[0]["mode"] == "basic"


def test_off_mode_without_mm_endpoints_stays_info(monkeypatch) -> None:
    with capture_logs() as logs:
        _run_check(monkeypatch, "off")
    assert not [event for event in logs if event.get("log_level") in ("warning", "error")]
    assert any("stores images without indexing" in event["event"] for event in logs)


def test_s3_backend_without_the_extra_fails_the_boot(monkeypatch) -> None:
    # The S3 backend imports minio lazily; without the mindmemos[s3] extra that ImportError
    # must fail startup, not surface as a 500 on the first image request.
    from mindmemos.errors import InvalidConfigError

    cfg = build(MemoryConfig, {"blob_store": {"backend": "s3"}})

    def missing_extra():
        raise ImportError("No module named 'minio'")

    monkeypatch.setattr(app_module, "get_config", lambda: cfg)
    monkeypatch.setattr(app_module, "get_blob_store", missing_extra)
    with pytest.raises(InvalidConfigError, match=r"mindmemos\[s3\]") as excinfo:
        app_module._check_blob_store()
    assert excinfo.value.field == "blob_store.backend"


def test_s3_backend_with_an_endpoint_the_client_rejects_fails_the_boot(monkeypatch) -> None:
    # The minio constructor validates more than config validation does (e.g. a non-numeric
    # port raises a bare ValueError); the boot must name the config field, not crash raw.
    from mindmemos.errors import InvalidConfigError

    cfg = build(MemoryConfig, {"blob_store": {"backend": "s3", "s3_endpoint": "minio.internal:70003"}})

    def bad_endpoint():
        raise ValueError("Port could not be cast to integer value as '70003'")

    monkeypatch.setattr(app_module, "get_config", lambda: cfg)
    monkeypatch.setattr(app_module, "get_blob_store", bad_endpoint)
    with pytest.raises(InvalidConfigError) as excinfo:
        app_module._check_blob_store()
    assert excinfo.value.field == "blob_store.s3_endpoint"


def test_local_backend_and_a_working_s3_extra_boot_normally(monkeypatch) -> None:
    calls = []

    monkeypatch.setattr(app_module, "get_config", lambda: build(MemoryConfig))
    monkeypatch.setattr(app_module, "get_blob_store", lambda: (calls.append(1), object())[1])
    app_module._check_blob_store()  # local backend: the store is never constructed
    assert calls == []

    monkeypatch.setattr(app_module, "get_config", lambda: build(MemoryConfig, {"blob_store": {"backend": "s3"}}))
    app_module._check_blob_store()  # s3 backend with the extra importable: green boot
    assert calls == [1]


def test_mm_embed_probe_timeout_warns_instead_of_stalling_the_boot(monkeypatch) -> None:
    # The probe goes out under the router's full retry policy; a dead endpoint would stall
    # boot for minutes. It must give up at the bound and continue like an unreachable one.

    class HangingClient:
        dimensions, model = 3, "m"

        async def embed_texts(self, texts, *, task=""):
            await asyncio.sleep(30)

    cfg = build(MemoryConfig)
    monkeypatch.setattr(app_module, "get_config", lambda: cfg)
    monkeypatch.setattr(app_module, "get_mm_embed_client", lambda: HangingClient())
    monkeypatch.setattr(app_module, "_MM_EMBED_PROBE_TIMEOUT_SECONDS", 0.05)

    class NoCollection:
        async def collection_vector_size(self):
            return None

    monkeypatch.setattr(app_module, "ImageMemoryStore", NoCollection)
    with capture_logs() as logs:
        asyncio.run(app_module._check_multimodal_embedding())
    assert any("timed out" in event["event"] for event in logs if event.get("log_level") == "warning")
