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


def test_s3_store_without_the_extra_fails_the_boot(monkeypatch) -> None:
    # The S3 store imports minio lazily; without the mindmemos[s3] extra that ImportError
    # must fail startup of an image-capable deployment, not surface as a 500 on the first
    # image request.
    from mindmemos.errors import InvalidConfigError

    cfg = build(MemoryConfig, {"algo_config": {"multimodal": {"mode": "basic"}}})

    def missing_extra():
        raise ImportError("No module named 'minio'")

    monkeypatch.setattr(app_module, "get_config", lambda: cfg)
    monkeypatch.setattr(app_module, "get_blob_store", missing_extra)
    with pytest.raises(InvalidConfigError, match=r"mindmemos\[s3\]") as excinfo:
        app_module._check_blob_store()
    assert excinfo.value.field == "blob_store.s3_endpoint"


def test_s3_endpoint_the_client_rejects_fails_the_boot(monkeypatch) -> None:
    # The minio constructor validates more than config validation does (e.g. a non-numeric
    # port raises a bare ValueError); the boot must name the config field, not crash raw.
    from mindmemos.errors import InvalidConfigError

    cfg = build(
        MemoryConfig,
        {
            "algo_config": {"multimodal": {"mode": "basic"}},
            "blob_store": {"s3_endpoint": "minio.internal:70003"},
        },
    )

    def bad_endpoint():
        raise ValueError("Port could not be cast to integer value as '70003'")

    monkeypatch.setattr(app_module, "get_config", lambda: cfg)
    monkeypatch.setattr(app_module, "get_blob_store", bad_endpoint)
    with pytest.raises(InvalidConfigError) as excinfo:
        app_module._check_blob_store()
    assert excinfo.value.field == "blob_store.s3_endpoint"


def test_a_working_s3_extra_boots_normally(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(app_module, "get_blob_store", lambda: (calls.append(1), object())[1])
    monkeypatch.setattr(
        app_module, "get_config", lambda: build(MemoryConfig, {"algo_config": {"multimodal": {"mode": "basic"}}})
    )
    app_module._check_blob_store()  # the store is built once on every image-capable boot
    assert calls == [1]


def test_off_mode_boots_without_the_s3_extra(monkeypatch) -> None:
    # A text-only deployment (multimodal off) must boot without the s3 extra and
    # without ever constructing the blob store — validation already exempts mode
    # "off" from every blob_store field, so the boot check must not contradict it.
    calls = []

    def missing_extra():
        calls.append(1)
        raise ImportError("No module named 'minio'")

    monkeypatch.setattr(app_module, "get_config", lambda: build(MemoryConfig))  # mode defaults to off
    monkeypatch.setattr(app_module, "get_blob_store", missing_extra)
    with capture_logs() as logs:
        app_module._check_blob_store()
    assert calls == []  # never constructed
    assert not [event for event in logs if event.get("log_level") in ("warning", "error")]
    assert any("blob store not constructed" in event["event"] for event in logs)


def test_multimodal_without_a_signing_secret_warns_off_mode_stays_quiet(monkeypatch) -> None:
    # with no secret the signed-URL tier is disabled, and the boot must say so instead of
    # letting search mint empty-key (forgeable) links; text-only (mode off) never warns.
    monkeypatch.setattr(app_module, "get_blob_store", lambda: object())

    cfg = build(MemoryConfig, {"algo_config": {"multimodal": {"mode": "basic"}}})
    monkeypatch.setattr(app_module, "get_config", lambda: cfg)
    with capture_logs() as logs:
        app_module._check_blob_store()
    assert any("image url signing disabled" in event["event"] for event in logs if event.get("log_level") == "warning")

    cfg = build(MemoryConfig)  # mode defaults to off
    monkeypatch.setattr(app_module, "get_config", lambda: cfg)
    with capture_logs() as logs:
        app_module._check_blob_store()
    assert not [event for event in logs if event.get("log_level") in ("warning", "error")]


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
