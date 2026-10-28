"""Startup multimodal check: a mode beyond off without mm embedding warns, off stays INFO."""

import asyncio

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
