from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest
from mindmemos.provider_bindings import (
    ProviderBindingRecord,
    ProviderBindingScope,
    QdrantProviderBindingStore,
    _config_hash,
)


class FakeQdrant:
    def __init__(self) -> None:
        self.existing = None
        self.upserts = []

    async def get_provider_binding(self, project_id: str, binding_id: str):
        del project_id, binding_id
        return self.existing

    async def upsert_provider_binding(self, point) -> None:
        self.upserts.append(point)
        self.existing = SimpleNamespace(payload=deepcopy(point.payload))


def _record(*, model: str = "openai/gpt-a") -> ProviderBindingRecord:
    return ProviderBindingRecord(
        binding_id="binding-1",
        project_id="project-1",
        scope=ProviderBindingScope(user_id="user-1"),
        routers={
            "chat_model_router": {
                "endpoints": [
                    {
                        "model": model,
                        "api_key": "EMPTY",
                        "api_base": "http://backend:8010/litellm_memory_proxy/{userId}/v1",
                        "transport": "platform_gateway",
                    }
                ]
            }
        },
    )


@pytest.mark.asyncio
async def test_store_skips_upsert_when_full_binding_content_is_unchanged() -> None:
    qdrant = FakeQdrant()
    store = QdrantProviderBindingStore(clients=SimpleNamespace(qdrant=qdrant))

    await store.upsert(_record())
    original_payload = deepcopy(qdrant.existing.payload)
    await store.upsert(_record())

    assert len(qdrant.upserts) == 1
    assert qdrant.existing.payload == original_payload


@pytest.mark.asyncio
async def test_store_upserts_when_binding_content_changes() -> None:
    qdrant = FakeQdrant()
    store = QdrantProviderBindingStore(clients=SimpleNamespace(qdrant=qdrant))

    await store.upsert(_record(model="openai/gpt-a"))
    await store.upsert(_record(model="openai/gpt-b"))

    assert len(qdrant.upserts) == 2
    assert qdrant.existing.payload["routers"]["chat_model_router"]["endpoints"][0]["model"] == "openai/gpt-b"


@pytest.mark.asyncio
async def test_store_rewrites_legacy_router_only_hash_once() -> None:
    qdrant = FakeQdrant()
    store = QdrantProviderBindingStore(clients=SimpleNamespace(qdrant=qdrant))
    record = _record()

    await store.upsert(record)
    qdrant.existing.payload["config_hash"] = _config_hash(record.routers)
    await store.upsert(record)

    assert len(qdrant.upserts) == 2
    assert qdrant.existing.payload["config_hash"] != _config_hash(record.routers)
