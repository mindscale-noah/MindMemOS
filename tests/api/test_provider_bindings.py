from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
from mindmemos.errors import BadRequestError
from mindmemos.provider_bindings import (
    ProviderBindingRecord,
    ProviderBindingResolver,
    ProviderBindingScope,
    validate_provider_binding_patch,
)
from mindmemos.typing import MemoryRequestContext


def make_context(**overrides) -> MemoryRequestContext:
    data = {
        "request_id": "00000000-0000-0000-0000-000000000001",
        "account_id": "acc-1",
        "project_id": "proj-1",
        "api_key_uuid": "key-1",
        "user_id": "user-1",
        "app_id": "app-1",
        "session_id": "session-1",
        "agent_id": "agent-1",
    }
    data.update(overrides)
    return MemoryRequestContext(**data)


def router_config(*, embed_model: str = "openai/text-embedding-3-large", dimensions: int = 1024) -> dict[str, Any]:
    return {
        "chat_model_router": {
            "routing_strategy": "simple-shuffle",
            "endpoints": [
                {
                    "model": "openai/gpt-4.1-mini",
                    "api_key": "chat-key",
                    "api_base": "https://chat.example/v1",
                    "timeout": 60,
                }
            ],
        },
        "embed_model_router": {
            "endpoints": [
                {
                    "model": embed_model,
                    "api_key": "embed-key",
                    "api_base": "https://embed.example/v1",
                    "dimensions": dimensions,
                    "timeout": 60,
                }
            ],
        },
        "rerank_model_router": {
            "endpoints": [
                {
                    "model": "cohere/rerank",
                    "api_key": "rerank-key",
                    "api_base": "https://rerank.example/v1",
                }
            ],
        },
    }


@dataclass
class FakeStore:
    records: list[ProviderBindingRecord]
    calls: list[str] = field(default_factory=list)

    async def list_project_bindings(self, project_id: str) -> list[ProviderBindingRecord]:
        self.calls.append(project_id)
        return self.records


@pytest.mark.asyncio
async def test_provider_binding_resolver_disabled_does_not_read_store() -> None:
    store = FakeStore(
        [
            ProviderBindingRecord(
                binding_id="binding-project",
                project_id="proj-1",
                scope=ProviderBindingScope(),
                routers=router_config(),
            )
        ]
    )
    resolver = ProviderBindingResolver(store=store, enabled=False)

    result = await resolver.resolve(make_context())

    assert result is None
    assert store.calls == []


@pytest.mark.asyncio
async def test_provider_binding_resolver_picks_most_specific_matching_scope() -> None:
    project_only = ProviderBindingRecord(
        binding_id="binding-project",
        project_id="proj-1",
        scope=ProviderBindingScope(),
        routers={"chat_model_router": {"endpoints": [{"model": "project"}]}},
    )
    user_only = ProviderBindingRecord(
        binding_id="binding-user",
        project_id="proj-1",
        scope=ProviderBindingScope(user_id="user-1"),
        routers={"chat_model_router": {"endpoints": [{"model": "user"}]}},
    )
    user_session = ProviderBindingRecord(
        binding_id="binding-user-session",
        project_id="proj-1",
        scope=ProviderBindingScope(user_id="user-1", session_id="session-1"),
        routers={"chat_model_router": {"endpoints": [{"model": "user-session"}]}},
    )
    other_session = ProviderBindingRecord(
        binding_id="binding-other-session",
        project_id="proj-1",
        scope=ProviderBindingScope(user_id="user-1", session_id="other"),
        routers={"chat_model_router": {"endpoints": [{"model": "other"}]}},
    )
    resolver = ProviderBindingResolver(
        store=FakeStore([project_only, user_only, user_session, other_session]),
        enabled=True,
    )

    result = await resolver.resolve(make_context())

    assert result == user_session.routers


def test_provider_binding_patch_allows_runtime_config_changes() -> None:
    old = router_config()
    new = router_config()
    new["chat_model_router"]["endpoints"][0]["model"] = "openai/gpt-4.1"
    new["chat_model_router"]["endpoints"][0]["timeout"] = 120
    new["embed_model_router"]["endpoints"][0]["api_key"] = "rotated"
    new["embed_model_router"]["endpoints"][0]["api_base"] = "https://new-embed.example/v1"
    new["embed_model_router"]["routing_strategy"] = "least-busy"

    validate_provider_binding_patch(
        old,
        new,
        project_id="proj-1",
        binding_id="binding-1",
        scope=ProviderBindingScope(user_id="user-1"),
        request_id="req-1",
    )


def test_provider_binding_patch_rejects_embedding_model_and_dimensions_changes(monkeypatch) -> None:
    warnings: list[dict[str, Any]] = []

    def fake_warning(event: str, **kwargs) -> None:
        warnings.append({"event": event, **kwargs})

    monkeypatch.setattr("mindmemos.provider_bindings.logger.warning", fake_warning)
    old = router_config()
    new = router_config(embed_model="openai/text-embedding-3-small", dimensions=512)

    with pytest.raises(BadRequestError) as exc_info:
        validate_provider_binding_patch(
            old,
            new,
            project_id="proj-1",
            binding_id="binding-1",
            scope=ProviderBindingScope(user_id="user-1"),
            request_id="req-1",
        )

    assert exc_info.value.code == "provider_binding.immutable_embedding_config"
    assert "embed_model_router.endpoints[0].model" in exc_info.value.message
    assert "embed_model_router.endpoints[0].dimensions" in exc_info.value.message
    assert warnings == [
        {
            "event": "provider_binding_immutable_update_blocked",
            "project_id": "proj-1",
            "binding_id": "binding-1",
            "scope": {"user_id": "user-1"},
            "blocked_fields": [
                "embed_model_router.endpoints[0].model",
                "embed_model_router.endpoints[0].dimensions",
            ],
            "request_id": "req-1",
        }
    ]
