"""HTTP contracts for strict memory update and delete mutations."""

from __future__ import annotations

from contextlib import nullcontext

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mindmemos.api.app import register_exception_handlers
from mindmemos.api.deps import get_request_context
from mindmemos.api.routes import router
from mindmemos.api.schemas import AuthContext
from mindmemos.api.services import get_memory_service
from mindmemos.api.services.memory_service import MemoryService
from mindmemos.errors import MemoryNotFoundError


class _MissingDeletePipeline:
    async def delete(self, inp, context):
        raise MemoryNotFoundError(inp.id)


class _MissingUpdatePipeline:
    async def update(self, inp, context):
        raise MemoryNotFoundError(inp.id)


@pytest.fixture
def auth_context() -> AuthContext:
    return AuthContext(
        request_id="req-memory-contract",
        account_id="account-1",
        project_id="project-1",
        api_key_uuid="key-1",
        memory_algorithm="vanilla",
        scopes=["memory:read", "memory:write"],
    )


def _client(monkeypatch: pytest.MonkeyPatch, auth_context: AuthContext, service: MemoryService) -> TestClient:
    async def _no_provider_binding(_context):
        return nullcontext()

    monkeypatch.setattr(service, "_provider_config_context", _no_provider_binding)
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(router)
    app.dependency_overrides[get_request_context] = lambda: auth_context
    app.dependency_overrides[get_memory_service] = lambda: service
    return TestClient(app)


def test_delete_rejects_hard_flag_over_http(monkeypatch: pytest.MonkeyPatch, auth_context: AuthContext) -> None:
    service = MemoryService(delete_pipeline=_MissingDeletePipeline(), skill_store=object())

    response = _client(monkeypatch, auth_context, service).post(
        "/v1/memory/delete",
        json={"memory_id": "missing", "hard": True},
    )

    assert response.status_code == 422
    assert response.json()["code"] == "invalid_request"


def test_update_rejects_delete_status_over_http(monkeypatch: pytest.MonkeyPatch, auth_context: AuthContext) -> None:
    service = MemoryService(update_pipeline=_MissingUpdatePipeline(), skill_store=object())

    response = _client(monkeypatch, auth_context, service).post(
        "/v1/memory/update",
        json={"memory_id": "missing", "status": "delete"},
    )

    assert response.status_code == 422
    assert response.json()["code"] == "invalid_request"


def test_delete_missing_memory_returns_standard_not_found(
    monkeypatch: pytest.MonkeyPatch, auth_context: AuthContext
) -> None:
    service = MemoryService(delete_pipeline=_MissingDeletePipeline(), skill_store=object())

    response = _client(monkeypatch, auth_context, service).post(
        "/v1/memory/delete",
        json={"memory_id": "missing"},
    )

    assert response.status_code == 404
    assert response.json() == {
        "code": "memory.not_found",
        "message": "memory not found: missing",
        "data": None,
    }


def test_update_missing_memory_returns_standard_not_found(
    monkeypatch: pytest.MonkeyPatch, auth_context: AuthContext
) -> None:
    service = MemoryService(update_pipeline=_MissingUpdatePipeline(), skill_store=object())

    response = _client(monkeypatch, auth_context, service).post(
        "/v1/memory/update",
        json={"memory_id": "missing", "content": "new content"},
    )

    assert response.status_code == 404
    assert response.json() == {
        "code": "memory.not_found",
        "message": "memory not found: missing",
        "data": None,
    }
