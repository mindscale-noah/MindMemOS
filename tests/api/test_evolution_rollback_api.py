"""Tests for the evolution rollback API service method."""

from __future__ import annotations

import pytest

from mindmemos.api.schemas import AuthContext, EvolutionRollbackRequest
from mindmemos.api.services import memory_service as ms_module
from mindmemos.config import init_config
from mindmemos.typing import EvolutionResult


def _auth() -> AuthContext:
    return AuthContext(
        request_id="00000000-0000-0000-0000-000000000001",
        account_id="acc-1",
        project_id="proj-1",
        api_key_uuid="key-1",
        memory_algorithm="feedback_evo",
    )


@pytest.mark.asyncio
async def test_rollback_evolution_returns_ok(monkeypatch):
    init_config(config_path="config/mindmemos/dev.example.yaml")

    async def _fake_rollback(project_id: str, version: int) -> EvolutionResult:
        del project_id
        return EvolutionResult(
            project_id="proj-1",
            version=version,
            is_rollback=True,
            changes=[],
        )

    monkeypatch.setattr(
        ms_module.EvolutionStateStore,
        "rollback",
        staticmethod(_fake_rollback),
    )
    service = ms_module.get_memory_service()
    result = await service.rollback_evolution(_auth(), EvolutionRollbackRequest(version=2))

    assert result.status == "ok"
    assert result.version == 2
    assert "rolled back" in result.message


@pytest.mark.asyncio
async def test_rollback_evolution_unknown_version_raises_bad_request(monkeypatch):
    init_config(config_path="config/mindmemos/dev.example.yaml")

    async def _fake_rollback(project_id: str, version: int):
        del project_id
        raise ValueError(f"evolution version {version} not found")

    monkeypatch.setattr(
        ms_module.EvolutionStateStore,
        "rollback",
        staticmethod(_fake_rollback),
    )
    service = ms_module.get_memory_service()

    with pytest.raises(ms_module.BadRequestError, match="version 9 not found"):
        await service.rollback_evolution(_auth(), EvolutionRollbackRequest(version=9))
