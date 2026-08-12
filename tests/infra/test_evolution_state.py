"""Tests for the evolution state store (Qdrant version log + file mirror)."""

from __future__ import annotations

import json
from typing import Any

import pytest
from qdrant_client import models as qmodels

from mindmemos.infra.db.collections.evolution_state import EvolutionStateRepository
from mindmemos.infra.db.evolution_state import EvolutionStateStore
from mindmemos.infra.db.models import QdrantRecord
from mindmemos.typing import EvolutionState, ParameterChange


class _FakeEvolutionStateRepository(EvolutionStateRepository):
    """In-memory repo matching the CollectionRepository surface used by the store."""

    def __init__(self) -> None:
        self._points: dict[str, dict[str, Any]] = {}

    @property
    def collection(self) -> str:
        return "evolution_state_v1"

    async def upsert(self, point_id: str, payload: dict[str, Any]) -> None:
        self._points[point_id] = payload

    async def get(self, project_id: str, point_id: str) -> QdrantRecord | None:
        payload = self._points.get(point_id)
        if payload is None or payload.get("project_id") != project_id:
            return None
        return QdrantRecord(point_id=point_id, payload=payload)

    async def scroll(
        self,
        project_id: str,
        *,
        filter_: qmodels.Filter | None = None,
        limit: int = 50,
        cursor: Any | None = None,
        order_by: Any | None = None,
    ) -> tuple[list[QdrantRecord], Any | None]:
        del cursor, order_by
        records = [
            QdrantRecord(point_id=pid, payload=payload)
            for pid, payload in self._points.items()
            if payload.get("project_id") == project_id
        ]
        if filter_ is not None and filter_.must:
            field = filter_.must[0].key
            value = filter_.must[0].match.value
            records = [r for r in records if r.payload.get(field) == value]
        return records[:limit], None


@pytest.fixture
def repo() -> _FakeEvolutionStateRepository:
    return _FakeEvolutionStateRepository()


def _change(path: str, after: Any) -> ParameterChange:
    return ParameterChange(path=path, before=None, after=after)


@pytest.mark.asyncio
async def test_apply_creates_versioned_state_and_flips_current(repo, tmp_path):
    store = EvolutionStateStore(repo=repo, file_history_dir=tmp_path / "evolved")

    first = await store.apply(
        "proj_1",
        add_config={"extraction_prompt": "v1"},
        search_config={"top_k": 10},
        changes=[_change("search_config.top_k", 10)],
    )
    assert first.version == 1

    second = await store.apply(
        "proj_1",
        add_config={"extraction_prompt": "v1"},
        search_config={"top_k": 15},
        changes=[_change("search_config.top_k", 15)],
    )
    assert second.version == 2

    current = await store.get_current("proj_1")
    assert current is not None
    assert current.version == 2
    assert current.is_current is True
    assert current.search_config["top_k"] == 15

    v1 = await store.get_version("proj_1", 1)
    assert v1 is not None
    assert v1.is_current is False


@pytest.mark.asyncio
async def test_rollback_restores_previous_version(repo, tmp_path):
    store = EvolutionStateStore(repo=repo, file_history_dir=tmp_path / "evolved")

    await store.apply(
        "proj_1",
        add_config={"extraction_prompt": "v1"},
        search_config={"top_k": 10},
        changes=[],
    )
    await store.apply(
        "proj_1",
        add_config={"extraction_prompt": "v1"},
        search_config={"top_k": 15},
        changes=[_change("search_config.top_k", 15)],
    )

    result = await store.rollback("proj_1", 1)
    assert result.version == 1
    assert result.is_rollback is True

    current = await store.get_current("proj_1")
    assert current is not None
    assert current.version == 1
    assert current.search_config["top_k"] == 10


@pytest.mark.asyncio
async def test_apply_mirrors_history_and_current_files(repo, tmp_path):
    store = EvolutionStateStore(repo=repo, file_history_dir=tmp_path / "evolved")

    await store.apply(
        "proj_1",
        add_config={"extraction_prompt": "v1"},
        search_config={"top_k": 10},
        changes=[_change("search_config.top_k", 10)],
    )

    history_dir = tmp_path / "evolved" / "proj_1" / "history"
    assert (history_dir / "001_v1.json").exists()
    assert (tmp_path / "evolved" / "proj_1" / "current.json").exists()
    mirrored = json.loads((history_dir / "001_v1.json").read_text(encoding="utf-8"))
    assert mirrored["version"] == 1
    assert mirrored["is_current"] is True


@pytest.mark.asyncio
async def test_rollback_unknown_version_raises(repo, tmp_path):
    store = EvolutionStateStore(repo=repo, file_history_dir=tmp_path / "evolved")

    with pytest.raises(ValueError, match="version 9 not found"):
        await store.rollback("proj_1", 9)
