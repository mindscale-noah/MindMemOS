"""Paged-walk invariants of ``PgSkillGraphStorageAdapter`` against stub backends.

The structured and vector backends implement the page cursor as an OFFSET.
A row inserted concurrently before the page boundary shifts the next page by
one and re-yields the previous page's last row; a walk that counted that row
twice would let ``gc_retired_generations`` delete inside the keep window. The
adapter must therefore return each record at most once.
"""

from __future__ import annotations

import base64
import json
from typing import Any

import pytest
from mindmemos_skill.infra.database import DatabaseScope, Record, RecordQuery
from mindmemos_skill.infra.skill_manage.pg_storage import PgSkillGraphStorageAdapter
from mindmemos_skill.infra.vector_store import DatabaseScope as VectorScope
from mindmemos_skill.infra.vector_store import Record as VectorRecord
from mindmemos_skill.infra.vector_store import RecordQuery as VectorRecordQuery
from mindmemos_skill.typing import SkillManageScope

SCOPE = SkillManageScope(project_id="p", graph_id="g")


def _cursor(offset: int) -> str:
    return base64.urlsafe_b64encode(json.dumps({"table": "t", "offset": offset}).encode()).decode()


def _offset(cursor: str | None) -> int:
    return 0 if cursor is None else int(json.loads(base64.urlsafe_b64decode(cursor))["offset"])


class _ShiftingDatabase:
    """OFFSET-paged structured store into which a row is inserted after the first page is served."""

    def __init__(self, table: str, rows: list[Record], inserted_after_first_page: Record) -> None:
        self._table = table
        self._rows = list(rows)
        self._pending = inserted_after_first_page
        self.pages_served = 0

    async def query_records(self, table: str, query: RecordQuery) -> tuple[list[Record], str | None]:
        assert table == self._table
        offset = _offset(query.page.cursor)
        limit = query.page.limit
        window = self._rows[offset : offset + limit + 1]
        self.pages_served += 1
        if self._pending is not None:
            # A concurrent writer lands a row that sorts before the boundary.
            self._rows.insert(0, self._pending)
            self._pending = None
        has_more = len(window) > limit
        return window[:limit], (_cursor(offset + limit) if has_more else None)


class _ShiftingVectorStore:
    def __init__(self, table: str, rows: list[VectorRecord], inserted_after_first_page: VectorRecord) -> None:
        self._table = table
        self._rows = list(rows)
        self._pending = inserted_after_first_page

    async def scroll(
        self, table: str, query: VectorRecordQuery, *, with_vectors: bool = False
    ) -> tuple[list[VectorRecord], str | None]:
        assert table == self._table
        offset = _offset(query.page.cursor)
        limit = query.page.limit
        window = self._rows[offset : offset + limit + 1]
        if self._pending is not None:
            self._rows.insert(0, self._pending)
            self._pending = None
        has_more = len(window) > limit
        return window[:limit], (_cursor(offset + limit) if has_more else None)


def _record(record_id: str) -> Record:
    return Record(
        table="t", record_id=record_id, scope=DatabaseScope(project_id="p", graph_id="g"), payload={"id": record_id}
    )


def _vector_record(record_id: str) -> VectorRecord:
    return VectorRecord(
        table="t", record_id=record_id, scope=VectorScope(project_id="p", graph_id="g"), payload={"id": record_id}
    )


@pytest.mark.asyncio
async def test_structured_paged_walk_returns_each_record_once_when_a_row_lands_before_the_boundary() -> None:
    rows = [_record(f"r{index}") for index in range(5)]
    database = _ShiftingDatabase("t", rows, _record("r-new"))
    adapter = PgSkillGraphStorageAdapter(database=database, vector_store=object(), page_size=2)  # type: ignore[arg-type]

    walked = await adapter._query_structured_paged("t", DatabaseScope(project_id="p", graph_id="g"))

    ids = [record.record_id for record in walked]
    # Without dedupe the OFFSET shift re-yields r1 on the second page.
    assert len(ids) == len(set(ids))
    assert set(ids) == {"r0", "r1", "r2", "r3", "r4"}
    assert database.pages_served == 3


@pytest.mark.asyncio
async def test_vector_paged_walk_returns_each_record_once_when_a_row_lands_before_the_boundary() -> None:
    rows = [_vector_record(f"v{index}") for index in range(5)]
    store = _ShiftingVectorStore("t", rows, _vector_record("v-new"))
    adapter = PgSkillGraphStorageAdapter(database=object(), vector_store=store, page_size=2)  # type: ignore[arg-type]

    walked = await adapter._query_vector_paged("t", VectorScope(project_id="p", graph_id="g"))

    ids = [record.record_id for record in walked]
    assert len(ids) == len(set(ids))
    assert set(ids) == {"v0", "v1", "v2", "v3", "v4"}


class _GenerationsDatabase(_ShiftingDatabase):
    """Serves ``skill_manage_generations`` rows; every other table is empty."""

    async def query_records(self, table: str, query: RecordQuery) -> tuple[list[Record], str | None]:
        if table != self._table:
            return [], None
        return await super().query_records(table, query)

    async def get_records(self, table: str, scope: DatabaseScope, record_ids: Any) -> list[Record]:
        return []

    async def delete_records(self, table: str, scope: DatabaseScope, record_ids: Any) -> None:
        self._rows = [row for row in self._rows if row.record_id not in set(record_ids)]


class _EmptyVectorStore:
    async def scroll(self, table, query, *, with_vectors=False):
        return [], None

    async def delete_records(self, table, scope, record_ids):
        return None


def _generation(generation_id: str, retired_at: str) -> Record:
    return Record(
        table="skill_manage_generations",
        record_id=generation_id,
        scope=DatabaseScope(project_id="p", graph_id="g", generation_id=generation_id),
        payload={"generation_id": generation_id, "status": "retired", "retired_at": retired_at},
    )


@pytest.mark.asyncio
async def test_gc_never_over_deletes_inside_the_keep_window_when_the_generation_walk_shifts() -> None:
    """Three retired generations, keep=2: exactly the oldest goes, even though the OFFSET walk re-yields one row."""

    retired = [_generation(f"gen-{index}", f"2026-09-03T00:0{index}:00+00:00") for index in range(3)]
    staged = Record(
        table="skill_manage_generations",
        record_id="gen-new",
        scope=DatabaseScope(project_id="p", graph_id="g", generation_id="gen-new"),
        payload={"generation_id": "gen-new", "status": "staging"},
    )
    database = _GenerationsDatabase("skill_manage_generations", retired, staged)
    adapter = PgSkillGraphStorageAdapter(database=database, vector_store=_EmptyVectorStore(), page_size=2)  # type: ignore[arg-type]

    removed = await adapter.gc_retired_generations(SCOPE, keep=2, min_age_seconds=0.0, now="2026-09-03T01:00:00+00:00")

    assert removed == ["gen-0"]
    assert {row.record_id for row in database._rows} == {"gen-new", "gen-1", "gen-2"}
