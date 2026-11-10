from __future__ import annotations

import pytest
from mindmemos.pipelines.memory_db.lifecycle import (
    decode_memory_graph_cursor,
    encode_memory_graph_cursor,
    project_inventory_item,
    project_memory_lifecycle,
    summarize_memory_insights,
)


def test_memory_graph_cursor_is_bound_to_node_identity() -> None:
    cursor = encode_memory_graph_cursor(node_type="entity", node_id="entity-1", offset=40)

    assert decode_memory_graph_cursor(cursor, node_type="entity", node_id="entity-1") == 40
    with pytest.raises(ValueError, match="does not match"):
        decode_memory_graph_cursor(cursor, node_type="memory", node_id="entity-1")


def test_project_memory_lifecycle_uses_explicit_evidence_without_inventing_history() -> None:
    result = project_memory_lifecycle(
        {
            "memory_id": "mem-1",
            "content": "user prefers concise answers",
            "status": "active",
            "mem_extract_type": "schema",
            "created_at": "2026-08-01T08:00:00Z",
            "update_at": "2026-08-02T09:30:00Z",
            "reinforcement_count": 2,
            "parent_ids": ["mem-parent"],
            "root_id": ["mem-root"],
            "metadata": {"source_add_record_id": "add-1"},
        },
        add_record={"add_record_id": "add-1", "request_submitted_at": "2026-08-01T07:59:59Z"},
        lineage=["mem-parent", "mem-root"],
    )

    assert [event.event_type for event in result.events] == ["created", "reinforced", "updated"]
    assert result.provenance.add_record_id == "add-1"
    assert result.lineage == ["mem-parent", "mem-root"]
    assert result.availability.lifecycle_complete is True


def test_project_memory_lifecycle_marks_legacy_gaps_unavailable() -> None:
    result = project_memory_lifecycle(
        {"memory_id": "legacy-1", "content": "legacy", "status": "active"},
        add_record=None,
        lineage=[],
    )

    assert result.events == []
    assert result.provenance.available is False
    assert result.availability.lifecycle_complete is False
    assert "created_at" in result.availability.missing_fields
    assert "provenance" in result.availability.missing_fields


def test_project_memory_lifecycle_projects_archival_only_from_status_evidence() -> None:
    result = project_memory_lifecycle(
        {
            "memory_id": "mem-archived",
            "content": "old preference",
            "status": "archived",
            "created_at": "2026-08-01T08:00:00Z",
            "status_changed_at": "2026-08-03T10:00:00Z",
        },
        add_record=None,
        lineage=[],
    )

    assert [event.event_type for event in result.events] == ["created", "archived"]
    assert result.events[-1].occurred_at == "2026-08-03T10:00:00Z"


@pytest.mark.parametrize("algorithm", ["vanilla", "schema", "feedback", "dreaming"])
def test_inventory_preserves_native_algorithm_and_pending_state(algorithm: str) -> None:
    item = project_inventory_item(
        {
            "content": "evidence",
            "created_at": "2026-08-01T08:00:00Z",
            "mem_extract_type": algorithm,
            "status": "vector_pending",
        },
        point_id=f"{algorithm}-1",
    )

    assert item.algorithm == algorithm
    assert item.status == "vector_pending"
    assert item.lifecycle_available is True


def test_insights_match_the_direct_fixture_without_fabricating_missing_dates() -> None:
    result = summarize_memory_insights(
        [
            (
                "active-1",
                {
                    "status": "active",
                    "created_at": "2026-08-01T08:00:00Z",
                    "update_at": "2026-08-02T09:30:00Z",
                    "reinforcement_count": 2,
                },
            ),
            (
                "archived-1",
                {
                    "status": "archived",
                    "created_at": "2026-08-01T10:00:00Z",
                    "status_changed_at": "2026-08-03T10:00:00Z",
                },
            ),
            ("legacy-1", {"status": "vector_pending"}),
        ],
        scanned_all=True,
    )

    assert result.counts == {
        "active": 1,
        "archived": 1,
        "vector_pending": 1,
        "total": 3,
    }
    assert result.actions == {
        "created": 2,
        "reinforced": 2,
        "updated": 1,
        "archived": 1,
    }
    assert [bucket.date for bucket in result.trend] == ["2026-08-01", "2026-08-02", "2026-08-03"]
    assert "created_at" in result.availability.missing_fields
