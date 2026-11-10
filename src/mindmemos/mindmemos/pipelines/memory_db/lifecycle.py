"""Deterministic, read-only memory lifecycle projections for the Console.

The projection deliberately reports only evidence persisted by the write
pipelines.  Missing historical fields are surfaced as unavailable instead of
being inferred from point order or timestamps.
"""

from __future__ import annotations

import base64
import binascii
import json
from collections import Counter, defaultdict
from datetime import UTC, date, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class LifecycleAvailability(BaseModel):
    lifecycle_complete: bool = False
    graph_available: bool = True
    scanned_all: bool | None = None
    missing_fields: list[str] = Field(default_factory=list)


class MemoryProvenance(BaseModel):
    available: bool = False
    add_record_id: str | None = None
    request_id: str | None = None
    source_type: str | None = None
    submitted_at: str | None = None


class MemoryLifecycleEvent(BaseModel):
    event_type: Literal["created", "reinforced", "updated", "archived", "vector_pending"]
    occurred_at: str | None = None
    count: int | None = None
    evidence: dict[str, Any] = Field(default_factory=dict)


class MemoryLifecycleProjection(BaseModel):
    events: list[MemoryLifecycleEvent] = Field(default_factory=list)
    provenance: MemoryProvenance = Field(default_factory=MemoryProvenance)
    lineage: list[str] = Field(default_factory=list)
    availability: LifecycleAvailability = Field(default_factory=LifecycleAvailability)


class MemoryInventoryItem(BaseModel):
    memory_id: str
    content: str = ""
    status: str = "active"
    mem_type: str | None = None
    algorithm: str | None = None
    user_id: str | None = None
    session_id: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    reinforcement_count: int = 0
    entity_id: str | None = None
    entity_type: str | None = None
    property_name: str | None = None
    lifecycle_available: bool = False


class MemoryInsightBucket(BaseModel):
    date: str
    created: int = 0
    reinforced: int = 0
    updated: int = 0
    archived: int = 0


class MemoryInsights(BaseModel):
    counts: dict[str, int]
    actions: dict[str, int]
    trend: list[MemoryInsightBucket]
    availability: LifecycleAvailability
    range_days: int = 30
    as_of: str


class MemoryGraphNode(BaseModel):
    id: str
    node_type: Literal["memory", "entity", "source"]
    label: str
    status: str | None = None


class MemoryGraphEdge(BaseModel):
    source: str
    target: str
    edge_type: Literal["mentions", "extracted_from", "derived_from", "has_property_memory"]


class MemoryGraph(BaseModel):
    nodes: list[MemoryGraphNode] = Field(default_factory=list)
    edges: list[MemoryGraphEdge] = Field(default_factory=list)
    truncated: bool = False
    next_cursor: str | None = None
    availability: LifecycleAvailability = Field(default_factory=LifecycleAvailability)


def encode_memory_graph_cursor(*, node_type: str, node_id: str, offset: int) -> str:
    """Encode a versioned cursor that cannot be silently reused for another root node."""

    if offset < 0:
        raise ValueError("memory graph cursor offset must be non-negative")
    payload = json.dumps(
        {"v": 1, "node_type": node_type, "node_id": node_id, "offset": offset},
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return base64.urlsafe_b64encode(payload).decode().rstrip("=")


def decode_memory_graph_cursor(cursor: str, *, node_type: str, node_id: str) -> int:
    """Decode and bind a graph cursor to the requested node identity."""

    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded.encode()).decode())
    except (binascii.Error, UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
        raise ValueError("invalid memory graph cursor") from exc

    if not isinstance(payload, dict) or payload.get("v") != 1:
        raise ValueError("invalid memory graph cursor")
    if payload.get("node_type") != node_type or payload.get("node_id") != node_id:
        raise ValueError("memory graph cursor does not match requested node")
    offset = payload.get("offset")
    if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
        raise ValueError("invalid memory graph cursor")
    return offset


def project_inventory_item(payload: dict[str, Any], *, point_id: str) -> MemoryInventoryItem:
    created_at = _timestamp(payload.get("created_at"))
    updated_at = _timestamp(payload.get("update_at") or payload.get("updated_at"))
    return MemoryInventoryItem(
        memory_id=str(payload.get("memory_id") or point_id),
        content=str(payload.get("content") or ""),
        status=str(payload.get("status") or "active"),
        mem_type=_optional_text(payload.get("mem_type")),
        algorithm=_optional_text(payload.get("mem_extract_type")),
        user_id=_optional_text(payload.get("user_id")),
        session_id=_optional_text(payload.get("session_id")),
        created_at=created_at,
        updated_at=updated_at,
        reinforcement_count=_non_negative_int(payload.get("reinforcement_count")),
        entity_id=_optional_text(payload.get("entity_id")),
        entity_type=_optional_text(payload.get("entity_type")),
        property_name=_optional_text(payload.get("property_name")),
        lifecycle_available=created_at is not None,
    )


def project_memory_lifecycle(
    payload: dict[str, Any],
    *,
    add_record: dict[str, Any] | None,
    lineage: list[str],
    provenance_unavailable: bool = False,
) -> MemoryLifecycleProjection:
    """Build a truthful timeline from persisted evidence only."""

    created_at = _timestamp(payload.get("created_at"))
    updated_at = _timestamp(payload.get("update_at") or payload.get("updated_at"))
    status_changed_at = _timestamp(payload.get("status_changed_at"))
    status = str(payload.get("status") or "active")
    reinforcement_count = _non_negative_int(payload.get("reinforcement_count"))
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}

    add_record_id = _source_add_record_id(payload, metadata, add_record)
    provenance = MemoryProvenance(
        available=not provenance_unavailable and bool(add_record_id or add_record),
        add_record_id=add_record_id,
        request_id=_optional_text((add_record or {}).get("request_id") or payload.get("request_id")),
        source_type=_optional_text(
            (add_record or {}).get("source_type") or (add_record or {}).get("input_type") or metadata.get("source")
        ),
        submitted_at=_timestamp((add_record or {}).get("request_submitted_at") or (add_record or {}).get("added_at")),
    )

    events: list[MemoryLifecycleEvent] = []
    if created_at:
        events.append(MemoryLifecycleEvent(event_type="created", occurred_at=created_at))
    if reinforcement_count:
        events.append(
            MemoryLifecycleEvent(
                event_type="reinforced",
                occurred_at=updated_at or created_at,
                count=reinforcement_count,
                evidence={"reinforcement_count": reinforcement_count},
            )
        )
    if updated_at and updated_at != created_at:
        events.append(MemoryLifecycleEvent(event_type="updated", occurred_at=updated_at))
    if status == "archived" and status_changed_at:
        events.append(MemoryLifecycleEvent(event_type="archived", occurred_at=status_changed_at))
    if status == "vector_pending" or metadata.get("vector_pending") is True:
        events.append(
            MemoryLifecycleEvent(
                event_type="vector_pending",
                occurred_at=updated_at or status_changed_at or created_at,
            )
        )

    missing: list[str] = []
    if not created_at:
        missing.append("created_at")
    if not provenance.available:
        missing.append("provenance")
    return MemoryLifecycleProjection(
        events=events,
        provenance=provenance,
        lineage=list(dict.fromkeys(str(item) for item in lineage if item)),
        availability=LifecycleAvailability(
            lifecycle_complete=not missing,
            missing_fields=missing,
        ),
    )


def summarize_memory_insights(
    records: list[tuple[str, dict[str, Any]]],
    *,
    scanned_all: bool,
    counts_override: dict[str, int] | None = None,
    range_days: int = 30,
    as_of: str | None = None,
) -> MemoryInsights:
    counts: Counter[str] = Counter()
    actions: Counter[str] = Counter()
    buckets: dict[str, Counter[str]] = defaultdict(Counter)
    missing_fields: set[str] = set()
    action_types = ("created", "reinforced", "updated", "archived")
    for point_id, payload in records:
        item = project_inventory_item(payload, point_id=point_id)
        counts[item.status] += 1
        counts["total"] += 1
        lifecycle = project_memory_lifecycle(payload, add_record=None, lineage=[])
        missing_fields.update(lifecycle.availability.missing_fields)
        for event in lifecycle.events:
            if event.event_type not in action_types:
                continue
            amount = event.count if event.event_type == "reinforced" and event.count else 1
            actions[event.event_type] += amount
            bucket_date = _date_part(event.occurred_at)
            if bucket_date:
                buckets[bucket_date][event.event_type] += amount
    if counts_override is not None:
        counts = Counter({key: _non_negative_int(value) for key, value in counts_override.items()})
    for key in ("active", "archived", "vector_pending", "total"):
        counts.setdefault(key, 0)
    for key in action_types:
        actions.setdefault(key, 0)
    trend = [
        MemoryInsightBucket(date=day, **{key: values[key] for key in action_types})
        for day, values in sorted(buckets.items())
    ]
    return MemoryInsights(
        counts=dict(counts),
        actions=dict(actions),
        trend=trend,
        availability=LifecycleAvailability(
            scanned_all=scanned_all,
            missing_fields=sorted(missing_fields),
        ),
        range_days=range_days,
        as_of=as_of or datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    )


def _source_add_record_id(
    payload: dict[str, Any], metadata: dict[str, Any], add_record: dict[str, Any] | None
) -> str | None:
    value = (
        payload.get("source_add_record_id")
        or payload.get("add_record_id")
        or metadata.get("source_add_record_id")
        or (add_record or {}).get("add_record_id")
    )
    return _optional_text(value)


def _timestamp(value: Any) -> str | None:
    if isinstance(value, datetime):
        return value.isoformat().replace("+00:00", "Z")
    return value.strip() if isinstance(value, str) and value.strip() else None


def _date_part(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value[:10]).isoformat()
    except ValueError:
        return None


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _non_negative_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0
