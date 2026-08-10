"""Mapping between persistence row models and generic database records."""

from __future__ import annotations

from typing import TypeVar, cast

from pydantic import BaseModel

from ..infra.database import DatabaseScope, Record
from .models import AlgorithmLogRecord, SkillRecord, SkillRemoteOperationRecord, SkillSyncStateRecord, TrajectoryRecord
from .tables import (
    ALGORITHM_LOG_TABLE,
    SKILL_REMOTE_OPERATION_TABLE,
    SKILL_SYNC_STATE_TABLE,
    SKILL_TABLE,
    TRAJECTORY_TABLE,
)

PersistenceRecord = SkillRecord | SkillSyncStateRecord | SkillRemoteOperationRecord | TrajectoryRecord | AlgorithmLogRecord
PersistenceRecordType = TypeVar("PersistenceRecordType", bound=PersistenceRecord)

_MODEL_TABLES: dict[type[BaseModel], tuple[str, str]] = {
    SkillRecord: (SKILL_TABLE, "version_id"),
    SkillSyncStateRecord: (SKILL_SYNC_STATE_TABLE, "skill_id"),
    SkillRemoteOperationRecord: (SKILL_REMOTE_OPERATION_TABLE, "operation_id"),
    TrajectoryRecord: (TRAJECTORY_TABLE, "trajectory_id"),
    AlgorithmLogRecord: (ALGORITHM_LOG_TABLE, "log_id"),
}


def to_database_record(model: PersistenceRecord) -> Record:
    """Serialize one business row without exposing it to infra."""

    try:
        table, identity_field = _MODEL_TABLES[type(model)]
    except KeyError as exc:
        raise TypeError(f"unsupported persistence model: {type(model).__name__}") from exc
    payload = model.model_dump(mode="json")
    return Record(
        table=table,
        record_id=str(payload[identity_field]),
        scope=DatabaseScope(),
        payload=payload,
    )


def from_database_record(record: Record, model_type: type[PersistenceRecordType]) -> PersistenceRecordType:
    """Validate a generic database row back at the business boundary."""

    try:
        expected_table, _ = _MODEL_TABLES[model_type]
    except KeyError as exc:
        raise TypeError(f"unsupported persistence model: {model_type.__name__}") from exc
    if record.table != expected_table:
        raise ValueError(f"record from table {record.table!r} cannot be loaded as {model_type.__name__}")
    return cast(PersistenceRecordType, model_type.model_validate(record.payload))


__all__ = ["PersistenceRecord", "from_database_record", "to_database_record"]
