"""Structured algorithm logging to the console and Skill database."""

from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any, TextIO

from ..infra.database import ScopedDatabase
from ..persistence.models import AlgorithmLogRecord
from ..persistence.records import to_database_record
from ..persistence.tables import ALGORITHM_LOG_TABLE


class LogLevel(StrEnum):
    """Severity written consistently to both logging backends."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class AlgorithmLogger:
    """Emit one structured algorithm log to the console and optional database."""

    def __init__(
        self,
        *,
        algorithm_name: str,
        algorithm_version: str | None = None,
        database: ScopedDatabase | None = None,
        console: TextIO | None = None,
    ) -> None:
        if not algorithm_name:
            raise ValueError("algorithm_name must not be empty")
        self.algorithm_name = algorithm_name
        self.algorithm_version = algorithm_version
        self._database = database
        self._console = console or sys.stdout

    async def log(
        self,
        *,
        component_name: str,
        step_name: str,
        message: str,
        status: str | None = None,
        payload: dict[str, Any] | None = None,
        level: LogLevel = LogLevel.INFO,
    ) -> AlgorithmLogRecord:
        """Write a log to every configured backend and return its durable row."""

        record = AlgorithmLogRecord(
            log_id=str(uuid.uuid4()),
            algorithm_name=self.algorithm_name,
            algorithm_version=self.algorithm_version,
            component_name=component_name,
            step_name=step_name,
            status=status,
            payload={
                **(payload or {}),
                "level": level.value,
                "message": message,
            },
        )
        self._write_console(record)
        if self._database is not None:
            await self._database.upsert_records(ALGORITHM_LOG_TABLE, [to_database_record(record)])
        return record

    def _write_console(self, record: AlgorithmLogRecord) -> None:
        timestamp = record.created_at.astimezone().strftime("%Y-%m-%d %H:%M:%S")
        level = str(record.payload["level"])
        status = f" status={record.status}" if record.status is not None else ""
        message = str(record.payload["message"])
        print(
            f"[{timestamp}] {level} {record.algorithm_name} "
            f"{record.component_name}.{record.step_name}{status}: {message}",
            file=self._console,
            flush=True,
        )


def format_log_value(value: Any) -> str:
    """Render structured values compactly for human-facing algorithm messages."""

    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


__all__ = ["AlgorithmLogger", "LogLevel", "format_log_value"]
