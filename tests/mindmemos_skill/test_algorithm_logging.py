from __future__ import annotations

from io import StringIO
from pathlib import Path

import pytest
from mindmemos_skill.infra.database import DatabaseScope
from mindmemos_skill.logging import AlgorithmLogger, LogLevel
from mindmemos_skill.persistence import (
    ALGORITHM_LOG_TABLE,
    AlgorithmLogRecord,
    bootstrap_skill_database,
    from_database_record,
)


@pytest.mark.asyncio
async def test_algorithm_logger_writes_console_and_sqlite(tmp_path: Path) -> None:
    database = await bootstrap_skill_database(tmp_path / "state.db")
    console = StringIO()
    logger = AlgorithmLogger(
        algorithm_name="demo_algorithm",
        algorithm_version="2.0.0",
        database=database,
        console=console,
    )

    try:
        record = await logger.log(
            component_name="phase",
            step_name="phase_completed",
            status="succeeded",
            level=LogLevel.INFO,
            message="phase train completed",
            payload={"run_id": "run-1", "score_mean": 0.75},
        )
        rows = await database.get_records(ALGORITHM_LOG_TABLE, DatabaseScope(), [record.log_id])
    finally:
        await database.close()

    assert len(rows) == 1
    restored = from_database_record(rows[0], AlgorithmLogRecord)
    assert restored.algorithm_name == "demo_algorithm"
    assert restored.algorithm_version == "2.0.0"
    assert restored.component_name == "phase"
    assert restored.step_name == "phase_completed"
    assert restored.status == "succeeded"
    assert restored.payload == {
        "run_id": "run-1",
        "score_mean": 0.75,
        "level": "INFO",
        "message": "phase train completed",
    }
    output = console.getvalue()
    assert "INFO demo_algorithm phase.phase_completed status=succeeded" in output
    assert output.endswith(": phase train completed\n")
