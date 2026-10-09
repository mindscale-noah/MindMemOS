"""Verify reported training evidence comes from saved rollout messages."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from mindmemos_skill.common.json import write_json

from scripts.treeskill.local_service import TrajectoryReporter


@pytest.mark.asyncio
async def test_collection_reports_original_rollout_once(tmp_path):
    messages = [{"role": "assistant", "content": "Saved workbook"}]
    directory = tmp_path / "training/train/rollouts/train/task-000000-sample-000"
    write_json(
        directory / "completed.json",
        {
            "payload": {
                "result": {
                    "trajectory": messages,
                    "usage": {"total_tokens": 12},
                }
            }
        },
    )
    write_json(directory.parent / "failed-slot/completed.json", {"payload": {"failure": {"error": "503"}}})
    service = SimpleNamespace(
        report_trajectory=AsyncMock(
            return_value=SimpleNamespace(model_dump=lambda **kwargs: {"trajectory_id": "reported"})
        )
    )
    reporter = TrajectoryReporter(service, tmp_path, "project", "initial-version")
    await reporter.report(stage="collection", message="Collected")
    await reporter.report(stage="collection", message="Resumed")
    service.report_trajectory.assert_awaited_once()
    request = service.report_trajectory.call_args.args[0]
    assert request.messages == messages
    assert request.skill_version_id == "initial-version"
    assert request.usage == {"total_tokens": 12}
