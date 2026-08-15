from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from mindmemos_skill.envs.registered_envs.spreadsheetbench import SYSTEM_PROMPT, build_messages, recalculation
from mindmemos_skill.typing import Task


def _write_workbook(path: Path, *, formula: bool) -> None:
    openpyxl = pytest.importorskip("openpyxl")
    workbook = openpyxl.Workbook()
    worksheet = workbook.active
    worksheet["A1"] = 1
    worksheet["A2"] = 2
    worksheet["B1"] = "=SUM(A1:A2)" if formula else 3
    workbook.save(path)
    workbook.close()


class _FakeOfficeProcess:
    def __init__(self) -> None:
        self.running = True
        self.terminated = False

    def poll(self) -> int | None:
        return None if self.running else 0

    def terminate(self) -> None:
        self.terminated = True
        self.running = False

    def kill(self) -> None:
        self.running = False

    def communicate(self, timeout: int | None = None) -> tuple[str, str]:
        del timeout
        self.running = False
        return "", ""


def test_recalculation_prompt_is_shared_without_changing_default_prompt(tmp_path: Path) -> None:
    task = Task(task_id="sheet-1", instruction="Write the required result.")
    default = build_messages(task=task, skill_names=[])
    routed = build_messages(task=task, skill_names=[], recalculation_workspace=tmp_path)
    full = build_messages(task=task, skill_names=["xlsx"], recalculation_workspace=tmp_path)

    assert default[0]["content"] == SYSTEM_PROMPT
    assert "with no formula recalculation" in default[0]["content"]
    assert "with no formula recalculation" not in routed[0]["content"]
    assert "with no formula recalculation" not in full[0]["content"]
    routed_suffix = routed[1]["content"].split("### formula_recalculation", 1)[1]
    full_suffix = full[1]["content"].split("### formula_recalculation", 1)[1]
    assert routed_suffix == full_suffix
    assert "output.xlsx" in routed_suffix
    assert recalculation.STATUS_FILENAME in routed_suffix
    assert f"./{recalculation.TASK_LOCAL_HELPER_FILENAME}" in routed_suffix
    assert str(Path(recalculation.__file__).resolve()) not in routed_suffix
    assert (tmp_path / recalculation.TASK_LOCAL_HELPER_FILENAME).resolve() == Path(recalculation.__file__).resolve()
    assert "Do not substitute another helper or change its paths" in routed_suffix


def test_task_local_recalculation_launcher_fallback_is_idempotent(tmp_path: Path) -> None:
    with patch.object(Path, "symlink_to", side_effect=OSError("symlinks unavailable")):
        first = recalculation.stage_task_local_helper(tmp_path)
        second = recalculation.stage_task_local_helper(tmp_path)

    assert first == second
    assert "runpy.run_path" in first.read_text(encoding="utf-8")


def test_formula_free_workbook_is_not_modified_or_launched(tmp_path: Path) -> None:
    workbook = tmp_path / "output.xlsx"
    status_path = tmp_path / "status.json"
    _write_workbook(workbook, formula=False)
    before = recalculation._sha256(workbook)

    with patch.object(
        recalculation,
        "_find_libreoffice_programs",
        side_effect=AssertionError("LibreOffice must not start"),
    ):
        status = recalculation.recalculate_workbook(workbook, status_path=status_path)

    assert status["status"] == "not_needed"
    assert status["attempted"] is False
    assert status["replaced_output"] is False
    assert recalculation._sha256(workbook) == before
    assert json.loads(status_path.read_text(encoding="utf-8")) == status


def test_recalculation_setup_failure_preserves_original_workbook(tmp_path: Path) -> None:
    workbook = tmp_path / "output.xlsx"
    _write_workbook(workbook, formula=True)
    before = recalculation._sha256(workbook)

    with patch.object(
        recalculation,
        "_find_libreoffice_programs",
        side_effect=FileNotFoundError("test LibreOffice unavailable"),
    ):
        status = recalculation.recalculate_workbook(workbook)

    assert status["status"] == "failure"
    assert "test LibreOffice unavailable" in status["error"]
    assert status["replaced_output"] is False
    assert recalculation._sha256(workbook) == before


def test_successful_recalculation_replaces_only_after_validation(tmp_path: Path) -> None:
    workbook = tmp_path / "output.xlsx"
    _write_workbook(workbook, formula=True)
    office_process = _FakeOfficeProcess()
    before_state = {
        "formula_count": 1,
        "cached_formula_value_count": 0,
        "excel_errors": [],
    }
    after_state = {
        "formula_count": 1,
        "cached_formula_value_count": 1,
        "excel_errors": [],
    }

    with (
        patch.object(
            recalculation,
            "_find_libreoffice_programs",
            return_value=(Path("/fake/soffice"), Path("/fake/python")),
        ),
        patch.object(
            recalculation,
            "_workbook_formula_state",
            side_effect=[before_state, after_state],
        ),
        patch.object(recalculation.subprocess, "Popen", return_value=office_process) as popen,
        patch.object(
            recalculation.subprocess,
            "run",
            return_value=subprocess.CompletedProcess(["/fake/python"], 0, stdout="ok", stderr=""),
        ) as run,
    ):
        status = recalculation.recalculate_workbook(workbook)

    assert status["status"] == "success"
    assert status["attempted"] is True
    assert status["replaced_output"] is True
    assert status["cached_formula_value_count_after"] == 1
    assert office_process.terminated is True
    assert "--accept=pipe,name=" in " ".join(popen.call_args.args[0])
    worker_command = run.call_args.args[0]
    assert worker_command[0] == "/fake/python"
    assert "--uno-worker" in worker_command
    assert worker_command[-1].endswith("output.xlsx")


def test_recalculated_formula_errors_are_reported(tmp_path: Path) -> None:
    workbook = tmp_path / "output.xlsx"
    _write_workbook(workbook, formula=True)
    office_process = _FakeOfficeProcess()
    before_state = {
        "formula_count": 1,
        "cached_formula_value_count": 0,
        "excel_errors": [],
    }
    after_state = {
        "formula_count": 1,
        "cached_formula_value_count": 1,
        "excel_errors": [{"sheet": "Sheet", "cell": "B1", "error": "#DIV/0!"}],
    }

    with (
        patch.object(
            recalculation,
            "_find_libreoffice_programs",
            return_value=(Path("/fake/soffice"), Path("/fake/python")),
        ),
        patch.object(
            recalculation,
            "_workbook_formula_state",
            side_effect=[before_state, after_state],
        ),
        patch.object(recalculation.subprocess, "Popen", return_value=office_process),
        patch.object(
            recalculation.subprocess,
            "run",
            return_value=subprocess.CompletedProcess(["/fake/python"], 0, stdout="ok", stderr=""),
        ),
    ):
        status = recalculation.recalculate_workbook(workbook)

    assert status["status"] == "errors_found"
    assert status["replaced_output"] is True
    assert status["excel_errors"] == [{"sheet": "Sheet", "cell": "B1", "error": "#DIV/0!"}]
