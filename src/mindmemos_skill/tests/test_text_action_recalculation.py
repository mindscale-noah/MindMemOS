"""Keep agent-driven PATH-based recalculation isolated from native-tool evaluation."""

import json
import shlex
import sys

import pytest
from mindmemos_skill.agents import OpenAIChatConfig
from mindmemos_skill.envs.builtin_envs.spreadsheetbench import recalculation_cli as recalculation
from mindmemos_skill.envs.builtin_envs.spreadsheetbench.config import SpreadsheetBenchTextActionConfig
from mindmemos_skill.envs.builtin_envs.spreadsheetbench.text_action_mode.environment import (
    SpreadsheetBenchTextActionEnvironment,
)
from mindmemos_skill.envs.builtin_envs.spreadsheetbench.text_action_mode.policy import run_reference_bash
from mindmemos_skill.envs.builtin_envs.spreadsheetbench.text_action_mode.reward import evaluate
from mindmemos_skill.envs.builtin_envs.spreadsheetbench.typing import (
    SpreadsheetBenchRewardData,
    SpreadsheetBenchTask,
    SpreadsheetContext,
    WorkbookInput,
    WorkbookReference,
)
from mindmemos_skill.infra.execution import ExecutionConfig
from openpyxl import Workbook


def save(path, value):
    workbook = Workbook()
    workbook.active["A1"] = value
    workbook.save(path)
    workbook.close()


@pytest.mark.parametrize("enabled", [True, False])
async def test_prompt_provides_only_agent_run_recalculation(tmp_path, enabled):
    save(tmp_path / "input.xlsx", 1)
    env = SpreadsheetBenchTextActionEnvironment(
        SpreadsheetBenchTextActionConfig(agent=OpenAIChatConfig(model="offline"), transactional_recalculation=enabled)
    )
    await env.initialize()
    task = SpreadsheetBenchTask(id="task", instruction="Edit A1", cases=[WorkbookInput(number="1", path=tmp_path)])
    context = SpreadsheetContext(task=task, directory=tmp_path, skill_directory=tmp_path, skill_text="")
    prompt = env._user_prompt(context)
    assert "automatically recalculates" not in prompt
    assert "Formula recalculation is disabled" not in prompt
    helper = tmp_path / recalculation.TASK_LOCAL_HELPER_FILENAME
    assert helper.exists() == enabled
    if enabled:
        command = prompt.split("run this exact command before signaling completion:\n")[1].splitlines()[0]
        assert shlex.split(command) == [
            sys.executable,
            "./.treeskill_recalculate.py",
            "output.xlsx",
            "--status-path",
            ".tree_only_recalc_status.json",
        ]
        assert "Do not substitute another helper" in prompt


async def test_standalone_helper_runs_without_importing_skill_package(tmp_path):
    save(tmp_path / "output.xlsx", 3)
    before = (tmp_path / "output.xlsx").read_bytes()
    helper = recalculation.stage_task_local_helper(tmp_path)
    assert "from mindmemos_skill" not in helper.read_text()
    observation = await run_reference_bash(
        f"python {shlex.quote(str(helper))} output.xlsx",
        working_dir=tmp_path,
        timeout_seconds=10,
        execution=ExecutionConfig(python_command=[sys.executable]),
    )
    status = json.loads(observation)
    assert status["status"] == "not_needed"
    assert (tmp_path / "output.xlsx").read_bytes() == before
    assert not (tmp_path / "output.recalc.xlsx").exists()


def test_text_scoring_does_not_recalculate_or_use_stale_converted_output(tmp_path):
    save(tmp_path / "gold.xlsx", 3)
    save(tmp_path / "output.xlsx", "=1+2")
    save(tmp_path / "output.recalc.xlsx", 3)
    data = SpreadsheetBenchRewardData(
        cases=[WorkbookReference(number="1", path=tmp_path / "gold.xlsx")], answer_position="A1"
    )
    reward = evaluate(data, [tmp_path])
    assert reward.score == 0
    assert "pred=None" in reward.detail
    assert reward.cases[0]["scored_file"] == "output.xlsx"
    assert not (tmp_path / recalculation.STATUS_FILENAME).exists()


def test_unavailable_soffice_preserves_original_workbook(tmp_path, monkeypatch):
    save(tmp_path / "output.xlsx", "=1+2")
    original = (tmp_path / "output.xlsx").read_bytes()

    def missing(**kwargs):
        raise FileNotFoundError("test LibreOffice unavailable")

    monkeypatch.setattr(recalculation, "find_soffice", missing)
    status = recalculation.recalculate_workbook(tmp_path / "output.xlsx")
    assert status["status"] == "failure"
    assert not status["replaced_output"]
    assert (tmp_path / "output.xlsx").read_bytes() == original


def test_text_scoring_and_analyzer_share_remote_whole_column_bounds(tmp_path):
    save(tmp_path / "gold.xlsx", 1)
    workbook = Workbook()
    workbook.active["A1"] = 1
    workbook.active["A2"] = 9
    workbook.save(tmp_path / "output.xlsx")
    workbook.close()
    data = SpreadsheetBenchRewardData(
        cases=[WorkbookReference(number="1", path=tmp_path / "gold.xlsx")], answer_position="A:A"
    )
    assert evaluate(data, [tmp_path]).score == 1


@pytest.mark.parametrize("field", ["soffice_path", "libreoffice_python"])
def test_runtime_paths_are_not_configurable(field):
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match=field):
        SpreadsheetBenchTextActionConfig(agent=OpenAIChatConfig(model="offline"), **{field: "/external/program"})


def test_soffice_is_found_only_on_path(tmp_path, monkeypatch):
    office = tmp_path / "soffice"
    office.write_text("#!/bin/sh\nexit 0\n")
    office.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    assert recalculation.find_soffice() == office
    monkeypatch.setenv("PATH", str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError, match="not found on PATH"):
        recalculation.find_soffice()


async def test_probe_checks_path_in_controlled_python_once(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from mindmemos_skill.envs.builtin_envs.spreadsheetbench.text_action_mode import environment

    probe = AsyncMock(return_value=SimpleNamespace(returncode=0, stderr=""))
    monkeypatch.setattr(environment, "execute", probe)
    env = SpreadsheetBenchTextActionEnvironment(
        SpreadsheetBenchTextActionConfig(agent=OpenAIChatConfig(model="offline"))
    )
    await env.initialize()
    await env.initialize()
    probe.assert_awaited_once()
    command, directory, execution = probe.call_args.args
    assert command[:2] == ["python", "-c"]
    assert "shutil.which('soffice')" in command[2]
    assert "import uno" not in command[2]
    assert execution.python_command == [sys.executable]


async def test_missing_soffice_fails_before_agent(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from mindmemos_skill.envs.builtin_envs.spreadsheetbench.text_action_mode import environment

    monkeypatch.setattr(environment, "execute", AsyncMock(return_value=SimpleNamespace(returncode=1, stderr="missing")))
    env = SpreadsheetBenchTextActionEnvironment(
        SpreadsheetBenchTextActionConfig(agent=OpenAIChatConfig(model="offline"))
    )
    with pytest.raises(RuntimeError, match="PATH check failed"):
        await env.initialize()
    assert not env._office_ready


async def test_disabled_recalculation_does_not_probe_office(monkeypatch):
    from unittest.mock import AsyncMock

    from mindmemos_skill.envs.builtin_envs.spreadsheetbench.text_action_mode import environment

    probe = AsyncMock(side_effect=AssertionError("LibreOffice must not be probed"))
    monkeypatch.setattr(environment, "execute", probe)
    env = SpreadsheetBenchTextActionEnvironment(
        SpreadsheetBenchTextActionConfig(agent=OpenAIChatConfig(model="offline"), transactional_recalculation=False)
    )
    await env.initialize()
    probe.assert_not_called()


@pytest.mark.parametrize("outcome", ["success", "error", "missing_output", "timeout", "formula_changed"])
def test_conversion_checks_caches_and_preserves_original_on_failure(tmp_path, monkeypatch, outcome):
    from subprocess import CompletedProcess, TimeoutExpired
    from xml.etree import ElementTree
    from zipfile import ZipFile

    from openpyxl import load_workbook

    output = tmp_path / "output.xlsx"
    save(output, "=1+2")
    original = output.read_bytes()
    monkeypatch.setattr(recalculation, "find_soffice", lambda: tmp_path / "soffice")

    def conversion(command, **kwargs):
        assert "--convert-to" in command
        assert "--uno-worker" not in command
        source = recalculation.Path(command[-1])
        destination = recalculation.Path(command[command.index("--outdir") + 1]) / source.name
        with ZipFile(source) as archive:
            entries = {name: archive.read(name) for name in archive.namelist()}
        namespace = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
        sheet = ElementTree.fromstring(entries["xl/worksheets/sheet1.xml"])
        cell = sheet.find(f".//{namespace}c[@r='A1']")
        assert cell.find(f"{namespace}v") is None
        if outcome == "timeout":
            raise TimeoutExpired(command, 1)
        if outcome == "missing_output":
            return CompletedProcess(command, 0, "", "")
        if outcome == "formula_changed":
            save(destination, 3)
        else:
            value = ElementTree.SubElement(cell, f"{namespace}v")
            value.text = "#DIV/0!" if outcome == "error" else "3"
            cell.set("t", "e" if outcome == "error" else "n")
            entries["xl/worksheets/sheet1.xml"] = ElementTree.tostring(sheet)
            with ZipFile(destination, "w") as archive:
                for name, content in entries.items():
                    archive.writestr(name, content)
        return CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(recalculation.subprocess, "run", conversion)
    status = recalculation.recalculate_workbook(output)
    if outcome in {"success", "error"}:
        assert status["status"] == ("success" if outcome == "success" else "errors_found")
        cached = load_workbook(output, data_only=True)
        assert cached.active["A1"].value == (3 if outcome == "success" else "#DIV/0!")
        cached.close()
        assert status["replaced_output"]
    else:
        assert status["status"] == "failure"
        assert not status["replaced_output"]
        assert output.read_bytes() == original
    assert not list(tmp_path.glob(".spreadsheetbench_recalc_*"))
    assert not (tmp_path / "output.recalc.xlsx").exists()


@pytest.mark.parametrize(
    ("sheet", "answer_sheet", "position"),
    [
        ("Consolidated Tracker", "Consolidated Tracker,Existing Task,Additions,Retired", "A1"),
        ("b2b, sez, de", "'b2b, sez, de'", "'b2b, sez, de'!A1"),
    ],
)
def test_text_scoring_handles_released_worksheet_lists(tmp_path, sheet, answer_sheet, position):
    from mindmemos_skill.envs.builtin_envs.spreadsheetbench.text_action_mode.reference_reward import compare_workbooks

    for filename, value in [("gold.xlsx", 3), ("output.xlsx", 4)]:
        workbook = Workbook()
        workbook.active.title = sheet
        workbook.active["A1"] = value
        workbook.save(tmp_path / filename)
        workbook.close()
    data = SpreadsheetBenchRewardData(
        cases=[WorkbookReference(number="1", path=tmp_path / "gold.xlsx")],
        answer_position=position,
        answer_sheet=answer_sheet,
    )
    assert evaluate(data, [tmp_path]).score == 0
    assert f"value@{sheet}!A1" in evaluate(data, [tmp_path]).detail
    assert compare_workbooks(tmp_path / "gold.xlsx", tmp_path / "gold.xlsx", position, answer_sheet=answer_sheet) == (
        True,
        "",
    )


def test_text_scoring_preserves_spaces_in_default_worksheet_name(tmp_path):
    from mindmemos_skill.envs.builtin_envs.spreadsheetbench.text_action_mode.reference_reward import compare_workbooks

    workbook = Workbook()
    workbook.active.title = "SANDBOX "
    workbook.active["A1"] = 7
    workbook.save(tmp_path / "gold.xlsx")
    workbook.close()
    assert compare_workbooks(tmp_path / "gold.xlsx", tmp_path / "gold.xlsx", "A1") == (True, "")
