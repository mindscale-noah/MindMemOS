"""Verify algorithm-independent SpreadsheetBench recalculation configuration."""

import json
import shutil
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock
from xml.etree import ElementTree
from zipfile import ZipFile

import openpyxl
import pytest
from mindmemos_skill.agents import OpenAIChatConfig
from mindmemos_skill.envs.builtin_envs.spreadsheetbench import (
    SpreadsheetBenchTextActionConfig,
    SpreadsheetBenchToolCallingConfig,
    recalculation,
)
from mindmemos_skill.envs.builtin_envs.spreadsheetbench.recalculation import SOFFICE_PATH_ENV
from mindmemos_skill.envs.builtin_envs.spreadsheetbench.reward import evaluate
from mindmemos_skill.envs.builtin_envs.spreadsheetbench.typing import SpreadsheetBenchRewardData, WorkbookReference
from openpyxl.worksheet.formula import ArrayFormula


def write_workbook(path, cells):
    book = openpyxl.Workbook()
    for coordinate, value in cells.items():
        book.active[coordinate] = value
    book.save(path)
    book.close()


def cache_formula(path, value):
    """Write a saved formula result as a spreadsheet engine would, retaining the formula."""
    with ZipFile(path) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    namespace = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    sheet = ElementTree.fromstring(entries["xl/worksheets/sheet1.xml"])
    cell = sheet.find(f".//{namespace}c[@r='A1']")
    cached = cell.find(f"{namespace}v")
    if cached is None:
        cached = ElementTree.SubElement(cell, f"{namespace}v")
    cached.text = str(value)
    cell.set("t", "e" if isinstance(value, str) else "n")
    entries["xl/worksheets/sheet1.xml"] = ElementTree.tostring(sheet)
    with ZipFile(path, "w") as archive:
        for name, content in entries.items():
            archive.writestr(name, content)


def fake_office(monkeypatch, value=2, failure=None):
    """Replace only the office transaction with a deterministic cache-writing worker."""
    monkeypatch.setattr(recalculation, "_find_soffice", lambda **kwargs: Path("soffice"))

    def worker(command, workspace, config):
        if failure:
            raise failure
        source = Path(command[-1])
        converted = Path(command[command.index("--outdir") + 1]) / source.name
        shutil.copyfile(source, converted)
        cache_formula(converted, value)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(recalculation, "execute", AsyncMock(side_effect=worker))


def scoring_data(tmp_path, position="A1", expected=2):
    gold = tmp_path / "gold.xlsx"
    write_workbook(gold, {position: expected})
    return SpreadsheetBenchRewardData(cases=[WorkbookReference(number="1", path=gold)], answer_position=position)


def executable(path):
    """Create a local executable fixture without running it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(0o755)
    return path


def test_protocol_recalculation_defaults_and_explicit_overrides():
    agent = OpenAIChatConfig(model="offline")
    assert SpreadsheetBenchToolCallingConfig(agent=agent).transactional_recalculation is False
    assert SpreadsheetBenchTextActionConfig(agent=agent).transactional_recalculation is True
    assert SpreadsheetBenchToolCallingConfig(agent=agent, transactional_recalculation=True).transactional_recalculation
    assert not SpreadsheetBenchTextActionConfig(
        agent=agent, transactional_recalculation=False
    ).transactional_recalculation


def test_recalculation_resolves_environment_and_explicit_overrides(tmp_path, monkeypatch):
    configured_office = executable(tmp_path / "configured/soffice")
    explicit_office = executable(tmp_path / "explicit/soffice")
    monkeypatch.setenv(SOFFICE_PATH_ENV, str(configured_office))
    assert recalculation._find_soffice() == configured_office
    assert recalculation._find_soffice(soffice_path=str(explicit_office)) == explicit_office


def test_recalculation_discovers_wrapper_by_layout(tmp_path, monkeypatch):
    launcher = executable(tmp_path / "tools/custom-office-launcher")
    program = launcher.parent / recalculation.LIBREOFFICE_PROGRAM_RELATIVE_PATH
    monkeypatch.setattr(recalculation.shutil, "which", lambda name: str(launcher))
    assert recalculation._program_dir_from_soffice_wrapper() is None
    program.mkdir(parents=True)
    assert recalculation._program_dir_from_soffice_wrapper() == program
    monkeypatch.setattr(recalculation.shutil, "which", lambda name: None)
    assert recalculation._program_dir_from_soffice_wrapper() is None


async def test_recalculation_uses_environment_owned_artifact_names(tmp_path):
    workbook = tmp_path / "output.xlsx"
    book = openpyxl.Workbook()
    book.active["A1"] = 42
    book.save(workbook)
    book.close()
    before = workbook.read_bytes()
    (tmp_path / "output.recalc.xlsx").write_bytes(b"stale previous result")
    result = await recalculation.recalculate_workbook(workbook)
    assert result["status"] == "not_needed"
    assert result["method"] == "libreoffice_transactional_recalculation"
    assert workbook.read_bytes() == before
    assert not (tmp_path / "output.recalc.xlsx").exists()
    assert not (tmp_path / ".spreadsheetbench_recalculate.py").exists()
    assert json.loads((tmp_path / ".spreadsheetbench_recalc_status.json").read_text()) == result
    assert "LibreOffice" in recalculation.build_arg_parser().format_help()


@pytest.mark.parametrize("expression", ["=1+1", ArrayFormula(ref="A1:A2", text="=SEQUENCE(2,1,2)")])
async def test_evaluation_recalculates_separate_output_and_scores_cached_values(tmp_path, monkeypatch, expression):
    data = scoring_data(tmp_path)
    output = tmp_path / "output.xlsx"
    write_workbook(output, {"A1": expression})
    before = output.read_bytes()
    fake_office(monkeypatch)
    result = await evaluate(data, [tmp_path])
    assert result.success and result.score == 1
    assert result.cases[0]["scored_file"] == "output.recalc.xlsx"
    assert result.cases[0]["recalculation"]["status"] == "success"
    assert output.read_bytes() == before
    formula = openpyxl.load_workbook(tmp_path / "output.recalc.xlsx")
    assert formula.active["A1"].data_type == "f"
    formula.close()
    assert (tmp_path / recalculation.STATUS_FILENAME).is_file()
    assert not [path for path in tmp_path.glob(".spreadsheetbench_recalc_*") if path.is_dir()]


async def test_evaluation_without_formulas_skips_office_and_ignores_stale_output(tmp_path, monkeypatch):
    data = scoring_data(tmp_path)
    write_workbook(tmp_path / "output.xlsx", {"A1": 2})
    write_workbook(tmp_path / "output.recalc.xlsx", {"A1": 999})
    monkeypatch.setattr(recalculation, "_find_soffice", lambda **kwargs: pytest.fail("office must not start"))
    result = await evaluate(data, [tmp_path])
    assert result.success and result.cases[0]["scored_file"] == "output.xlsx"
    assert result.cases[0]["recalculation"]["status"] == "not_needed"
    assert not (tmp_path / "output.recalc.xlsx").exists()


@pytest.mark.parametrize("failure", [RuntimeError("office unavailable"), subprocess.TimeoutExpired("office", 90)])
async def test_evaluation_recalculation_failure_preserves_source_and_does_not_score_stale_file(
    tmp_path, monkeypatch, failure
):
    data = scoring_data(tmp_path)
    output = tmp_path / "output.xlsx"
    write_workbook(output, {"A1": "=1+1"})
    write_workbook(tmp_path / "output.recalc.xlsx", {"A1": 2})
    before = output.read_bytes()
    fake_office(monkeypatch, failure=failure)
    with pytest.raises(RuntimeError, match="recalculation failed"):
        (await evaluate(data, [tmp_path]))
    assert output.read_bytes() == before
    assert not (tmp_path / "output.recalc.xlsx").exists()
    assert json.loads((tmp_path / recalculation.STATUS_FILENAME).read_text())["status"] == "failure"


@pytest.mark.parametrize("position,expected,passed", [("A1", 2, False), ("B1", 42, True)])
async def test_formula_errors_are_scored_only_in_the_answer_range(tmp_path, monkeypatch, position, expected, passed):
    data = scoring_data(tmp_path, position, expected)
    write_workbook(tmp_path / "output.xlsx", {"A1": "=1/0", "B1": 42})
    fake_office(monkeypatch, value="#DIV/0!")
    result = await evaluate(data, [tmp_path])
    assert result.success is passed
    assert result.cases[0]["scored_file"] == "output.recalc.xlsx"
    assert result.cases[0]["recalculation"]["status"] == "errors_found"


async def test_disabled_recalculation_scores_original_without_reusing_recalculated_file(tmp_path, monkeypatch):
    data = scoring_data(tmp_path)
    write_workbook(tmp_path / "output.xlsx", {"A1": "=1+1"})
    write_workbook(tmp_path / "output.recalc.xlsx", {"A1": 2})
    monkeypatch.setattr(recalculation, "_find_soffice", lambda **kwargs: pytest.fail("office must not start"))
    result = await evaluate(data, [tmp_path], transactional_recalculation=False)
    assert not result.success and result.cases[0]["scored_file"] == "output.xlsx"


async def test_recalculation_rejects_source_alias_and_output_symlink(tmp_path):
    output = tmp_path / "output.xlsx"
    write_workbook(output, {"A1": "=1+1"})
    with pytest.raises(ValueError, match="separate file"):
        (await recalculation.recalculate_workbook(output, output_path=output))
    (tmp_path / "output.recalc.xlsx").symlink_to(output)
    with pytest.raises(ValueError, match="symbolic link"):
        (await recalculation.recalculate_workbook(output))


@pytest.mark.parametrize("returncode", [0, 1])
async def test_conversion_without_output_is_a_failure_even_with_zero_exit(tmp_path, monkeypatch, returncode):
    source = tmp_path / "output.xlsx"
    write_workbook(source, {"A1": "=1+1"})
    original = source.read_bytes()
    fake_office(monkeypatch)
    monkeypatch.setattr(
        recalculation,
        "execute",
        AsyncMock(
            side_effect=lambda command, workspace, config: subprocess.CompletedProcess(command, returncode, "", "")
        ),
    )
    result = await recalculation.recalculate_workbook(source)
    assert result["status"] == "failure"
    assert "conversion failed" in result["error"]
    assert source.read_bytes() == original
    assert not (tmp_path / "output.recalc.xlsx").exists()


def test_recalculation_copy_invalidates_stale_caches_and_preserves_source(tmp_path):
    source, copied = tmp_path / "output.xlsx", tmp_path / "temporary.xlsx"
    write_workbook(source, {"A1": "=1+1", "B1": 42})
    cache_formula(source, 999)
    original = source.read_bytes()
    recalculation._copy_for_recalculation(source, copied)
    assert source.read_bytes() == original
    cached = openpyxl.load_workbook(copied, data_only=True)
    assert cached.active["A1"].value is None
    assert cached.active["B1"].value == 42
    cached.close()
    formulas = openpyxl.load_workbook(copied, data_only=False)
    assert formulas.active["A1"].value == "=1+1"
    assert formulas.calculation.calcMode == "auto"
    formulas.close()
