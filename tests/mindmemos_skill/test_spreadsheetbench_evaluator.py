"""Regression coverage for SpreadsheetBench answer-range compatibility."""

from pathlib import Path

import pytest
from mindmemos_skill.envs.registered_envs.spreadsheetbench.evaluator import compare_workbooks

openpyxl = pytest.importorskip("openpyxl")


def _workbook(path: Path, sheet: str = "Answer", value: int = 7):
    workbook = openpyxl.Workbook()
    workbook.active.title = "Unrelated"
    target = workbook.create_sheet(sheet)
    target["A1"] = 1
    target["B2"] = value
    workbook.save(path)
    workbook.close()


@pytest.mark.parametrize("position", ["A:B", "1:2", "$A$1:$B$2", "$B$2", "A1,B2"])
def test_answer_ranges_use_answer_sheet_and_find_mismatch(tmp_path, position):
    golden, output = tmp_path / "golden.xlsx", tmp_path / "output.xlsx"
    _workbook(golden)
    _workbook(output)
    assert compare_workbooks(golden, output, position, answer_sheet="Answer") == (True, "")
    _workbook(output, value=8)
    correct, detail = compare_workbooks(golden, output, position, answer_sheet="Answer")
    assert not correct
    assert "Answer!B2" in detail


@pytest.mark.parametrize(
    "sheet,position",
    [
        ("Sales, Q1", "'Sales, Q1'!A1,'Sales, Q1'!B2"),
        ("Sheet3", "Sheet3'!A:B"),
        ("Team's data", '"Team\'s data"!A1:B2'),
    ],
)
def test_qualified_ranges_handle_upstream_sheet_names(tmp_path, sheet, position):
    golden, output = tmp_path / "golden.xlsx", tmp_path / "output.xlsx"
    _workbook(golden, sheet=sheet)
    _workbook(output, sheet=sheet, value=8)
    correct, detail = compare_workbooks(golden, output, position)
    assert not correct
    assert f"{sheet}!B2" in detail


def test_whole_column_checks_extra_output_rows(tmp_path):
    golden, output = tmp_path / "golden.xlsx", tmp_path / "output.xlsx"
    _workbook(golden)
    _workbook(output)
    workbook = openpyxl.load_workbook(output)
    workbook["Answer"]["B3"] = 9
    workbook.save(output)
    workbook.close()
    correct, detail = compare_workbooks(golden, output, "B:B", answer_sheet="Answer")
    assert not correct
    assert "Answer!B3" in detail


def test_workbook_is_closed_when_output_cannot_be_loaded(tmp_path, monkeypatch):
    golden, output = tmp_path / "golden.xlsx", tmp_path / "output.xlsx"
    _workbook(golden)
    output.write_text("not a workbook", encoding="utf-8")
    opened = []
    load = openpyxl.load_workbook

    def tracked_load(*args, **kwargs):
        workbook = load(*args, **kwargs)
        opened.append(workbook)
        return workbook

    monkeypatch.setattr(openpyxl, "load_workbook", tracked_load)
    correct, detail = compare_workbooks(golden, output, "A1")
    assert not correct and detail.startswith("load error:")
    assert len(opened) == 1 and opened[0]._archive.fp is None
