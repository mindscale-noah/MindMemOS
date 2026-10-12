"""Check SpreadsheetBench defaults and real LibreOffice recalculation without a model."""

import argparse
import asyncio
import json
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

from mindmemos_skill.agents import OpenAIChatConfig
from mindmemos_skill.envs.builtin_envs.spreadsheetbench import (
    SpreadsheetBenchTextActionConfig,
    SpreadsheetBenchToolCallingConfig,
)
from mindmemos_skill.envs.builtin_envs.spreadsheetbench.reward import evaluate
from mindmemos_skill.envs.builtin_envs.spreadsheetbench.typing import SpreadsheetBenchRewardData, WorkbookReference
from openpyxl import Workbook, load_workbook
from openpyxl.workbook.properties import CalcProperties


def _write_workbook(path: Path, *, formulas: bool, error: bool = False, stale_cache: bool = False) -> None:
    """Create a small workbook with dependent and cross-sheet formulas or saved values."""
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Results"
    sheet["A1"], sheet["A2"] = 1, 2
    sheet["B1"] = "=SUM(A1:A2)" if formulas else 3
    sheet["B2"] = "=B1*10" if formulas else 30
    sheet["B3"] = "=Data!A1+B2" if formulas else 37
    sheet["B4"] = '=IF(B3=37,"ok","bad")' if formulas else "ok"
    if error:
        sheet["C1"] = "=1/0"
    workbook.create_sheet("Data")["A1"] = 7
    if stale_cache:
        workbook.calculation = CalcProperties(calcMode="manual", fullCalcOnLoad=False, forceFullCalc=False)
    workbook.save(path)
    workbook.close()
    if stale_cache:
        with ZipFile(path) as archive:
            entries = {name: archive.read(name) for name in archive.namelist()}
        namespace = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
        sheet_xml = ElementTree.fromstring(entries["xl/worksheets/sheet1.xml"])
        for cell in sheet_xml.iter(f"{namespace}c"):
            if cell.find(f"{namespace}f") is not None:
                cell.find(f"{namespace}v").text = "999"
        entries["xl/worksheets/sheet1.xml"] = ElementTree.tostring(sheet_xml)
        with ZipFile(path, "w") as archive:
            for name, content in entries.items():
                archive.writestr(name, content)


async def main() -> None:
    """Run real scoring checks and save a JSON report with all generated workbooks.

    Raises:
        AssertionError: Defaults, cached results, selected files, or source preservation differ.
        RuntimeError: LibreOffice fails to recalculate a formula workbook.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New directory for check artifacts")
    root = parser.parse_args().output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    gold = root / "gold.xlsx"
    _write_workbook(gold, formulas=False)
    data = SpreadsheetBenchRewardData(cases=[WorkbookReference(number="1", path=gold)], answer_position="Results!B1:B4")
    agent = OpenAIChatConfig(model="offline")
    tool = SpreadsheetBenchToolCallingConfig(agent=agent)
    text = SpreadsheetBenchTextActionConfig(agent=agent)
    assert tool.transactional_recalculation is False
    assert text.transactional_recalculation is True
    scenarios = [
        (
            "tool_calling_conversion",
            SpreadsheetBenchToolCallingConfig(agent=agent, transactional_recalculation=True),
            True,
            False,
            1,
            "success",
        ),
        ("tool_calling_default", tool, True, False, 0, "disabled"),
        (
            "tool_calling_enabled",
            SpreadsheetBenchToolCallingConfig(agent=agent, transactional_recalculation=True),
            True,
            False,
            1,
            "success",
        ),
        (
            "no_formulas",
            SpreadsheetBenchToolCallingConfig(agent=agent, transactional_recalculation=True),
            False,
            False,
            1,
            "not_needed",
        ),
        (
            "error_outside_answer",
            SpreadsheetBenchToolCallingConfig(agent=agent, transactional_recalculation=True),
            True,
            True,
            1,
            "errors_found",
        ),
        (
            "manual_stale_cache",
            SpreadsheetBenchToolCallingConfig(agent=agent, transactional_recalculation=True),
            True,
            False,
            1,
            "success",
        ),
    ]
    rows = []
    for name, config, formulas, error, expected_score, expected_status in scenarios:
        directory = root / name
        directory.mkdir()
        source = directory / "output.xlsx"
        _write_workbook(source, formulas=formulas, error=error, stale_cache=name == "manual_stale_cache")
        original = source.read_bytes()
        reward = await evaluate(data, [directory], transactional_recalculation=config.transactional_recalculation)
        case = reward.cases[0]
        recalculated = directory / "output.recalc.xlsx"
        needs_output = expected_status in {"success", "errors_found"}
        assert source.read_bytes() == original, f"Source changed: {name}"
        assert reward.score == expected_score, f"Unexpected score: {name}"
        assert case["recalculation"]["status"] == expected_status, f"Unexpected status: {name}"
        assert recalculated.exists() == needs_output, f"Unexpected recalculated output: {name}"
        assert case["scored_file"] == ("output.recalc.xlsx" if needs_output else "output.xlsx")
        cached = load_workbook(directory / case["scored_file"], data_only=True)
        values = [cached["Results"][f"B{index}"].value for index in range(1, 5)]
        cached.close()
        assert values == ([None] * 4 if expected_status == "disabled" else [3, 30, 37, "ok"])
        rows.append(
            {
                "scenario": name,
                "recalculation_enabled": config.transactional_recalculation,
                "score": reward.score,
                "scored_file": case["scored_file"],
                "recalculation": case["recalculation"],
                "cached_values": values,
                "source_unchanged": source.read_bytes() == original,
            }
        )
    report = {"status": "passed", "checks": rows}
    (root / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
