"""Tests for SpreadsheetBench evaluation helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from mindmemos_eval.skills.agents import ReactAgentTools
from mindmemos_eval.skills.envs.spreadsheetbench import (
    SpreadsheetBenchEnv,
    compare_cell_value,
    compare_workbooks,
    generate_cell_names,
)

pytest.importorskip("openpyxl")

SKILL_RL_SPREADSHEET_SYSTEM_PROMPT = (
    "You are an expert spreadsheet assistant. Your working directory contains a "
    "source Excel file named 'input.xlsx'. Do NOT modify 'input.xlsx'. Instead, "
    "produce a new file 'output.xlsx' in the same directory that fully satisfies "
    "the user's request (start from a copy of 'input.xlsx' and apply your changes).\n"
    "Work by writing and running Python (openpyxl is available) through the shell "
    "tool — do not answer from memory. Inspect the sheets first, apply the changes, "
    "save to 'output.xlsx', and verify.\n"
    "IMPORTANT: 'output.xlsx' is graded by reading cached cell VALUES, with no "
    "formula recalculation. Write the final computed values into the target cells "
    "(not bare formulas), since an unevaluated formula reads back as empty.\n"
    "When you are done, stop without calling any tool."
)


def _make_workbook(path: Path, value: Any) -> None:
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    ws["A1"] = value
    wb.save(path)
    wb.close()


def _make_dataset(root: Path) -> Path:
    data_root = root / "SpreadsheetBench"
    sheet_dir = data_root / "spreadsheetbench_verified_400" / "spreadsheet" / "1"
    split_dir = data_root / "spreadsheetbench_id_split" / "test"
    sheet_dir.mkdir(parents=True)
    split_dir.mkdir(parents=True)

    _make_workbook(sheet_dir / "1_init.xlsx", 1)
    _make_workbook(sheet_dir / "1_golden.xlsx", 42)
    (data_root / "spreadsheetbench_verified_400" / "dataset.json").write_text(
        json.dumps(
            [
                {
                    "id": "1",
                    "instruction": "Set A1 to 42.",
                    "spreadsheet_path": "spreadsheet/1",
                    "answer_position": "A1",
                    "answer_sheet": "Sheet1",
                    "instruction_type": "unit",
                }
            ]
        ),
        encoding="utf-8",
    )
    (split_dir / "items.json").write_text(json.dumps([{"id": "1"}]), encoding="utf-8")
    return data_root


def _make_all_dataset(root: Path, size: int = 3) -> Path:
    data_root = root / "SpreadsheetBench"
    verified = data_root / "spreadsheetbench_verified_400"
    records = []
    for index in range(1, size + 1):
        sheet_dir = verified / "spreadsheet" / str(index)
        sheet_dir.mkdir(parents=True)
        _make_workbook(sheet_dir / f"{index}_init.xlsx", index)
        _make_workbook(sheet_dir / f"{index}_golden.xlsx", index)
        records.append(
            {
                "id": str(index),
                "instruction": f"Leave A1 as {index}.",
                "spreadsheet_path": f"spreadsheet/{index}",
                "answer_position": "A1",
                "answer_sheet": "Sheet1",
                "instruction_type": "unit",
            }
        )
    (verified / "dataset.json").write_text(json.dumps(records), encoding="utf-8")
    return data_root


def _make_mixed_id_dataset(root: Path) -> Path:
    data_root = root / "SpreadsheetBench"
    verified = data_root / "spreadsheetbench_verified_400"
    records = []
    for record_id in ["10", "task_a", "2"]:
        sheet_dir = verified / "spreadsheet" / record_id
        sheet_dir.mkdir(parents=True)
        _make_workbook(sheet_dir / f"{record_id}_init.xlsx", 1)
        _make_workbook(sheet_dir / f"{record_id}_golden.xlsx", 1)
        records.append(
            {
                "id": record_id,
                "instruction": "Leave A1 unchanged.",
                "spreadsheet_path": f"spreadsheet/{record_id}",
                "answer_position": "A1",
                "answer_sheet": "Sheet1",
                "instruction_type": "unit",
            }
        )
    (verified / "dataset.json").write_text(json.dumps(records), encoding="utf-8")
    return data_root


def test_compare_workbooks_uses_spreadsheetbench_value_semantics(tmp_path: Path) -> None:
    gt = tmp_path / "gt.xlsx"
    pred = tmp_path / "pred.xlsx"
    _make_workbook(gt, 1.234)
    _make_workbook(pred, "1.23")

    ok, message = compare_workbooks(gt, pred, "Sheet1!A1")

    assert ok
    assert message == ""
    assert compare_cell_value("", None)
    assert generate_cell_names("A1:B2") == ["A1", "A2", "B1", "B2"]


def test_agent_tools_keep_access_inside_workdir(tmp_path: Path) -> None:
    tools = ReactAgentTools(tmp_path)

    assert "Successfully wrote" in tools.write("note.txt", "hello")
    assert tools.read("note.txt") == "hello"
    assert "outside the working directory" in tools.read("../secret.txt")
    assert "installing packages is disabled" in tools.shell(["pip install nope"])


class _CopyingAgentFactory:
    """Build a deterministic workbook agent for runner integration tests."""

    def __init__(self, *, value: int | None = None) -> None:
        self.value = value

    def build(self, workdir: Path, system_prompt: str) -> tuple[Any, None]:
        import shutil

        from mindmemos_eval.skills.agents import RunResult

        del system_prompt
        value = self.value

        class Agent:
            async def run(self, messages: list[dict[str, Any]]) -> RunResult:
                import openpyxl

                shutil.copyfile(workdir / "input.xlsx", workdir / "output.xlsx")
                if value is not None:
                    workbook = openpyxl.load_workbook(workdir / "output.xlsx")
                    workbook.active["A1"] = value
                    workbook.save(workdir / "output.xlsx")
                    workbook.close()
                return RunResult(messages=messages, turns=1, finished=True)

        return Agent(), None


@pytest.mark.asyncio
async def test_spreadsheetbench_runner_scores_one_case(tmp_path: Path) -> None:
    from mindmemos_eval.skills.runners import SkillEvalRunner, SkillRunConfig

    data_root = _make_dataset(tmp_path)
    run_dir = tmp_path / "run"
    trajectory_path = tmp_path / "custom" / "traces.jsonl"
    env = SpreadsheetBenchEnv(data_root=data_root, run_dir=run_dir, trajectory_path=trajectory_path)

    result = await SkillEvalRunner(env, _CopyingAgentFactory(value=42), SkillRunConfig(split="test")).run()

    assert result.total == 1
    assert result.correct == 1
    assert result.accuracy == 1.0
    assert (run_dir / "test_summary.json").exists()
    assert Path(result.results[0].workdir, "output.xlsx").exists()
    assert json.loads(trajectory_path.read_text(encoding="utf-8").splitlines()[0])["messages"]


@pytest.mark.asyncio
async def test_spreadsheetbench_runner_scores_concurrent_cases(tmp_path: Path) -> None:
    from mindmemos_eval.skills.runners import SkillEvalRunner, SkillRunConfig

    data_root = _make_all_dataset(tmp_path, size=3)
    run_dir = tmp_path / "run"
    env = SpreadsheetBenchEnv(data_root=data_root, run_dir=run_dir)

    result = await SkillEvalRunner(env, _CopyingAgentFactory(), SkillRunConfig(concurrency=2)).run()

    assert result.total == 3
    assert result.correct == 3
    assert result.accuracy == 1.0
    assert (run_dir / "all_trajectories.jsonl").exists()


def test_spreadsheetbench_env_loads_all_cases_without_split(tmp_path: Path) -> None:
    data_root = _make_all_dataset(tmp_path, size=3)

    env = SpreadsheetBenchEnv(data_root=data_root, run_dir=tmp_path / "run")
    cases = env.load_cases("all")
    shuffled_a = env.load_cases("all")
    shuffled_b = env.load_cases("all")

    import random

    random.Random(1447).shuffle(shuffled_a)
    random.Random(1447).shuffle(shuffled_b)

    assert [case.id for case in cases] == ["1", "2", "3"]
    assert [case.id for case in shuffled_a] == [case.id for case in shuffled_b]
    assert [case.id for case in shuffled_a] != ["1", "2", "3"]


def test_spreadsheetbench_env_sorts_mixed_ids(tmp_path: Path) -> None:
    data_root = _make_mixed_id_dataset(tmp_path)

    env = SpreadsheetBenchEnv(data_root=data_root, run_dir=tmp_path / "run")

    assert [case.id for case in env.load_cases("all")] == ["2", "10", "task_a"]


def test_spreadsheetbench_system_prompt_matches_skill_rl_baseline(tmp_path: Path) -> None:
    data_root = _make_all_dataset(tmp_path, size=1)

    env = SpreadsheetBenchEnv(data_root=data_root, run_dir=tmp_path / "run")

    assert env.system_prompt() == SKILL_RL_SPREADSHEET_SYSTEM_PROMPT
