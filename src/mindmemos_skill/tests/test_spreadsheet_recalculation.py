"""Verify algorithm-independent SpreadsheetBench recalculation configuration."""

import json

import openpyxl
from mindmemos_skill.envs.builtin_envs.spreadsheetbench import recalculation
from mindmemos_skill.envs.builtin_envs.spreadsheetbench.constants import LIBREOFFICE_PYTHON_ENV, SOFFICE_PATH_ENV


def executable(path):
    """Create a local executable fixture without running it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(0o755)
    return path


def test_recalculation_resolves_environment_and_explicit_overrides(tmp_path, monkeypatch):
    configured_office = executable(tmp_path / "configured/soffice")
    configured_python = executable(tmp_path / "configured/python")
    explicit_office = executable(tmp_path / "explicit/soffice")
    explicit_python = executable(tmp_path / "explicit/python")
    monkeypatch.setenv(SOFFICE_PATH_ENV, str(configured_office))
    monkeypatch.setenv(LIBREOFFICE_PYTHON_ENV, str(configured_python))
    assert recalculation._find_libreoffice_programs() == (configured_office, configured_python)
    assert recalculation._find_libreoffice_programs(
        soffice_path=str(explicit_office), libreoffice_python=str(explicit_python)
    ) == (explicit_office, explicit_python)


def test_recalculation_discovers_wrapper_by_layout(tmp_path, monkeypatch):
    launcher = executable(tmp_path / "tools/custom-office-launcher")
    program = launcher.parent / recalculation.LIBREOFFICE_PROGRAM_RELATIVE_PATH
    monkeypatch.setattr(recalculation.shutil, "which", lambda name: str(launcher))
    assert recalculation._program_dir_from_soffice_wrapper() is None
    program.mkdir(parents=True)
    assert recalculation._program_dir_from_soffice_wrapper() == program
    monkeypatch.setattr(recalculation.shutil, "which", lambda name: None)
    assert recalculation._program_dir_from_soffice_wrapper() is None


def test_recalculation_uses_environment_owned_artifact_names(tmp_path):
    helper = recalculation.stage_task_local_helper(tmp_path)
    assert helper.name == ".spreadsheetbench_recalculate.py"
    assert recalculation.stage_task_local_helper(tmp_path) == helper
    assert "mindmemos_skill.envs.builtin_envs.spreadsheetbench.recalculation" in helper.read_text()
    workbook = tmp_path / "output.xlsx"
    book = openpyxl.Workbook()
    book.active["A1"] = 42
    book.save(workbook)
    book.close()
    before = workbook.read_bytes()
    result = recalculation.recalculate_workbook(workbook)
    assert result["status"] == "not_needed"
    assert result["method"] == "libreoffice_transactional_recalculation"
    assert workbook.read_bytes() == before
    assert json.loads((tmp_path / ".spreadsheetbench_recalc_status.json").read_text()) == result
    assert "LibreOffice" in recalculation.build_arg_parser().format_help()
