#!/usr/bin/env python3
"""Transactional formula recalculation for SpreadsheetBench policies."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any
from xml.dom import minidom
from zipfile import ZipFile

EXCEL_ERRORS = ("#VALUE!", "#DIV/0!", "#REF!", "#NAME?", "#NULL!", "#NUM!", "#N/A")
STATUS_FILENAME = ".tree_only_recalc_status.json"
TASK_LOCAL_HELPER_FILENAME = ".treeskill_recalculate.py"
VCL_PLUGIN_ENV = "SAL_USE_VCLPLUGIN"
SPREADSHEET_XML_NAMESPACE = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"


def stage_task_local_helper(working_dir: str | Path) -> Path:
    """Copy the standalone helper into a dependency-only task Python environment.

    Args:
        working_dir: Agent-visible directory where the helper must be created.

    Returns:
        Task-local script path, identical to this canonical implementation.

    Raises:
        FileExistsError: An existing helper has different contents.
    """

    destination = Path(working_dir).absolute() / TASK_LOCAL_HELPER_FILENAME
    script = Path(__file__).read_text(encoding="utf-8")
    if destination.exists():
        if destination.is_file() and destination.read_text(encoding="utf-8") == script:
            return destination
        raise FileExistsError(f"task-local recalculation helper already exists: {destination}")
    destination.write_text(script, encoding="utf-8")
    return destination


def append_recalculation_instructions(
    prompt: str,
    *,
    working_dir: str | Path,
    output_file: str | Path,
    python_executable: Path,
) -> str:
    """Append the released transactional recalculation instructions.

    Args:
        prompt: Public task prompt before recalculation guidance.
        working_dir: Agent-visible directory where bash commands run.
        output_file: Workbook the agent must save and recalculate.
        python_executable: Managed task interpreter used to run the standalone helper.

    Returns:
        Prompt with the exact task-local command and released status instructions.
    """

    workspace = Path(working_dir).absolute()
    helper_path = stage_task_local_helper(workspace)
    output_path = Path(output_file).absolute()
    status_path = workspace / STATUS_FILENAME
    output_argument = (
        output_path.relative_to(workspace).as_posix() if output_path.is_relative_to(workspace) else str(output_path)
    )
    command = shlex.join(
        [
            str(python_executable.absolute()),
            f"./{helper_path.name}",
            output_argument,
            "--status-path",
            status_path.name,
        ]
    )
    return (
        f"{prompt}\n\n"
        "### formula_recalculation\n"
        "If you create or modify formulas in the output workbook, save it and then "
        "run this exact command before signaling completion:\n"
        f"{command}\n\n"
        "This command is the canonical task-local recalculation command for this run. "
        "Do not substitute another helper or change its paths. A `success` or `not_needed` JSON "
        "status permits completion. If it reports `errors_found` or `failure`, inspect "
        "the reported problem, correct the workbook when possible using task-visible "
        "information, and rerun the same command."
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_url(path: Path) -> str:
    return path.resolve().as_uri()


def find_soffice() -> Path:
    """Find LibreOffice on PATH, preferring the Windows console launcher.

    Returns:
        Absolute executable path from the calling process's PATH.

    Raises:
        FileNotFoundError: No soffice command is available on PATH.
    """
    executable = shutil.which("soffice.com") if os.name == "nt" else None
    executable = executable or shutil.which("soffice")
    if executable is None:
        raise FileNotFoundError(
            "LibreOffice soffice was not found on PATH; add its program directory to the system PATH"
        )
    return Path(executable).absolute()


def _copy_for_recalculation(source: Path, destination: Path) -> None:
    """Copy the OOXML package, invalidating formula caches while preserving XML namespaces."""
    with ZipFile(source) as original, ZipFile(destination, "w") as copied:
        for entry in original.infolist():
            content = original.read(entry)
            is_sheet = entry.filename.startswith("xl/worksheets/") and entry.filename.endswith(".xml")
            if entry.filename == "xl/workbook.xml" or is_sheet:
                document = minidom.parseString(content)
                try:
                    if is_sheet:
                        for cell in document.getElementsByTagNameNS(SPREADSHEET_XML_NAMESPACE, "c"):
                            if cell.getElementsByTagNameNS(SPREADSHEET_XML_NAMESPACE, "f"):
                                for value in list(cell.getElementsByTagNameNS(SPREADSHEET_XML_NAMESPACE, "v")):
                                    cell.removeChild(value)
                                if cell.hasAttribute("t"):
                                    cell.removeAttribute("t")
                    else:
                        properties = document.getElementsByTagNameNS(SPREADSHEET_XML_NAMESPACE, "calcPr")
                        if properties:
                            properties[0].setAttribute("calcMode", "auto")
                            properties[0].setAttribute("fullCalcOnLoad", "1")
                            properties[0].setAttribute("forceFullCalc", "1")
                    content = document.toxml(encoding="utf-8")
                finally:
                    document.unlink()
            copied.writestr(entry, content)


def _workbook_formula_state(path: Path) -> dict[str, Any]:
    from openpyxl import load_workbook

    formulas: list[tuple[str, str]] = []
    workbook = load_workbook(path, data_only=False, read_only=False)
    try:
        for worksheet in workbook.worksheets:
            for row in worksheet.iter_rows():
                for cell in row:
                    if cell.data_type == "f":
                        formulas.append((worksheet.title, cell.coordinate))
    finally:
        workbook.close()

    cached_formula_values = 0
    errors: list[dict[str, str]] = []
    cached = load_workbook(path, data_only=True, read_only=False)
    try:
        for sheet_name, coordinate in formulas:
            value = cached[sheet_name][coordinate].value
            if value is not None:
                cached_formula_values += 1
            if isinstance(value, str):
                for excel_error in EXCEL_ERRORS:
                    if excel_error in value:
                        errors.append(
                            {
                                "sheet": sheet_name,
                                "cell": coordinate,
                                "error": excel_error,
                            }
                        )
                        break
    finally:
        cached.close()

    return {
        "formula_count": len(formulas),
        "cached_formula_value_count": cached_formula_values,
        "excel_errors": errors,
    }


def _write_status(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def recalculate_workbook(
    workbook: Path,
    *,
    status_path: Path | None = None,
    timeout: int = 90,
) -> dict[str, Any]:
    """Recalculate a temporary copy with LibreOffice from PATH, then replace the original.

    Args:
        workbook: Agent-produced workbook to recalculate in place.
        status_path: JSON report destination, defaulting beside the workbook.
        timeout: Maximum LibreOffice conversion duration in seconds.

    Returns:
        Status report with success, not_needed, errors_found, or failure, formula counts,
        cache counts, replacement status, runtime paths, and failure details when applicable.
    """

    workbook = workbook.expanduser().resolve()
    status_path = (status_path or workbook.parent / STATUS_FILENAME).expanduser().resolve()
    status: dict[str, Any] = {
        "method": "libreoffice_agent_recalculation",
        "workbook": str(workbook),
        "status": "failure",
        "attempted": False,
        "replaced_output": False,
    }
    if not workbook.is_file():
        status["error"] = f"workbook not found: {workbook}"
        _write_status(status_path, status)
        return status

    work_root: Path | None = None
    office_stdout = ""
    office_stderr = ""

    try:
        before = _workbook_formula_state(workbook)
        status.update(
            {
                "sha256_before": _sha256(workbook),
                "formula_count_before": before["formula_count"],
                "cached_formula_value_count_before": before["cached_formula_value_count"],
            }
        )
        if before["formula_count"] == 0:
            status.update({"status": "not_needed", "error": ""})
            return status

        soffice = find_soffice()
        status.update({"attempted": True, "soffice": str(soffice), "engine": "libreoffice_headless_conversion"})
        work_root = Path(tempfile.mkdtemp(prefix=".spreadsheetbench_recalc_", dir=workbook.parent))
        profile_dir = work_root / "profile"
        profile_dir.mkdir()
        temporary_workbook = work_root / workbook.name
        _copy_for_recalculation(workbook, temporary_workbook)
        converted_dir = work_root / "converted"
        converted_dir.mkdir()
        converted_workbook = converted_dir / workbook.name
        office_environment = os.environ.copy()
        office_environment[VCL_PLUGIN_ENV] = "svp"
        office = subprocess.run(
            [
                str(soffice),
                f"-env:UserInstallation={_file_url(profile_dir)}",
                "--headless",
                "--nologo",
                "--nodefault",
                "--norestore",
                "--nofirststartwizard",
                "--convert-to",
                workbook.suffix.lstrip("."),
                "--outdir",
                str(converted_dir),
                str(temporary_workbook),
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=office_environment,
        )
        office_stdout, office_stderr = office.stdout, office.stderr
        if office.returncode != 0 or not converted_workbook.is_file():
            detail = office.stderr.strip() or office.stdout.strip() or "converted workbook was not created"
            raise RuntimeError(f"LibreOffice conversion failed with exit {office.returncode}: {detail}")
        after = _workbook_formula_state(converted_workbook)
        if after["formula_count"] != before["formula_count"]:
            raise RuntimeError(
                f"formula count changed during recalculation: {before['formula_count']} -> {after['formula_count']}"
            )

        os.replace(converted_workbook, workbook)
        status.update(
            {
                "status": "errors_found" if after["excel_errors"] else "success",
                "error": "",
                "replaced_output": True,
                "sha256_after": _sha256(workbook),
                "formula_count_after": after["formula_count"],
                "cached_formula_value_count_after": after["cached_formula_value_count"],
                "excel_errors": after["excel_errors"],
            }
        )
        return status
    except Exception as exc:
        status["error"] = str(exc)
        return status
    finally:
        if status["status"] == "failure":
            status["libreoffice_stdout"] = office_stdout.strip()
            status["libreoffice_stderr"] = office_stderr.strip()
        if work_root is not None:
            shutil.rmtree(work_root, ignore_errors=True)
        _write_status(status_path, status)


def build_arg_parser() -> argparse.ArgumentParser:
    """Return the standalone helper command options."""
    parser = argparse.ArgumentParser(description="Recalculate formulas for the TreeSkill spreadsheet policy")
    parser.add_argument("workbook", nargs="?", type=Path)
    parser.add_argument("--status-path", type=Path, default=None)
    parser.add_argument("--timeout", type=int, default=90)
    return parser


def main() -> int:
    """Run the task helper and return its exit status."""
    args = build_arg_parser().parse_args()
    if args.workbook is None:
        raise SystemExit("workbook path is required")
    result = recalculate_workbook(
        args.workbook,
        status_path=args.status_path,
        timeout=args.timeout,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] in {"success", "not_needed"} else 1


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "STATUS_FILENAME",
    "append_recalculation_instructions",
    "find_soffice",
    "recalculate_workbook",
]
