#!/usr/bin/env python3
"""Recalculate formula outputs into separate workbooks for SpreadsheetBench evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from ....infra.execution import ExecutionConfig, execute
from .recalculation_cli import _copy_for_recalculation

SOFFICE_PATH_ENV = "SPREADSHEETBENCH_SOFFICE_PATH"
"""Explicit path to the LibreOffice binary."""

EXCEL_ERRORS = ("#VALUE!", "#DIV/0!", "#REF!", "#NAME?", "#NULL!", "#NUM!", "#N/A")
LIBREOFFICE_PROGRAM_RELATIVE_PATH = Path("libreoffice-appimage/squashfs-root/opt/libreoffice24.2/program")
STATUS_FILENAME = ".spreadsheetbench_recalc_status.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_url(path: Path) -> str:
    return path.resolve().as_uri()


def _local_program_dir() -> Path:
    source = Path(__file__).resolve()
    for parent in source.parents:
        candidate = parent / "tools" / LIBREOFFICE_PROGRAM_RELATIVE_PATH
        if candidate.is_dir():
            return candidate
    return source.parent / "tools" / LIBREOFFICE_PROGRAM_RELATIVE_PATH


def _program_dir_from_soffice_wrapper() -> Path | None:
    executable = shutil.which("soffice")
    if not executable:
        return None
    resolved = Path(executable).resolve()
    candidate = resolved.parent / LIBREOFFICE_PROGRAM_RELATIVE_PATH
    return candidate if candidate.is_dir() else None


def _first_executable(candidates: list[str | Path | None], label: str) -> Path:
    checked: list[str] = []
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser().resolve()
        checked.append(str(path))
        if path.is_file() and os.access(path, os.X_OK):
            return path
    raise FileNotFoundError(f"{label} not found or not executable; checked: {checked}")


def _find_soffice(*, soffice_path: str | None = None) -> Path:
    """Locate a configured, packaged, or system LibreOffice executable."""
    local_program_dir = _local_program_dir()
    wrapper_program_dir = _program_dir_from_soffice_wrapper()
    system_soffice = shutil.which("soffice")

    return _first_executable(
        [
            soffice_path,
            os.getenv(SOFFICE_PATH_ENV),
            local_program_dir / "soffice",
            wrapper_program_dir / "soffice" if wrapper_program_dir else None,
            system_soffice,
            "/usr/lib/libreoffice/program/soffice",
            "/opt/libreoffice/program/soffice",
            "/Applications/LibreOffice.app/Contents/MacOS/soffice",
        ],
        "LibreOffice soffice",
    )


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


async def recalculate_workbook(
    workbook: Path,
    *,
    status_path: Path | None = None,
    output_path: Path | None = None,
    soffice_path: str | None = None,
    timeout: int = 90,
) -> dict[str, Any]:
    """Recalculate a temporary copy and publish it without changing the source workbook.

    Args:
        workbook: Workbook path handled by the recalculation process.
        status_path: Optional JSON status output path, defaulting beside the workbook.
        output_path: Recalculated file; defaults to <source stem>.recalc<suffix> beside the source.
        soffice_path: Optional explicit LibreOffice executable path.
        timeout: Maximum runtime in seconds.

    Returns:
        JSON status containing status (success, errors_found, not_needed, or failure),
        workbook and output_workbook paths, created_output, formula counts, and any error details.

    Raises:
        ValueError: The output aliases the source or is a symbolic link.
    """

    workbook = workbook.expanduser().resolve()
    output_path = (
        (output_path or workbook.with_name(f"{workbook.stem}.recalc{workbook.suffix}")).expanduser().absolute()
    )
    if output_path.is_symlink() or output_path.resolve() == workbook:
        raise ValueError("Recalculated output must be a separate file and must not be a symbolic link")
    status_path = (status_path or workbook.parent / STATUS_FILENAME).expanduser().resolve()
    status: dict[str, Any] = {
        "method": "libreoffice_transactional_recalculation",
        "workbook": str(workbook),
        "output_workbook": str(output_path),
        "status": "failure",
        "attempted": False,
        "created_output": False,
    }
    if not workbook.is_file():
        status["error"] = f"workbook not found: {workbook}"
        _write_status(status_path, status)
        return status

    work_root: Path | None = None
    office_stdout = ""
    office_stderr = ""

    try:
        output_path.unlink(missing_ok=True)
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

        soffice = _find_soffice(soffice_path=soffice_path)
        status.update(
            {
                "attempted": True,
                "soffice": str(soffice),
                "engine": "libreoffice_headless_conversion",
            }
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        work_root = Path(tempfile.mkdtemp(prefix=".spreadsheetbench_recalc_", dir=output_path.parent))
        profile_dir = work_root / "profile"
        profile_dir.mkdir()
        temporary_workbook = work_root / workbook.name
        _copy_for_recalculation(workbook, temporary_workbook)
        converted_dir = work_root / "converted"
        converted_dir.mkdir()
        converted_workbook = converted_dir / workbook.name

        office_command = [
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
        ]
        office_environment = os.environ.copy()
        office_environment["SAL_USE_VCLPLUGIN"] = "svp"
        office = await execute(
            office_command,
            temporary_workbook.parent,
            ExecutionConfig(timeout_s=timeout, environment=office_environment),
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

        os.replace(converted_workbook, output_path)
        status.update(
            {
                "status": "errors_found" if after["excel_errors"] else "success",
                "error": "",
                "created_output": True,
                "sha256_after": _sha256(output_path),
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
    """Return command-line options for recalculating a separate workbook."""
    parser = argparse.ArgumentParser(description="Recalculate spreadsheet formulas with LibreOffice")
    parser.add_argument("workbook", nargs="?", type=Path)
    parser.add_argument("--status-path", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--soffice-path", default=None)
    parser.add_argument("--timeout", type=int, default=90)
    return parser


async def main() -> int:
    """Recalculate the supplied workbook path and print its JSON status."""
    args = build_arg_parser().parse_args()
    if args.workbook is None:
        raise SystemExit("workbook path is required")

    result = await recalculate_workbook(
        args.workbook,
        status_path=args.status_path,
        output_path=args.output_path,
        soffice_path=args.soffice_path,
        timeout=args.timeout,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] in {"success", "not_needed"} else 1


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "STATUS_FILENAME",
    "recalculate_workbook",
]
