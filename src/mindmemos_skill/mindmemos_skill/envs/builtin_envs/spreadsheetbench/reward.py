"""Shared cached-value scoring with shared range parsing and cell reports."""

import datetime
import re
from contextlib import ExitStack
from pathlib import Path

from ...typing import Reward
from .typing import SpreadsheetBenchRewardData


def _value(value):
    """Apply the official two-decimal quantization and datetime conversion."""
    if isinstance(value, (int, float)):
        return round(float(value), 2)
    if isinstance(value, datetime.time):
        return str(value)[:-3]
    if isinstance(value, datetime.datetime):
        delta = value - datetime.datetime(1899, 12, 30)
        return round(delta.days + delta.seconds / 86400.0, 0)
    if isinstance(value, str):
        try:
            return round(float(value), 2)
        except ValueError:
            return value
    return value


def _same(expected, actual) -> bool:
    """Compare normalized values without style or formula-source scoring."""
    expected, actual = _value(expected), _value(actual)
    if expected in ("", None) and actual in ("", None):
        return True
    return type(expected) is type(actual) and expected == actual


def _targets(position: str) -> list[str]:
    """Parse comma-separated targets while preserving quoted sheet names."""
    position = position.strip()
    bang = position.find("!")
    if bang > 0:
        sheet = position[:bang].lstrip()
        if sheet[-1:] in {"'", '"'} and sheet[:1] != sheet[-1:]:
            position = sheet[-1] + position
    parts, current, quote = [], "", None
    for character in position:
        if character in {"'", '"'}:
            quote = character if quote is None else (None if quote == character else quote)
        if character == "," and quote is None:
            if current.strip():
                parts.append(current.strip())
            current = ""
        else:
            current += character
    if current.strip():
        parts.append(current.strip())
    return parts


def _coordinates(region: str, *, max_row: int = 1, max_column: int = 1) -> list[str]:
    """Expand shared ranges in row-major order using workbook bounds."""
    from openpyxl.utils import column_index_from_string, get_column_letter

    if ":" not in region:
        return [region]
    first, last = region.split(":", 1)
    matches = [re.fullmatch(r"\$?([A-Za-z]+)?\$?(\d+)?", part.strip()) for part in (first, last)]
    if any(match is None or not any(match.groups()) for match in matches):
        raise ValueError(f"invalid cell coordinate: {region!r}")
    first_column, first_row = matches[0].groups()
    last_column, last_row = matches[1].groups()
    left = column_index_from_string(first_column) if first_column else 1
    right = column_index_from_string(last_column) if last_column else (left if first_column else max_column)
    top = int(first_row) if first_row else 1
    bottom = int(last_row) if last_row else max_row
    if left > right or top > bottom:
        raise ValueError(f"answer range is reversed: {region!r}")
    return [f"{get_column_letter(column)}{row}" for row in range(top, bottom + 1) for column in range(left, right + 1)]


def _region(target: str, default_sheet: str) -> tuple[str, str]:
    """Split an answer target while retaining quoted sheet names."""
    if "!" in target:
        sheet, region = target.rsplit("!", 1)
    else:
        sheet, region = default_sheet, target
    return sheet.strip().strip("'\""), region.strip().strip("'\"")


def _default_sheet(sheetnames: list[str], answer_sheet: str | None) -> str:
    """Resolve the shared default sheet for unqualified ranges."""
    if not answer_sheet:
        return sheetnames[0]
    default = answer_sheet.strip().strip("'\"")
    if default in sheetnames:
        return default
    return next(
        (part.strip().strip("'\"") for part in _targets(answer_sheet) if part.strip().strip("'\"") in sheetnames),
        default,
    )


def compare_workbooks(
    gold_path: Path,
    output_path: Path,
    answer_position: str,
    *,
    answer_sheet: str | None = None,
) -> tuple[bool, str]:
    """Compare cached cell values using shared range parsing without recalculating formulas.

    Args:
        gold_path: Private expected workbook.
        output_path: Generated workbook.
        answer_position: Released scoring regions.
        answer_sheet: shared default sheet for unqualified ranges.

    Returns:
        Accepted flag and the original first-mismatch text.
    """
    import openpyxl

    if not _targets(answer_position):
        raise ValueError("answer_position must contain at least one scoring range")
    if not output_path.exists():
        return False, "file not exist"
    with ExitStack() as stack:
        try:
            gold = openpyxl.load_workbook(gold_path, data_only=True)
            stack.callback(gold.close)
            output = openpyxl.load_workbook(output_path, data_only=True)
            stack.callback(output.close)
        except Exception as exc:
            return False, f"load error: {exc}"
        default = _default_sheet(gold.sheetnames, answer_sheet)
        for target in _targets(answer_position):
            sheet, region = _region(target, default)
            if sheet not in output.sheetnames or sheet not in gold.sheetnames:
                return False, f"worksheet not found: {sheet}"
            expected, actual = gold[sheet], output[sheet]
            for coordinate in _coordinates(
                region,
                max_row=max(expected.max_row, actual.max_row, 1),
                max_column=max(expected.max_column, actual.max_column, 1),
            ):
                left, right = expected[coordinate].value, actual[coordinate].value
                if not _same(left, right):
                    return False, f"value@{sheet}!{coordinate}: gt={left!r} pred={right!r}"
    return True, ""


def verification_report(
    output_path: Path, gold_path: Path, answer_position: str, *, answer_sheet: str | None = None
) -> str:
    """Render a cell report using the same ranges and values as the scorer.

    Args:
        output_path: Generated workbook.
        gold_path: Private expected workbook.
        answer_position: Scored cells in released order.
        answer_sheet: Default sheet for unqualified ranges.

    Returns:
        Original report wording, whitespace, symbols, and head/tail truncation.
    """
    import openpyxl

    if not output_path.exists():
        return "Verification: output file does not exist."
    with ExitStack() as stack:
        try:
            output = openpyxl.load_workbook(output_path, data_only=True)
            stack.callback(output.close)
            gold = openpyxl.load_workbook(gold_path, data_only=True)
            stack.callback(gold.close)
        except Exception as exc:
            return f"Verification: could not open workbooks: {exc}"
        lines = ["## Output Verification"]
        for target in _targets(answer_position):
            sheet, region = _region(target, _default_sheet(gold.sheetnames, answer_sheet))
            if sheet not in output.sheetnames or sheet not in gold.sheetnames:
                lines.append(f"  Sheet '{sheet}' NOT FOUND in output.")
                continue
            empty = 0
            for coordinate in _coordinates(
                region,
                max_row=max(gold[sheet].max_row, output[sheet].max_row, 1),
                max_column=max(gold[sheet].max_column, output[sheet].max_column, 1),
            ):
                expected = gold[sheet][coordinate].value if sheet in gold.sheetnames else "N/A"
                actual = output[sheet][coordinate].value
                correct = sheet in gold.sheetnames and _same(expected, actual)
                if correct and expected in (None, "") and actual in (None, ""):
                    empty += 1
                    continue
                mark = "✓" if correct else "✗"
                lines.append(f"  {sheet}!{coordinate}: got={actual!r}, expected={expected!r} {mark}")
            if empty:
                lines.append(f"  (+{empty} empty cells correct, omitted)")
        formulas = []
        for sheet in output:
            for row in sheet.iter_rows(max_row=min(sheet.max_row, 200)):
                for cell in row:
                    if isinstance(cell.value, str) and cell.value.startswith("="):
                        formulas.append(f"{sheet.title}!{cell.coordinate}={cell.value}")
                        if len(formulas) >= 10:
                            break
                if len(formulas) >= 10:
                    break
            if len(formulas) >= 10:
                break
        if formulas:
            lines.append(f"\n  WARNING: {len(formulas)} cells contain Excel formulas (openpyxl cannot evaluate them):")
            lines.extend(f"    {formula}" for formula in formulas[:5])
            if len(formulas) > 5:
                lines.append(f"    ... and {len(formulas) - 5} more")
    report = "\n".join(lines)
    if len(report) > 12000:
        report = (
            report[:6000] + f"\n  ...[verification report truncated, {len(report)} chars total]...\n" + report[-6000:]
        )
    return report


def evaluate(reward_data: SpreadsheetBenchRewardData, directories: list[Path]) -> Reward:
    """Score all outputs with one shared aggregation and feedback contract.

    Args:
        reward_data: Private workbook references and scoring regions.
        directories: Public case directories in reference order.

    Returns:
        Case pass fraction, all-cases success, and post-execution cell reports.

    Raises:
        ValueError: Case counts differ, the scoring range is empty or invalid,
            or an output is a symbolic link.
    """
    if len(directories) != len(reward_data.cases):
        raise ValueError("Output directories must match reference cases")
    if not _targets(reward_data.answer_position):
        raise ValueError("answer_position must contain at least one scoring range")
    cases, reports = [], []
    passed, executed = 0, 0
    failure = ""
    for reference, directory in zip(reward_data.cases, directories):
        prediction = directory / "output.xlsx"
        if prediction.is_symlink():
            raise ValueError("Generated output must not be a symbolic link")
        if not prediction.is_file():
            cases.append({"no": reference.number, "stage": "exec", "ok": False, "error": "output-not-found"})
            failure = failure or "output-not-found"
            reports.append(f"## Execution (case {reference.number})\nERROR: output-not-found")
            continue
        executed += 1
        ok, reason = compare_workbooks(
            reference.path, prediction, reward_data.answer_position, answer_sheet=reward_data.answer_sheet
        )
        passed += int(ok)
        if not ok:
            failure = failure or f"eval-mismatch: {reason[:200]}"
        cases.append({"no": reference.number, "stage": "eval", "ok": ok, "reason": reason})
        report = verification_report(
            prediction, reference.path, reward_data.answer_position, answer_sheet=reward_data.answer_sheet
        )
        reports.append(f"## Eval Result (case {reference.number}): {'PASS' if ok else 'FAIL'}\n{reason}\n\n{report}")
    success = passed == len(cases)
    return Reward(
        score=passed / len(cases),
        success=success,
        detail="[POST-EXECUTION VERIFICATION]\n\n" + "\n\n---\n\n".join(reports),
        hard=int(success),
        soft=passed / len(cases),
        correct=success,
        n_cases=len(cases),
        n_pass=passed,
        n_exec_pass=executed,
        cases=cases,
        exec_ok=executed == len(cases),
        fail_reason="" if success else failure,
    )
