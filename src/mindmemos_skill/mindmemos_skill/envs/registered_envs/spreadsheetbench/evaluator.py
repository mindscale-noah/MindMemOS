"""Official cached-cell-value SpreadsheetBench comparison semantics."""

from __future__ import annotations

import datetime
import re
from collections.abc import Sequence
from itertools import zip_longest
from pathlib import Path
from typing import Any

from ....errors import SkillCapabilityUnavailableError


def _openpyxl():
    try:
        import openpyxl
    except ImportError as exc:
        raise SkillCapabilityUnavailableError(
            "SpreadsheetBench requires the mindmemos-skill[spreadsheetbench] optional dependency"
        ) from exc
    return openpyxl


def _transform_value(value: Any) -> Any:
    if isinstance(value, bool):
        return round(float(value), 2)
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


def _same_value(left: Any, right: Any) -> bool:
    left, right = _transform_value(left), _transform_value(right)
    if left in {"", None} and right in {"", None}:
        return True
    return type(left) is type(right) and left == right


_CELL_PART_RE = re.compile(r"^\$?(?P<column>[A-Za-z]+)?\$?(?P<row>\d+)?$")


def _split_targets(value: str) -> list[str]:
    normalized = value.strip()
    if not normalized:
        return []

    # One upstream record omits the opening quote in ``Sheet3'!A:G``.
    first_bang = normalized.find("!")
    if first_bang > 0:
        raw_sheet = normalized[:first_bang].lstrip()
        if raw_sheet[-1:] in {"'", '"'} and raw_sheet[:1] != raw_sheet[-1:]:
            normalized = f"{raw_sheet[-1]}{normalized}"

    targets: list[str] = []
    current: list[str] = []
    quote: str | None = None
    for character in normalized:
        if character in {"'", '"'}:
            if quote is None:
                quote = character
            elif quote == character:
                quote = None
        if character == "," and quote is None:
            target = "".join(current).strip()
            if target:
                targets.append(target)
            current = []
            continue
        current.append(character)
    target = "".join(current).strip()
    if target:
        targets.append(target)
    return targets


def _sheet_name(value: str) -> str:
    return value.strip().strip("'\"")


def _default_sheet(sheet_names: Sequence[str], answer_sheet: str | None) -> str:
    if answer_sheet:
        exact = _sheet_name(answer_sheet)
        if exact in sheet_names:
            return exact
        for candidate in _split_targets(answer_sheet):
            name = _sheet_name(candidate)
            if name in sheet_names:
                return name
        return exact
    return sheet_names[0]


def _cell_part(value: str) -> tuple[str | None, int | None]:
    match = _CELL_PART_RE.fullmatch(value.strip())
    if match is None or not any(match.groups()):
        raise ValueError(f"invalid cell coordinate: {value!r}")
    column = match.group("column")
    row = match.group("row")
    return column, int(row) if row is not None else None


def _cell_bounds(
    cell_range: str,
    *,
    max_row: int,
    max_column: int,
) -> tuple[int, int, int, int]:
    openpyxl = _openpyxl()
    normalized = cell_range.strip().strip("'\"")
    start_raw, end_raw = normalized.split(":", 1) if ":" in normalized else (normalized, normalized)
    start_column_name, start_row = _cell_part(start_raw)
    end_column_name, end_row = _cell_part(end_raw)

    if ":" not in normalized and (start_column_name is None or start_row is None):
        raise ValueError(f"single-cell answer position must include a column and row: {normalized!r}")

    if start_column_name is not None:
        start_column = openpyxl.utils.column_index_from_string(start_column_name)
    else:
        start_column = 1
    if end_column_name is not None:
        end_column = openpyxl.utils.column_index_from_string(end_column_name)
    elif start_column_name is not None:
        end_column = start_column
    else:
        end_column = max_column
    first_row = start_row or 1
    last_row = end_row or max_row

    if end_column < start_column or last_row < first_row:
        raise ValueError(f"answer range is reversed: {normalized!r}")
    return first_row, last_row, start_column, end_column


def _compare_range(golden_sheet: Any, output_sheet: Any, cell_range: str) -> tuple[bool, str]:
    openpyxl = _openpyxl()
    min_row, max_row, min_column, max_column = _cell_bounds(
        cell_range,
        max_row=max(golden_sheet.max_row or 0, output_sheet.max_row or 0, 1),
        max_column=max(golden_sheet.max_column or 0, output_sheet.max_column or 0, 1),
    )
    golden_rows = golden_sheet.iter_rows(
        min_row=min_row,
        max_row=max_row,
        min_col=min_column,
        max_col=max_column,
        values_only=True,
    )
    output_rows = output_sheet.iter_rows(
        min_row=min_row,
        max_row=max_row,
        min_col=min_column,
        max_col=max_column,
        values_only=True,
    )
    empty_row = (None,) * (max_column - min_column + 1)
    for row_offset, (expected_row, actual_row) in enumerate(zip_longest(golden_rows, output_rows, fillvalue=empty_row)):
        for column_offset, (expected, actual) in enumerate(zip_longest(expected_row, actual_row, fillvalue=None)):
            if not _same_value(expected, actual):
                column = openpyxl.utils.get_column_letter(min_column + column_offset)
                cell = f"{column}{min_row + row_offset}"
                return False, f"value@{golden_sheet.title}!{cell}: gt={expected!r} pred={actual!r}"
    return True, ""


def compare_workbooks(
    golden_path: str | Path,
    output_path: str | Path,
    answer_position: str,
    *,
    answer_sheet: str | None = None,
) -> tuple[bool, str]:
    output_path = Path(output_path)
    if not output_path.exists():
        return False, "file not exist"
    openpyxl = _openpyxl()
    golden: Any | None = None
    output: Any | None = None
    try:
        try:
            golden = openpyxl.load_workbook(golden_path, data_only=True, read_only=True)
            output = openpyxl.load_workbook(output_path, data_only=True, read_only=True)
        except Exception as exc:
            return False, f"load error: {exc}"

        default_sheet = _default_sheet(golden.sheetnames, answer_sheet)
        for target in _split_targets(answer_position):
            if "!" in target:
                sheet, cell_range = target.rsplit("!", 1)
                sheet = _sheet_name(sheet)
            else:
                sheet, cell_range = default_sheet, target
            if sheet not in golden.sheetnames or sheet not in output.sheetnames:
                return False, f"worksheet not found: {sheet}"
            golden_sheet = golden[sheet]
            output_sheet = output[sheet]
            correct, detail = _compare_range(golden_sheet, output_sheet, cell_range)
            if not correct:
                return False, detail
        return True, ""
    finally:
        if golden is not None:
            golden.close()
        if output is not None:
            output.close()


__all__ = ["compare_workbooks"]
