"""Parse released SpreadsheetBench answer ranges and worksheet lists."""


def split_targets(position: str) -> list[str]:
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


def split_region(target: str, default_sheet: str) -> tuple[str, str]:
    """Split an answer target while retaining quoted sheet names."""
    if "!" in target:
        sheet, region = target.rsplit("!", 1)
        sheet = sheet.strip().strip("'\"")
    else:
        sheet, region = default_sheet, target
    return sheet, region.strip().strip("'\"")


def default_sheet(sheetnames: list[str], answer_sheet: str | None) -> str:
    """Resolve the shared default sheet for unqualified ranges."""
    if not answer_sheet:
        return sheetnames[0]
    default = answer_sheet.strip().strip("'\"")
    if default in sheetnames:
        return default
    return next(
        (part.strip().strip("'\"") for part in split_targets(answer_sheet) if part.strip().strip("'\"") in sheetnames),
        default,
    )
