"""OfficeQA binary scoring, adapted from Apache-2.0 databricks/officeqa.

Protocol revision: 309dc01ad21113d240700eecb74d2f3e59f163b6.
Keep direct-answer guards, ordered lists, contextual years, and unit semantics.
"""

import re
from functools import lru_cache


def _normalize(text: str) -> str:
    """Normalize Unicode minus and whitespace before numeric parsing."""
    return " ".join(text.replace("−", "-").split())


def _year(value: float) -> bool:
    """Identify incidental integral year values under the pinned protocol."""
    return 1900 <= value <= 2100 and value == int(value)


def _numeric_format(text: str) -> str:
    """Remove currency symbols and translate accounting negatives."""
    symbols = r"$£€¥₹¢₩₽"
    number = r"\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?"

    def accounting(match: re.Match) -> str:
        """Preserve parenthesized years unless marked as currency."""
        currency, value = match.groups()
        return match.group() if currency is None and _year(float(value.replace(",", ""))) else f"-{value}"

    text = re.sub(rf"\(\s*([{symbols}])?\s*({number})\s*\)", accounting, text)
    return re.sub(rf"[{symbols}]", "", text)


def _numbers(text: str) -> list[tuple[float, str]]:
    """Extract values with the same twenty-character context window."""
    text = _numeric_format(_normalize(text))
    text = re.sub(r"(?<![\d.])\d{1,3}(?:,\d{3})+(?:\.\d+)?(?!\d)", lambda m: m.group().replace(",", ""), text)
    return [
        (float(match.group().rstrip("%")), text[max(0, match.start() - 20) : match.end() + 20].lower())
        for match in re.finditer(r"-?\d+\.?\d*%?", text)
    ]


def _unit(context: str) -> str | None:
    """Recognize explicit units without rescaling the numeric base value."""
    for name, pattern in (
        ("trillion", r"\btrillions?\b"),
        ("billion", r"\bbillions?\b|\bb\b"),
        ("million", r"\bmillions?\b|\bm\b"),
        ("thousand", r"\bthousands?\b|\bk\b"),
    ):
        if re.search(pattern, context.lower()):
            return name
    return None


def _significant(text: str) -> str:
    """Remove numbers, units, and punctuation to identify hybrid answers."""
    text = re.sub(r"-?\d+\.?\d*%?|,", "", _normalize(text).lower())
    for word in (
        "trillion",
        "trillions",
        "billion",
        "billions",
        "million",
        "millions",
        "thousand",
        "thousands",
        "hundred",
        "hundreds",
        "percent",
        "percentage",
        "%",
    ):
        text = re.sub(r"\b" + word + r"\b", "", text)
    return " ".join(re.sub(r"[^\w\s]", " ", text).split())


def _text_overlap(gold: str, prediction: str) -> bool:
    """Apply the pinned hybrid-text overlap check."""
    expected, actual = _significant(gold), _significant(prediction)
    return len(expected) < 2 or (len(actual) >= 2 and (expected in actual or actual in expected))


def _direct_text(text: str) -> str:
    """Normalize direct text answers, including parenthetical abbreviations."""
    text = text.strip().lower().strip('"').strip("'")
    return " ".join(re.sub(r"\([^)]*\)", "", text).split())


def _bracket_body(text: str) -> str | None:
    """Find a single bracketed list with no numeric surrounding label."""
    text = _normalize(text)
    matches = list(re.finditer(r"\[([^\[\]]+)\]", text))
    if len(matches) != 1:
        return None
    match = matches[0]
    outside = text[: match.start()] + text[match.end() :]
    return None if re.search(r"\d", outside) else match.group(1)


def _list_candidates(text: str, count: int) -> list[list[float]]:
    """Resolve ambiguous commas only within an expected numeric-list shape."""
    if count < 2:
        return []
    body = _bracket_body(text) or _normalize(text)
    chunks = body.split(",")

    @lru_cache(None)
    def segment(index: int, remaining: int) -> list[list[float]]:
        """Partition comma chunks into exactly the expected number of values."""
        if remaining == 0:
            return [[]] if index == len(chunks) else []
        values = []
        for end in range(index + 1, len(chunks) - remaining + 2):
            raw = ",".join(chunks[index:end]).strip()
            if re.search(r"\s,|,\s", raw):
                continue
            normalized = _numeric_format(_normalize(raw))
            if re.fullmatch(r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?", normalized):
                value = float(normalized.rstrip("%").replace(",", ""))
                values.extend([value, *tail] for tail in segment(end, remaining - 1))
        return values

    return segment(0, count)


def _without_context_years(
    gold: list[tuple[float, str]], prediction: list[tuple[float, str]], text: str
) -> list[tuple[float, str]]:
    """Filter incidental prediction years using direct-answer guard rules."""
    retain = any(_year(value) for value, _ in gold) or (len(gold) == 1 and len(_significant(text)) >= 2)
    return prediction if retain else [(value, context) for value, context in prediction if not _year(value)]


def _same_number(gold: tuple[float, str], prediction: tuple[float, str]) -> bool:
    """Match base values with zero tolerance and optional compatible units."""
    expected, actual = _unit(gold[1]), _unit(prediction[1])
    return gold[0] == prediction[0] and (expected is None or actual is None or expected == actual)


def score_answer(ground_truth: str, predicted: str) -> float:
    """Apply the pinned official zero-tolerance binary scoring protocol.

    Args:
        ground_truth: Private expected direct answer.
        predicted: Extracted model answer, optionally wrapped in FINAL_ANSWER.

    Returns:
        One for an accepted answer, otherwise zero.
    """
    matches = re.findall(r"<FINAL_ANSWER>\s*(.*?)\s*</FINAL_ANSWER>", predicted, re.DOTALL | re.IGNORECASE)
    predicted = (matches[-1] if matches else predicted).strip()
    if not ground_truth or not predicted or len(predicted) > 250:
        return 0.0
    if len([line for line in predicted.splitlines() if line.strip()]) > 1:
        return 0.0
    gold, actual = _numbers(ground_truth), _numbers(predicted)
    filtered = _without_context_years(gold, actual, ground_truth) if gold else actual
    body = _bracket_body(ground_truth)
    gold_list = bool(body and len(gold) >= 2 and [value for value, _ in gold] in _list_candidates(body, len(gold)))
    candidates = _list_candidates(predicted, len(gold)) if gold_list else []
    if gold:
        if len(filtered) != len(gold) and not candidates:
            return 0.0
        if re.search(r"<[^>]*>", predicted):
            return 0.0
    elif actual or _direct_text(ground_truth) != _direct_text(predicted):
        return 0.0
    if "unable to determine" in predicted.lower():
        return 0.0
    if gold and actual:
        if gold_list:
            alternatives = [[(value, "") for value in values] for values in candidates] + [filtered]
            return float(
                any(
                    len(values) == len(gold) and all(_same_number(g, p) for g, p in zip(gold, values))
                    for values in alternatives
                )
            )
        if len(gold) > 1:
            eligible = (
                actual
                if any(_year(value) for value, _ in gold)
                else [(value, context) for value, context in actual if not _year(value)]
            )
            return float(
                _text_overlap(ground_truth, predicted) and all(any(_same_number(g, p) for p in eligible) for g in gold)
            )
        return float(_text_overlap(ground_truth, predicted) and any(_same_number(gold[0], p) for p in filtered))
    return float(_direct_text(ground_truth) in _direct_text(predicted))
