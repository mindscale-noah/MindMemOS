"""Render public source pages from released OfficeQA parsed documents."""

import html
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qs, urlparse


class _TableMarkdownParser(HTMLParser):
    """Collect cell text in the original row order without adding HTML layout spacing."""

    def __init__(self) -> None:
        """Initialize one independent table parse."""
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "tr":
            self._row = []
        elif tag.lower() in {"td", "th"} and self._row is not None:
            self._cell = []

    def handle_data(self, data: str) -> None:
        """Append text emitted inside the current cell."""
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        normalized_tag = tag.lower()
        if normalized_tag in {"td", "th"} and self._cell is not None and self._row is not None:
            cell = re.sub(r"\s+", " ", "".join(self._cell)).strip()
            self._row.append(cell)
            self._cell = None
        elif normalized_tag == "tr" and self._row is not None:
            if any(cell for cell in self._row):
                self.rows.append(self._row)
            self._row = None
            self._cell = None


def _escape_markdown_cell(value: str) -> str:
    return str(value).replace("\n", " ").replace("|", "\\|").strip()


def _html_table_to_markdown(raw_html: str) -> str:
    """Preserve released cell order and pad ragged rows to a shared width."""
    parser = _TableMarkdownParser()
    try:
        parser.feed(raw_html)
    except Exception:  # noqa: BLE001
        parser.rows = []
    rows = parser.rows
    if not rows:
        text = re.sub(r"(?is)<[^>]+>", " ", raw_html)
        return re.sub(r"\s+", " ", html.unescape(text)).strip()
    width = max(len(row) for row in rows)
    normalized_rows = [row + [""] * (width - len(row)) for row in rows]
    header = normalized_rows[0]
    body = normalized_rows[1:]
    lines = [
        "| " + " | ".join(_escape_markdown_cell(cell) for cell in header) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    lines.extend("| " + " | ".join(_escape_markdown_cell(cell) for cell in row) + " |" for row in body)
    return "\n".join(lines)


def _render_parsed_content(content: str) -> str:
    """Normalize public prose or reproduce the legacy table representation."""
    text = content.strip()
    if not text:
        return ""
    if "<table" in text.lower():
        return _html_table_to_markdown(text)
    text = html.unescape(text)
    text = re.sub(r"\r\n?", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def oracle_pages(root: Path, files: list[str], urls: list[str]) -> str:
    """Read only page IDs explicitly named by the dataset source hints.

    Args:
        root: Official public document corpus root.
        files: Source document filenames.
        urls: Source URLs carrying page, pagenum, or page_id parameters.

    Returns:
        Bounded public page evidence, without any question answer fields.
    """
    blocks, seen, total = [], set(), 0
    for index, url in enumerate(urls):
        parameters = parse_qs(urlparse(url).query)
        pages = [
            value for key in ("page", "pagenum", "page_id") for value in parameters.get(key, []) if value.isdigit()
        ]
        if not pages or not files or (index >= len(files) and len(files) != 1):
            continue
        number = int(pages[0])
        filename = files[index] if index < len(files) else files[0]
        candidates = [
            root / "jsons" / f"{Path(filename).stem}.json",
            root / "treasury_bulletins_parsed/jsons" / f"{Path(filename).stem}.json",
        ]
        path = next((path for path in candidates if path.is_file() and path.resolve().is_relative_to(root)), None)
        if path is None or (path, number) in seen:
            continue
        seen.add((path, number))
        payload = json.loads(path.read_text(encoding="utf-8"))
        sections = []
        for element in payload.get("document", {}).get("elements", []):
            page_ids = {str(box.get("page_id", "")) for box in element.get("bbox", []) if isinstance(box, dict)}
            content = element.get("content", "")
            if str(number) not in page_ids or not isinstance(content, str) or not content.strip():
                continue
            sections.append(_render_parsed_content(content))
        text = "\n\n".join(sections).strip()
        if not text:
            continue
        if len(text) > 24000:
            text = text[:24000].rstrip() + f"\n\n[... {len(text) - 24000} characters omitted from this parsed page ...]"
        block = f"### {filename} page {number}\nSource URL: {url}\n\n{text}"
        if total + len(block) > 80000:
            blocks.append(block[: 80000 - total].rstrip() + "\n\n[... oracle parsed page context truncated ...]")
            break
        blocks.append(block)
        total += len(block)
    if not blocks:
        return ""
    return (
        "The following content is pre-parsed from the oracle OfficeQA source page(s). Treat it as primary document evidence and combine it with local document tool evidence when useful.\n\n"
        + "\n\n".join(blocks)
    )
