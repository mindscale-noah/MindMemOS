"""Render public source pages from released OfficeQA parsed documents."""

import html
import json
import re
from pathlib import Path
from urllib.parse import parse_qs, urlparse


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
            content = re.sub(r"</t[dh]>\s*", " | ", content, flags=re.IGNORECASE)
            content = re.sub(r"</(?:tr|p|div)>|<br\s*/?>", "\n", content, flags=re.IGNORECASE)
            content = html.unescape(re.sub(r"<[^>]+>", "", content))
            sections.append(re.sub(r"\n{3,}", "\n\n", content).strip())
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
        "The following content is pre-parsed from the oracle OfficeQA source page(s). Treat it as primary document evidence.\n\n"
        + "\n\n".join(blocks)
    )
