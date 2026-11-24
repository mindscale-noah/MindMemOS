"""Backend-neutral Markdown/frontmatter parsing for Skill documents.

Ported from the GOS ``core/parsing.py`` pure logic, tightened to the P2 frozen
contract: strict YAML (duplicate keys rejected), NFC normalization, CRLF->LF,
typed errors instead of silent ``None``/truncation, and no filesystem scanning.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

import yaml

from .constant import (
    METADATA_ALLOWED_TOOLS,
    METADATA_COMPATIBILITY,
    METADATA_DOMAIN,
    METADATA_DOMAINS,
    METADATA_EXAMPLE_TASKS,
    METADATA_EXAMPLES,
    METADATA_KEY,
    METADATA_LIBRARIES,
    METADATA_SUMMARY,
    METADATA_TAGS,
    METADATA_TOOLING,
    METADATA_VALUE,
)
from .errors import input_too_large, invalid_skill_document

FRONTMATTER_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)
SECTION_PATTERN_TEMPLATE = r"(?ims)^\s*#{{1,6}}\s*(?:{titles})\s*$\n(.*?)(?=^\s*#{{1,6}}\s+|\Z)"


class StrictYamlLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _no_duplicate_keys(loader: StrictYamlLoader, node: yaml.MappingNode, deep: bool = False) -> dict:
    mapping: dict = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


StrictYamlLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _no_duplicate_keys,
)


def normalize_content(content: str) -> str:
    """Normalize to NFC Unicode and CRLF->LF before any parsing.

    Args:
        content: Canonical Skill document text.
    """

    return unicodedata.normalize("NFC", re.sub(r"\r\n?", "\n", content))


def extract_frontmatter(content: str, *, version_id: str) -> tuple[dict[str, Any], str]:
    """Read strict YAML frontmatter and the remaining Markdown body.

    Args:
        content: Canonical Skill document text.
        version_id: Optional source identity used in parsing errors.
    """
    match = FRONTMATTER_PATTERN.search(content)
    if match:
        raw_frontmatter = match.group(1)
        body = content[match.end() :].strip()
    else:
        # Canonical Skill bundles strip the leading '---' delimiters. Fall back
        # to treating the leading YAML block (up to the first blank line + body)
        # as frontmatter so the same parser accepts both forms.
        first_blank = re.search(r"\n\s*\n", content)
        if first_blank is None or first_blank.start() == 0:
            raise invalid_skill_document(version_id=version_id, reason="missing frontmatter")
        raw_frontmatter = content[: first_blank.start()]
        body = content[first_blank.end() :].strip()
        if not raw_frontmatter.strip():
            raise invalid_skill_document(version_id=version_id, reason="missing frontmatter")
    try:
        frontmatter = yaml.load(raw_frontmatter, Loader=StrictYamlLoader) or {}
    except yaml.YAMLError as exc:
        raise invalid_skill_document(version_id=version_id, reason=f"invalid YAML frontmatter: {exc}") from exc
    if not isinstance(frontmatter, dict):
        raise invalid_skill_document(version_id=version_id, reason="frontmatter must be a mapping")
    return frontmatter, body


def normalize_string_list(value: Any) -> list[str]:
    """Normalize common YAML field shapes into ordered strings.

    Args:
        value: Source field value to normalize.
    """
    if value is None:
        return []
    if isinstance(value, str):
        parts = re.split(r"[\n,;]+", value)
        return [part.strip() for part in parts if part.strip()]
    if isinstance(value, dict):
        items: list[str] = []
        for key, item in value.items():
            if isinstance(item, (dict, list, tuple, set)):
                items.append(f"{key}: {item}")
            else:
                items.append(f"{key}: {str(item).strip()}")
        return [item for item in items if item]
    if isinstance(value, (list, tuple, set)):
        normalized: list[str] = []
        for item in value:
            normalized.extend(normalize_string_list(item))
        return normalized
    return [str(value).strip()]


def compact_text(text: str) -> str:
    """Normalize spaces while preserving meaningful paragraph breaks.

    Args:
        text: Text to normalize or truncate.
    """
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clip_text(text: str, max_chars: int) -> str:
    """Produce a bounded display snippet without modifying canonical content.

    Args:
        text: Text to normalize or truncate.
        max_chars: Maximum accepted or rendered character count.
    """
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    clipped = text[: max_chars - 3].rstrip()
    return f"{clipped}..."


def build_rendered_snippet(
    name: str,
    description: str,
    body: str,
    compatibility: list[str],
    allowed_tools: list[str],
    max_chars: int,
) -> str:
    """Render source-compatible searchable Skill text.

    Args:
        name: Display name extracted from frontmatter.
        description: Parsed capability description.
        body: Markdown body following frontmatter.
        compatibility: Declared compatibility descriptions.
        allowed_tools: Declared tool names.
        max_chars: Maximum accepted or rendered character count.
    """
    parts = [f"# {name}", description.strip()]
    if compatibility:
        parts.append(f"Compatibility: {', '.join(compatibility)}")
    if allowed_tools:
        parts.append(f"Allowed tools: {', '.join(allowed_tools)}")
    if body.strip():
        parts.append(body.strip())
    return clip_text(compact_text("\n\n".join(part for part in parts if part)), max_chars)


def extract_markdown_section(body: str, titles: list[str]) -> list[str]:
    """Extract explicit list items from the first matching section.

    Args:
        body: Markdown body following frontmatter.
        titles: Accepted Markdown section titles.
    """
    if not body.strip():
        return []
    title_pattern = "|".join(re.escape(title) for title in titles)
    pattern = re.compile(SECTION_PATTERN_TEMPLATE.format(titles=title_pattern))
    match = pattern.search(body)
    if not match:
        return []
    section_body = match.group(1).strip()
    if not section_body:
        return []
    bullet_items = re.findall(r"(?m)^\s*[-*+]\s+(.*)$", section_body)
    if bullet_items:
        return [item.strip() for item in bullet_items if item.strip()]
    return [line.strip() for line in section_body.splitlines() if line.strip()]


def _first_sentence(text: str) -> str:
    stripped = compact_text(text)
    if not stripped:
        return ""
    match = re.search(r"(?<=[.!?])\s+", stripped)
    if match:
        return stripped[: match.start()].strip()
    return stripped


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        normalized = compact_text(str(value))
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped


def parse_skill_document(
    content: str,
    *,
    skill_id: str,
    version_id: str = "",
    snippet_chars: int = 800,
    max_chars: int = 1_000_000,
) -> dict[str, Any]:
    """Parse one SKILL.md into a stable, JSON-safe field mapping.

    Raises ``SKILL_MANAGE_INVALID_SKILL_DOCUMENT`` on a missing/malformed
    frontmatter, missing name or underivable description, and
    ``SKILL_MANAGE_INPUT_TOO_LARGE`` when the canonical content exceeds
    ``max_chars``. Never truncates canonical content silently.

    Args:
        content: Canonical Skill document text.
        skill_id: Database Skill family ID resolved by the caller.
        version_id: Optional source identity used in parsing errors.
        snippet_chars: Maximum rendered snippet length.
        max_chars: Maximum accepted or rendered character count.

    Returns:
        Parsed document fields with the caller-supplied database Skill ID.
    """

    if len(content) > max_chars:
        raise input_too_large(field="content", limit=max_chars, actual=len(content))
    content = normalize_content(content)
    frontmatter, body = extract_frontmatter(content, version_id=version_id)

    name = str(frontmatter.get("name", "")).strip()
    description = str(frontmatter.get("description", "")).strip()
    if not description:
        overview = extract_markdown_section(body, ["Overview", "Summary", "Description"])
        description = compact_text("\n".join(overview)) if overview else ""
    if not description:
        description = _first_sentence(compact_text(body))

    if not name:
        raise invalid_skill_document(version_id=version_id, reason="missing name")
    if not description:
        raise invalid_skill_document(version_id=version_id, reason="cannot derive description")

    metadata = frontmatter.get(METADATA_KEY) or {}
    if not isinstance(metadata, dict):
        metadata = {METADATA_VALUE: metadata}

    inputs = normalize_string_list(frontmatter.get("inputs") or frontmatter.get("input"))
    outputs = normalize_string_list(frontmatter.get("outputs") or frontmatter.get("output"))
    if not inputs:
        inputs = extract_markdown_section(body, ["Inputs", "Input", "Requirements", "Arguments"])
    if not outputs:
        outputs = extract_markdown_section(body, ["Outputs", "Output", "Returns", "Results"])

    compatibility = normalize_string_list(frontmatter.get(METADATA_COMPATIBILITY))
    allowed_tools = normalize_string_list(frontmatter.get("allowed-tools") or frontmatter.get(METADATA_ALLOWED_TOOLS))

    one_line_capability = compact_text(
        str(
            frontmatter.get("one_line_capability")
            or frontmatter.get("capability")
            or frontmatter.get(METADATA_SUMMARY)
            or metadata.get(METADATA_SUMMARY)
            or _first_sentence(description)
        )
    )

    domain_tags = _dedupe_preserve_order(
        normalize_string_list(frontmatter.get(METADATA_DOMAIN))
        + normalize_string_list(frontmatter.get(METADATA_DOMAINS))
        + normalize_string_list(frontmatter.get(METADATA_TAGS))
        + normalize_string_list(frontmatter.get("category"))
        + normalize_string_list(frontmatter.get("categories"))
        + normalize_string_list(metadata.get(METADATA_DOMAIN))
        + normalize_string_list(metadata.get(METADATA_DOMAINS))
        + normalize_string_list(metadata.get(METADATA_TAGS))
        + extract_markdown_section(body, ["Domain", "Domains", "Tags", "Category"])
    )

    tooling = _dedupe_preserve_order(
        normalize_string_list(frontmatter.get(METADATA_TOOLING))
        + normalize_string_list(frontmatter.get(METADATA_LIBRARIES))
        + normalize_string_list(frontmatter.get("dependencies"))
        + normalize_string_list(metadata.get(METADATA_TOOLING))
        + normalize_string_list(metadata.get(METADATA_LIBRARIES))
        + allowed_tools
        + extract_markdown_section(body, ["Tools", "Tooling", "Libraries", "Dependencies"])
    )

    example_tasks = _dedupe_preserve_order(
        normalize_string_list(frontmatter.get(METADATA_EXAMPLES))
        + normalize_string_list(frontmatter.get(METADATA_EXAMPLE_TASKS))
        + normalize_string_list(metadata.get(METADATA_EXAMPLES))
        + normalize_string_list(metadata.get(METADATA_EXAMPLE_TASKS))
        + extract_markdown_section(body, ["Examples", "Example Tasks", "Use When", "Use Cases"])
    )

    rendered_snippet = build_rendered_snippet(
        name=name,
        description=description,
        body=body,
        compatibility=compatibility,
        allowed_tools=allowed_tools,
        max_chars=snippet_chars,
    )

    return {
        "name": name,
        "description": description,
        "one_line_capability": one_line_capability,
        "body": body,
        "inputs": inputs,
        "outputs": outputs,
        "domain_tags": domain_tags,
        METADATA_TOOLING: tooling,
        METADATA_EXAMPLE_TASKS: example_tasks,
        METADATA_COMPATIBILITY: compatibility,
        METADATA_ALLOWED_TOOLS: allowed_tools,
        METADATA_KEY: metadata,
        "rendered_snippet": rendered_snippet,
        "skill_id": skill_id,
    }


__all__ = [
    "StrictYamlLoader",
    "build_rendered_snippet",
    "compact_text",
    "extract_frontmatter",
    "extract_markdown_section",
    "normalize_content",
    "normalize_string_list",
    "parse_skill_document",
]
