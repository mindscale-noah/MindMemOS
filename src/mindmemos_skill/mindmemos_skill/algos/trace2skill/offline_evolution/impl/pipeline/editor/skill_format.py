"""The shared parser and sole final-export validity check for one Skill."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlsplit

import yaml

_NAME_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_MARKDOWN_LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
_RESOURCE_PATH_RE = re.compile(r"(?<![A-Za-z0-9_./-])(?:scripts|references|assets)/[A-Za-z0-9][A-Za-z0-9._+@/-]*")
_ALLOWED_FRONTMATTER_KEYS = {
    "name",
    "description",
    "license",
    "allowed-tools",
    "metadata",
    "compatibility",
}


@dataclass(frozen=True)
class SkillDocument:
    """The three progressively disclosed parts of a ``SKILL.md`` file."""

    name: str
    description: str
    body: str
    body_start_line: int
    errors: tuple[str, ...]


def parse_skill_md(text: str) -> SkillDocument:
    """Parse discovery metadata and body without validating supporting files."""
    errors: list[str] = []
    if not text.startswith("---"):
        return SkillDocument("", "", text, 1, ("SKILL.md frontmatter is missing",))

    lines = text.splitlines()
    closing = next(
        (index for index in range(1, len(lines)) if lines[index].strip() == "---"),
        None,
    )
    if closing is None:
        return SkillDocument(
            "",
            "",
            "",
            len(lines) + 1,
            ("SKILL.md frontmatter is not closed",),
        )

    frontmatter = "\n".join(lines[1:closing])
    body = "\n".join(lines[closing + 1 :])
    try:
        loaded = yaml.safe_load(frontmatter)
    except yaml.YAMLError as exc:
        loaded = None
        errors.append(f"SKILL.md frontmatter is invalid YAML: {exc}")
    if loaded is None:
        metadata: dict[object, object] = {}
    elif isinstance(loaded, dict):
        metadata = loaded
    else:
        metadata = {}
        errors.append("SKILL.md frontmatter must be a YAML mapping")

    unexpected = sorted(str(key) for key in metadata if key not in _ALLOWED_FRONTMATTER_KEYS)
    if unexpected:
        errors.append("SKILL.md frontmatter contains unexpected field(s): " + ", ".join(unexpected))

    raw_name = metadata.get("name")
    raw_description = metadata.get("description")
    name = ""
    if raw_name is None:
        errors.append("SKILL.md name is missing or empty")
    elif not isinstance(raw_name, str):
        errors.append("SKILL.md name must be a string")
    else:
        name = raw_name.strip()
        if not name:
            errors.append("SKILL.md name is missing or empty")
        elif len(name) > 63:
            errors.append("SKILL.md name must not exceed 63 characters")
        elif _NAME_RE.fullmatch(name) is None:
            errors.append("SKILL.md name is not a valid lowercase hyphenated identifier")

    description = ""
    if raw_description is None:
        errors.append("SKILL.md description is missing or empty")
    elif not isinstance(raw_description, str):
        errors.append("SKILL.md description must be a string")
    else:
        description = raw_description.strip()
        if not description:
            errors.append("SKILL.md description is missing or empty")
        else:
            if "<" in description or ">" in description:
                errors.append("SKILL.md description must not contain angle brackets")
            if len(description) > 1024:
                errors.append("SKILL.md description must not exceed 1024 characters")

    compatibility = metadata.get("compatibility")
    if compatibility:
        if not isinstance(compatibility, str):
            errors.append("SKILL.md compatibility must be a string")
        elif len(compatibility) > 500:
            errors.append("SKILL.md compatibility must not exceed 500 characters")
    if not body.strip():
        errors.append("SKILL.md body is missing or empty")
    return SkillDocument(
        name,
        description,
        body,
        closing + 2,
        tuple(errors),
    )


def _markdown_target(raw: str) -> str:
    value = raw.strip()
    if value.startswith("<") and ">" in value:
        return value[1 : value.index(">")].strip()
    return value.split(maxsplit=1)[0] if value else ""


def _resource_references(body: str) -> set[str]:
    references = {match.group(0).rstrip(".,;:") for match in _RESOURCE_PATH_RE.finditer(body)}
    for match in _MARKDOWN_LINK_RE.finditer(body):
        target = _markdown_target(match.group(1))
        if target:
            references.add(target)
    return references


def skill_resource_reference_errors(skill_dir: str | Path, body: str) -> list[str]:
    """Check the fixed point of explicitly disclosed local resources."""
    root = Path(skill_dir).resolve()
    errors: list[str] = []
    pending: list[tuple[str, str]] = [("SKILL.md", body)]
    visited: set[str] = set()
    while pending:
        source_name, source_body = pending.pop(0)
        if source_name in visited:
            continue
        visited.add(source_name)
        for raw in sorted(_resource_references(source_body)):
            parsed = urlsplit(raw)
            if parsed.scheme or parsed.netloc or raw.startswith("#"):
                continue
            local = unquote(parsed.path).strip()
            if not local:
                continue
            relative = Path(local)
            if relative.is_absolute() or ".." in relative.parts:
                errors.append(f"{source_name} resource reference escapes Skill directory: {raw}")
                continue
            target = (root / relative).resolve()
            try:
                target.relative_to(root)
            except ValueError:
                errors.append(f"{source_name} resource reference escapes Skill directory: {raw}")
                continue
            if not target.exists():
                errors.append(f"{source_name} referenced resource is missing: {raw}")
                continue
            if target.is_file() and target.suffix.lower() == ".md":
                target_name = target.relative_to(root).as_posix()
                if target_name in visited:
                    continue
                try:
                    nested_body = target.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    errors.append(f"{source_name} referenced Markdown is not UTF-8: {raw}")
                    continue
                pending.append((target_name, nested_body))
    return errors


def skill_format_errors(skill_dir: str | Path) -> list[str]:
    """Check progressive-disclosure structure and explicit local resources."""
    root = Path(skill_dir)
    skill_md = root / "SKILL.md"
    if not skill_md.is_file():
        return ["SKILL.md is missing"]
    try:
        text = skill_md.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return ["SKILL.md is not UTF-8"]
    document = parse_skill_md(text)
    errors = list(document.errors)
    if document.name and document.name != root.name:
        errors.append("SKILL.md name does not match its directory")
    errors.extend(skill_resource_reference_errors(root, document.body))
    return errors


__all__ = [
    "SkillDocument",
    "parse_skill_md",
    "skill_format_errors",
    "skill_resource_reference_errors",
]
