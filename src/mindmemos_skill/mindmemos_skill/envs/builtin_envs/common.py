"""Small shared helpers for local benchmark files and text-agent rollouts."""

import json
from pathlib import Path

from ...agents.typing import AgentResult
from ...skill import Skill, SkillContent
from ..typing import Usage


def load_records(path: Path) -> list[dict]:
    """Read a benchmark JSON array, keyed object, or JSONL file.

    Args:
        path: Local dataset file.

    Returns:
        Source records in source order.
    """
    content = path.read_text(encoding="utf-8")
    try:
        records = json.loads(content)
    except json.JSONDecodeError:
        records = [json.loads(line) for line in content.splitlines() if line.strip()]
    if isinstance(records, dict):
        records = [records] if "id" in records or "uid" in records else records.get("data") or list(records.values())
    if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
        raise ValueError(f"Expected benchmark records in {path}")
    if not records:
        raise ValueError(f"No benchmark records in {path}")
    return records


def skill_text(skills: list[Skill]) -> str:
    """Render ordered prepared Skills without exposing evaluation references.

    Args:
        skills: Complete Skill copies whose documents are already assembled.

    Returns:
        Readable skill headings and contents, or an empty string.
    """
    return "\n\n".join(skill.content for skill in skills)


def export_skills(skills: list[Skill], directory: Path) -> str:
    """Export raw Skill text and resources under one benchmark entrypoint.

    Args:
        skills: Complete task-time Skill copies in agent-visible order.
        directory: Agent-accessible Skill directory.

    Returns:
        Original entrypoint texts joined with blank lines, without added instructions.

    Raises:
        ValueError: Resource paths conflict across Skills.
    """
    if not skills:
        return ""
    resources = {}
    for skill in skills:
        for path, content in skill.resources.items():
            if path in resources and resources[path] != content:
                raise ValueError(f"Conflicting Skill resource: {path}")
            resources[path] = content
    text = skill_text(skills)
    source = SkillContent(
        content=text,
        resources=resources,
        name="assembled",
        description="Task-selected instructions",
        standard_format=False,
    )
    # Benchmark inputs already occupy this workspace; install only Skill files.
    directory = directory.absolute()
    files = {"SKILL.md": source.content.encode("utf-8"), **source.resources}
    for name in files:
        target = directory / name
        if any(path.is_symlink() for path in (target, *target.parents)):
            raise ValueError("Skill workspace destinations must not contain symbolic links")
        if target.exists() and not target.is_file():
            raise ValueError(f"Skill workspace destination is not a file: {name}")
    for name, content in files.items():
        target = directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    return text


def total_usage(responses: list[AgentResult]) -> Usage:
    """Aggregate token totals while retaining every provider usage payload.

    Args:
        responses: All model calls, including refinement calls.

    Returns:
        Normalized totals and unmodified per-call usage details.
    """
    return Usage(
        input_tokens=sum(int(response.usage.get("input_tokens", 0)) for response in responses),
        output_tokens=sum(int(response.usage.get("output_tokens", 0)) for response in responses),
        total_tokens=sum(int(response.usage.get("total_tokens", 0)) for response in responses),
        reported=all(bool(response.usage) for response in responses),
        calls=[response.usage for response in responses],
    )
